# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Photoreal background renderers for hybrid (sim + photoreal) rendering.

A :class:`BackgroundRenderer` maps a camera (intrinsics + world-from-camera
pose + image size) to an ``(rgb, depth)`` pair that
:class:`strands_robots.rendering.HybridCompositor` blends behind a simulation
backend's foreground render. Two implementations ship with the library:

* :class:`PanoramaBackground` -- equirectangular HDRI / panorama lookup. No ML
  deps. Treats the panorama as a sphere at infinity (``depth = zfar``), so
  every foreground pixel "wins" the depth test. If no image path is given a
  procedural kitchen-ish gradient is generated so demos run anywhere.

* :class:`GsplatBackground` -- true 3D Gaussian Splatting via the ``gsplat``
  library, behind the ``sim-gs`` extra (CUDA-only in practice). Produces real
  per-pixel depth, so foreground objects correctly occlude / are occluded by
  GS geometry. Use :func:`gsplat_rasterizer_available` to probe whether the
  installed ``gsplat`` can actually CUDA-rasterize before relying on it.

Both implement the :class:`BackgroundRenderer` protocol -- drop-in pluggable.
The renderers are backend-agnostic: they only consume
:class:`~strands_robots.rendering.camera.CameraParams` (which every
``SimEngine.get_camera_params`` produces in the same OpenGL optical frame).

Scope note: assets that carry higher-order spherical harmonics render with
full view-dependent color (the SH coefficients are evaluated per-view by
``gsplat``); DC-only assets take a baked-RGB fast path. No relighting.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from strands_robots.utils import boolean_flag_error, finite_number_error, require_optional

from .camera import CameraParams

logger = logging.getLogger(__name__)

_GSPLAT_PURPOSE = "3D Gaussian Splatting backgrounds (GsplatBackground)"


class BackgroundRenderer(Protocol):
    """Render a photoreal background at a given camera pose.

    Implementations should be deterministic for a fixed camera and idempotent
    across calls (the compositor will call this every frame).
    """

    name: str

    def render(self, cam: CameraParams) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rgb_uint8, depth_metric_float32)`` for ``cam``.

        Args:
            cam: pinhole camera parameters at the desired image size.

        Returns:
            ``rgb`` as ``(H, W, 3) uint8`` and ``depth`` as ``(H, W) float32``
            in meters, where ``(H, W)`` is exactly
            ``(cam.height, cam.width)`` - the composited frame reports ``cam``,
            so a layer at any other size is refused by
            :meth:`~strands_robots.rendering.HybridCompositor.render` rather
            than truncated into a frame ``cam.K`` does not describe. Pixels at
            "infinity" should report ``depth = np.inf`` (or any value larger
            than ``cam.zfar``) so the compositor's depth test always picks the
            foreground.
        """


# --------------------------------------------------------------------------- #
# Panorama (equirectangular) background
# --------------------------------------------------------------------------- #


class PanoramaBackground:
    """Equirectangular sky/scene panorama as the photoreal background.

    The panorama is interpreted as the inside of a unit sphere centred on the
    camera. For each output pixel we cast a ray from the camera into the
    world, normalise it, and look up the corresponding texel using
    :math:`(\\theta, \\phi)` spherical coords. Depth is fixed at ``cam.zfar``
    so MuJoCo geometry always occludes correctly.

    This is *not* a 3DGS scene -- there is no parallax as the camera moves --
    but it ships zero ML deps and gives the demo a believable photoreal
    backdrop on day 0. Swap in :class:`GsplatBackground` once you have a
    trained scene.

    Args:
        image_path: optional path to an equirectangular ``.jpg``/``.png``.
            If ``None`` a procedural gradient (sky + warm-tone "kitchen wall"
            + floor) is generated so the example runs without external assets.
        rotation_deg: yaw rotation of the panorama in degrees (rotates around
            the world +Z axis). Useful for aligning the panorama with the
            scene without re-rendering it.

    Raises:
        ValueError: if ``rotation_deg`` is not a finite number. The yaw builds
            the rotation matrix every output ray is turned by, so a non-finite
            angle turns every ray into ``nan`` and the panorama samples nothing
            -- the backdrop renders black rather than rotated, and nothing
            raises to trigger a caller's fallback.
    """

    name = "panorama"

    def __init__(
        self,
        image_path: str | Path | None = None,
        rotation_deg: float = 0.0,
    ) -> None:
        self._image_path = Path(image_path) if image_path else None
        # Checked before the coercion, on the shared signed-finite domain, for
        # the reason :func:`~strands_robots.rendering.compositor._shadow_plane_z_error`
        # gives for a plane height: only a finite number can be honored. The
        # yaw builds the ``Rz`` every world ray is turned by, so ``nan``/``inf``
        # turned every direction into ``nan``, the equirect lookup sampled
        # nothing, and the backdrop came back uniformly black -- with no
        # exception, so a caller's photoreal-to-procedural fallback never fired.
        # A signed domain because both senses of yaw are legitimate and a
        # rotation wraps, so there is no floor or ceiling to impose.
        if text := finite_number_error(rotation_deg, "rotation_deg", "PanoramaBackground"):
            raise ValueError(text)
        self._rotation_rad = float(np.deg2rad(rotation_deg))
        self._panorama: np.ndarray | None = None

    # ----- panorama loading ----- #

    def _ensure_panorama(self) -> np.ndarray:
        if self._panorama is not None:
            return self._panorama
        if self._image_path is not None and self._image_path.exists():
            from PIL import Image

            pano = np.array(Image.open(self._image_path).convert("RGB"))
            logger.info("PanoramaBackground: loaded %s (%s)", self._image_path, pano.shape)
        else:
            if self._image_path is not None:
                logger.warning(
                    "Panorama path %s does not exist -- falling back to procedural panorama.",
                    self._image_path,
                )
            pano = _make_procedural_kitchen_panorama(width=2048, height=1024)
        self._panorama = pano
        return pano

    # ----- BackgroundRenderer interface ----- #

    def render(self, cam: CameraParams) -> tuple[np.ndarray, np.ndarray]:
        """Sample the equirectangular panorama along ``cam``'s per-pixel rays.

        Casts one world-space ray per output pixel, maps each ray direction to
        spherical ``(theta, phi)`` coordinates, applies the configured yaw
        rotation, and bilinearly samples the panorama texture. Depth is a
        constant ``cam.zfar`` plane so any MuJoCo foreground always wins the
        compositor's depth test.

        Args:
            cam: pinhole camera parameters at the desired image size.

        Returns:
            ``(rgb, depth)`` with ``rgb`` as ``(H, W, 3) uint8`` and ``depth``
            as ``(H, W) float32`` filled with ``cam.zfar`` meters.
        """
        pano = self._ensure_panorama()  # (Hp, Wp, 3) uint8
        H, W = cam.height, cam.width

        # Per-pixel camera-frame direction. MuJoCo / OpenGL convention:
        # +X right, +Y up, -Z forward.
        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        Kinv = np.linalg.inv(cam.K)
        homo = np.stack([u, v, np.ones_like(u)], axis=-1)  # (H, W, 3)
        dirs_cam = homo @ Kinv.T  # (H, W, 3) -- image-plane rays
        # Flip Z to match GL's "-Z forward" convention.
        dirs_cam[..., 2] *= -1.0
        dirs_cam[..., 1] *= -1.0  # image v grows down -> world up flips
        dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True) + 1e-12

        # World-frame directions = R_world_cam @ dirs_cam.
        R = cam.T_world_cam[:3, :3]
        dirs_world = dirs_cam @ R.T  # (H, W, 3)

        # Optional yaw rotation around world +Z (lets users spin the pano
        # without recomputing the texture).
        if self._rotation_rad != 0.0:
            c, s = np.cos(self._rotation_rad), np.sin(self._rotation_rad)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            dirs_world = dirs_world @ Rz.T

        # Spherical mapping. World convention: +Z up, atan2 in XY plane.
        x, y, z = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]
        theta = np.arctan2(y, x)  # in [-pi, pi]   azimuth
        phi = np.arcsin(np.clip(z, -1.0, 1.0))  # in [-pi/2, pi/2]   elevation

        Hp, Wp, _ = pano.shape
        # Equirectangular UV: u in [0, 1] left->right (theta), v in [0, 1] top->bottom (phi).
        uu = (theta + np.pi) / (2.0 * np.pi)
        vv = 0.5 - phi / np.pi
        # Bilinear lookup.
        rgb = _bilinear_sample(pano, uu, vv)
        depth = np.full((H, W), cam.zfar, dtype=np.float32)
        return rgb, depth


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample an equirectangular image at normalised ``(u, v)``.

    ``u`` wraps around the seam; ``v`` is clamped at the poles.
    """
    Hp, Wp, _ = image.shape
    # Wrap u, clamp v.
    u = np.mod(u, 1.0)
    v = np.clip(v, 0.0, 1.0)
    fx = u * (Wp - 1)
    fy = v * (Hp - 1)
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    x1 = (x0 + 1) % Wp
    y1 = np.clip(y0 + 1, 0, Hp - 1)
    wx = fx - x0
    wy = fy - y0

    img = image.astype(np.float32)
    p00 = img[y0, x0]
    p01 = img[y0, x1]
    p10 = img[y1, x0]
    p11 = img[y1, x1]
    out = (
        p00 * ((1 - wx) * (1 - wy))[..., None]
        + p01 * (wx * (1 - wy))[..., None]
        + p10 * ((1 - wx) * wy)[..., None]
        + p11 * (wx * wy)[..., None]
    )
    result: np.ndarray = np.clip(out, 0, 255).astype(np.uint8)
    return result


def _make_procedural_kitchen_panorama(width: int = 2048, height: int = 1024) -> np.ndarray:
    """Generate a procedural "warm kitchen" equirectangular panorama.

    No external assets required. The vertical layout mirrors a typical indoor
    panorama: blue ceiling at the top, warm wall band in the middle, parquet
    floor at the bottom. We add a soft horizontal gradient and a light source
    to give the cube a sense of room context.
    """
    rng = np.random.default_rng(seed=42)

    # Vertical bands (top -> bottom).
    img = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        t = y / (height - 1)  # 0 at top, 1 at bottom
        if t < 0.30:
            # Ceiling: cool off-white.
            base = np.array([235.0, 240.0, 245.0])
        elif t < 0.55:
            # Upper wall: warm beige.
            base = np.array([218.0, 198.0, 168.0])
        elif t < 0.78:
            # Lower wall / cabinet line.
            base = np.array([198.0, 170.0, 140.0])
        else:
            # Floor: parquet brown.
            base = np.array([130.0, 90.0, 60.0])
        img[y, :, :] = base[None, :]

    # Horizontal soft "window light" lobe.
    x = np.linspace(0, 2 * np.pi, width, endpoint=False)
    light = 1.0 + 0.10 * np.cos(x - np.pi / 2)  # brighter on one wall
    img *= light[None, :, None]

    # Faint vertical falloff towards floor (ambient occlusion-ish).
    falloff = np.linspace(1.05, 0.85, height)[:, None, None]
    img *= falloff

    # Speckle / noise so the texture isn't dead-flat (helps you tell the
    # background apart from the MuJoCo render).
    img += rng.normal(0, 4.0, size=img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Optional 3D Gaussian Splatting background (soft dep on `gsplat`, extra `sim-gs`)
# --------------------------------------------------------------------------- #


def gsplat_rasterizer_available() -> tuple[bool, str]:
    """Check that ``gsplat`` can actually CUDA-rasterize, not just import.

    A plain ``pip install gsplat`` is *importable* even when its CUDA
    kernels are unavailable: gsplat JIT-builds them from source on first
    use via ``nvcc``, and many GPU containers ship the CUDA runtime but no
    CUDA *toolkit*. gsplat then disables its CUDA backend ("No CUDA toolkit
    found") and the first :func:`gsplat.rasterization` call raises
    ``AttributeError: 'NoneType' object has no attribute 'CameraModelType'``.

    Importing alone can't catch this, so we probe with a trivial
    one-gaussian rasterization. Callers can then fall back to
    :class:`PanoramaBackground` *up front* instead of erroring on every
    render. Install a pre-built ``gsplat`` wheel (ships compiled kernels,
    no nvcc needed) to avoid the JIT path.

    Returns:
        ``(ok, reason)`` -- ``ok`` is ``True`` when a probe rasterization
        succeeded; ``reason`` explains any failure.
    """
    try:
        import torch
        from gsplat import rasterization
    except Exception as exc:  # noqa: BLE001 -- capability probe: ANY import failure means "unavailable"
        return False, f"gsplat/torch not importable ({type(exc).__name__}: {exc})"
    if not torch.cuda.is_available():
        return False, "no CUDA device available to torch"
    try:
        dev = "cuda"
        means = torch.tensor([[0.0, 0.0, 2.0]], device=dev)
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
        scales = torch.full((1, 3), 0.1, device=dev)
        opacities = torch.ones(1, device=dev)
        colors = torch.ones(1, 3, device=dev)
        viewmats = torch.eye(4, device=dev)[None]
        Ks = torch.tensor([[[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]]], device=dev)
        rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width=16, height=16)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 -- capability probe: a disabled CUDA backend surfaces here
        return False, f"gsplat CUDA rasterizer unavailable ({type(exc).__name__}: {exc})"


class GsplatBackground:
    """Real 3D Gaussian Splatting background using the `gsplat` library.

    This is the upgrade path from :class:`PanoramaBackground` once you have a
    trained 3DGS scene (e.g. exported from Nerfstudio, Polycam, or World Labs
    Marble as a ``.ply``). ``.spz`` (Niantic) scene files are also supported
    natively via a pure-numpy reader.

    Install:

        pip install 'strands-robots[sim-gs]'

    Note that ``gsplat`` JIT-compiles its CUDA kernels with ``nvcc`` on first
    use when installed from source; call :func:`gsplat_rasterizer_available`
    to verify an install can actually rasterize.

    Usage:

        bg = GsplatBackground(ply_path="scenes/kitchen.ply")
        compositor = HybridCompositor(sim, background=bg)

    Notes:
        * ``gsplat`` rasterises in *batch* (B, H, W, 3); we run B=1 every
          frame and convert to numpy on the way out, so this is not as fast as
          a JS-side sparkjs viewer but plenty fine for an offline demo.
        * Depth is read from gsplat's accumulated alpha-weighted Z, then
          divided through to give metric depth. Empty pixels (no Gaussians
          along the ray) report ``cam.zfar`` so the compositor falls through
          to whatever fallback you pass.
        * You'll likely want to align the GS scene to your MuJoCo world frame
          via the ``transform`` kwarg (4x4 SE(3) ``world_from_gs``).

    Raises:
        FileNotFoundError: if ``ply_path`` names no existing file.
        PermissionError: if ``ply_path`` exists but is not readable.
        ValueError: if ``auto_backdrop``, ``skybox``, ``metric`` or
            ``own_floor`` is not a ``bool``. Each selects an alignment or
            compositing branch, so a value read by truthiness picks a branch
            instead of being refused - and every spelling of *off* a caller
            reaches for is a truthy string, so it picks the branch it asks to
            skip.
    """

    name = "gsplat"

    def __init__(
        self,
        ply_path: str | Path,
        device: str = "cuda",
        transform: np.ndarray | None = None,
        auto_backdrop: bool = False,
        backdrop_center: tuple | None = (0.05, 0.05, 0.25),
        backdrop_radius: float = 3.0,
        skybox: bool = False,
        up_sign: float | None = None,
        yaw_deg: float = 0.0,
        radius: float = 2.5,
        center: tuple = (0.05, 0.05),
        floor_z: float = -0.3,
        clip_below: float | None = 0.0,
        min_opacity: float = 0.25,
        floor_pct: float = 2.0,
        metric: bool = False,
        bg_fill: tuple | None = None,
        own_floor: bool = False,
        up_axis: tuple | None = None,
        major_axis: tuple | None = None,
    ) -> None:
        self._ply_path = Path(ply_path)
        # Validate the scene file where the caller supplied it (issue #2321).
        # The heavy work (torch/gsplat imports, decoding the splats) stays
        # lazy in ``_load`` so construction is still cheap and CPU-safe, but a
        # wrong path is a configuration error: raising here surfaces it at the
        # construction site instead of at the first ``render()`` -- which in
        # app contexts sits inside a catch-all that demotes the photoreal
        # background to a procedural fallback.
        if not self._ply_path.is_file():
            raise FileNotFoundError(
                f"Gaussian Splat not found: {self._ply_path}. "
                "GsplatBackground requires an existing .ply/.spz scene file."
            )
        if not os.access(self._ply_path, os.R_OK):
            raise PermissionError(f"Gaussian Splat is not readable: {self._ply_path}")
        # Each of these selects an alignment or compositing branch, so it is
        # checked on the shared boolean domain beside the path above rather than
        # read by truthiness where the branch is taken. Truthiness inverts
        # exactly the spellings an operator reaches for: ``skybox="false"``,
        # ``metric="no"`` and ``own_floor="off"`` each selected the branch the
        # value asks to skip, and nothing raised or logged. ``metric`` is the
        # sharpest of the four because it also decides whether ``radius`` is
        # read at all: a truthy string kept the capture's raw scale and dropped
        # the requested one, so a scene stood up at whatever size it was
        # captured at. The undeclared falsy values are the other half - ``0``,
        # ``""``, ``[]`` and ``None`` took the default branch without being a
        # spelling of it, and ``skybox``/``auto_backdrop`` composed with
        # ``transform is None``, so the attribute held that raw value rather
        # than a bool. ``HybridCompositor`` checks its own ``blend_in_linear``
        # for the same reason: every such option "was previously coerced or
        # clamped into a plausible-but-different render".
        for _param, _value in (
            ("auto_backdrop", auto_backdrop),
            ("skybox", skybox),
            ("metric", metric),
            ("own_floor", own_floor),
        ):
            if text := boolean_flag_error(_value, _param, "GsplatBackground"):
                raise ValueError(text)
        # The alignment *numbers*, on the shared signed-finite domain, beside
        # the posture flags above. The flags decide which branch the fit takes;
        # these are the quantities that branch scales and offsets by, and each
        # was coerced with a bare ``float()`` that accepts ``nan``/``inf``. A
        # non-finite value here poisons the whole ``world_from_gs``: the fitted
        # 4x4 comes back with non-finite cells, so every gaussian is placed
        # nowhere and the photoreal scene silently does not appear. ``bool`` is
        # rejected for the reason the flags' own domain gives in reverse -- a
        # truth value is not a length, an angle or a fraction. Signed because a
        # yaw, a floor height and an up-sign are all legitimately negative; the
        # domain constrains finiteness only, so no bound is imposed here.
        for _number_param, _number in (
            ("backdrop_radius", backdrop_radius),
            ("yaw_deg", yaw_deg),
            ("radius", radius),
            ("floor_z", floor_z),
            ("min_opacity", min_opacity),
            ("floor_pct", floor_pct),
        ):
            if text := finite_number_error(_number, _number_param, "GsplatBackground"):
                raise ValueError(text)
        # ``up_sign`` (``None`` = auto-detect from PCA) and ``clip_below``
        # (``None`` = drop nothing) carry a sentinel, so the domain applies to
        # the supplied number and the sentinel passes through untouched.
        for _opt_param, _opt in (("up_sign", up_sign), ("clip_below", clip_below)):
            if _opt is not None and (text := finite_number_error(_opt, _opt_param, "GsplatBackground")):
                raise ValueError(text)
        self._device = device
        self._explicit_transform = transform is not None
        self._transform = np.asarray(transform, dtype=np.float64) if transform is not None else np.eye(4)
        # When True and no explicit transform is given, fit a ``world_from_gs``
        # that stands the captured scene upright, scales it to ~``backdrop_radius``
        # metres, and centres it on ``backdrop_center`` -- so it reads as a
        # photoreal room *around/behind* the arm (the arm + cube + MuJoCo
        # ground composite in front via depth). Alignment is approximate.
        self._auto_backdrop = auto_backdrop and transform is None
        self._backdrop_center = np.asarray(backdrop_center, dtype=np.float64)
        self._backdrop_radius = float(backdrop_radius)
        # ``skybox`` mode: the validated "live backdrop" recipe. Stand the scene
        # upright (PCA up x ``up_sign``), scale, push the GS floor to ``floor_z``
        # (below the MuJoCo ground so the MuJoCo floor owns everything below the
        # horizon -- no floor-fight, no nadir void), then drop sub-floor gaussians
        # (``clip_below``, world-z) and low-opacity floaters (``min_opacity``).
        # ``up_sign`` is per-scene (see ``GSPLAT_SKYBOX_ALIGN``); ``None`` =
        # best-effort auto-detect (good for curated presets, rough for uploads).
        self._skybox = skybox and transform is None
        self._up_sign = up_sign
        self._yaw_deg = float(yaw_deg)
        self._radius = float(radius)
        self._center = tuple(center)
        self._floor_z = float(floor_z)
        self._clip_below = clip_below
        self._min_opacity = float(min_opacity)
        self._floor_pct = float(floor_pct)
        self._metric = bool(metric)
        # Explicit up / in-plane axis (GS frame) -- overrides PCA when the asset
        # has a known convention (e.g. a Y-up .spz). None => PCA-estimate.
        self._up_axis = tuple(up_axis) if up_axis is not None else None
        self._major_axis = tuple(major_axis) if major_axis is not None else None
        # Fill colour (0..255 RGB) shown where the splat has no coverage (the
        # capture's unobserved zenith/edges), so voids read as a plain
        # ceiling/sky instead of black holes. Defaults to a neutral light grey
        # in skybox mode (cf. MuJoCo-GS-Web's dark-grey scene background).
        if bg_fill is None:
            bg_fill = (188, 188, 192) if self._skybox else (0, 0, 0)
        self._bg_fill = np.asarray(bg_fill, dtype=np.float32).reshape(3)
        # When True, this background supplies its own (photoreal) floor, so the
        # compositor hides the MuJoCo grid ground. The arm + cube then rest on
        # the GS surface (collision still comes from the hidden MuJoCo floor).
        self.own_floor = bool(own_floor)
        self._splats: dict[str, Any] | None = None  # lazily loaded
        # Set at load time: ``.spz`` assets trained with mip-splatting carry an
        # antialiased flag (header bit) and must be rasterized with the matching
        # AA opacity compensation, not the classic mode.
        self._rasterize_mode: str = "classic"

    # ----- lazy load ----- #

    def _load(self) -> None:
        require_optional("torch", extra="sim-gs", purpose=_GSPLAT_PURPOSE)
        require_optional("gsplat", extra="sim-gs", purpose=_GSPLAT_PURPOSE)
        import torch

        # Fail loud and early on a CUDA mismatch -- the `sim-gs` extra pins
        # torch without a CUDA constraint, so a CPU-only (or wrong-CUDA-build)
        # install pip-installs fine, then would otherwise die with a generic
        # "no CUDA-capable device" deep inside `gsplat.rasterization` on the
        # first frame.
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "GsplatBackground(device='cuda') was requested but "
                "torch.cuda.is_available() is False. Install a CUDA-matched "
                "torch build (the gsplat backdrop needs a GPU), or use "
                "PanoramaBackground on CPU-only hosts."
            )
        if not self._ply_path.exists():
            raise FileNotFoundError(f"Gaussian Splat not found: {self._ply_path}")
        if self._ply_path.suffix.lower() == ".spz":
            self._splats = _load_spz_splats(self._ply_path, device=self._device)
        else:
            self._splats = _load_ply_splats(self._ply_path, device=self._device)
        # Honor the SPZ antialiased training flag (header bit 0): opacities
        # trained WITH the mip-splatting compensation must be rendered with
        # gsplat's "antialiased" rasterize mode, or partial-coverage regions
        # come out too opaque. Popped here so ``self._splats`` stays
        # tensors-only (``_clip_splats`` indexes every value by mask).
        self._rasterize_mode = "antialiased" if self._splats.pop("antialiased", False) else "classic"
        if self._skybox:
            means = self._splats["means"].detach().cpu().numpy()
            up_sign = self._up_sign if self._up_sign is not None else _auto_up_sign(means)
            self._transform = _fit_skybox_transform(
                means,
                up_sign=up_sign,
                yaw_deg=self._yaw_deg,
                radius=self._radius,
                center=self._center,
                floor_z=self._floor_z,
                floor_pct=self._floor_pct,
                metric=self._metric,
                up_axis=self._up_axis,
                major_axis=self._major_axis,
            )
            if self._clip_below is not None:
                kept, total = self._clip_splats(self._clip_below, self._min_opacity)
                logger.info(
                    "GsplatBackground: skybox align (up_sign=%+.0f) + clip -> kept %d/%d gaussians for %s",
                    up_sign,
                    kept,
                    total,
                    self._ply_path.name,
                )
        elif self._auto_backdrop:
            means = self._splats["means"].detach().cpu().numpy()
            self._transform = _fit_backdrop_transform(means, self._backdrop_center, self._backdrop_radius)
            logger.info("GsplatBackground: fitted backdrop transform for %s", self._ply_path.name)

    def _clip_splats(self, clip_below: float, min_opacity: float) -> tuple[int, int]:
        """Drop gaussians below ``clip_below`` (world-z, after ``self._transform``)
        and low-opacity floaters. Returns ``(kept, total)``."""
        import torch

        s = self._splats
        assert s is not None
        means = s["means"]
        M = torch.from_numpy(self._transform[:3, :3]).float().to(means.device)
        b = torch.from_numpy(self._transform[:3, 3]).float().to(means.device)
        keep = (means @ M.T + b)[:, 2] >= float(clip_below)
        if min_opacity > 0:
            keep = keep & (s["opacities"].reshape(-1) >= float(min_opacity))
        total = int(keep.numel())
        self._splats = {k: v[keep] for k, v in s.items()}
        return int(keep.sum()), total

    # ----- BackgroundRenderer interface ----- #

    def render(self, cam: CameraParams) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize the loaded 3D Gaussian Splatting scene from ``cam``.

        Lazily loads the ``.ply``/``.spz`` splats on first call, builds the
        gaussian->camera view matrix (converting the stored camera->world
        MuJoCo/OpenGL pose and the scene-alignment transform into gsplat's
        OpenCV convention), and rasterizes in ``RGB+D`` mode (``antialiased``
        rasterize mode when the asset was trained with it, e.g. an ``.spz``
        with the AA header flag set; ``classic`` otherwise). gsplat returns
        alpha-premultiplied RGB and accumulated (alpha-weighted) depth: the
        RGB is composited over the neutral background fill with the
        premultiplied over-operator (``rgb + (1 - alpha) * fill``) so
        unobserved regions read as plain sky/ceiling rather than black, the
        depth is divided by the accumulated alpha to give metric depth, and
        zero-contribution pixels are promoted to ``cam.zfar`` depth so they
        lose the depth test against any MuJoCo foreground.

        Args:
            cam: pinhole camera parameters at the desired image size.

        Returns:
            ``(rgb, depth)`` with ``rgb`` as ``(H, W, 3) uint8`` and ``depth``
            as ``(H, W) float32`` in meters (camera frame).

        Raises:
            ImportError: if the ``sim-gs`` extra (``torch`` + ``gsplat``) is
                not installed.
        """
        if self._splats is None:
            self._load()
        import torch
        from gsplat import rasterization

        s = self._splats  # type: ignore[assignment]
        assert s is not None  # for type checker

        # View matrix: gsplat wants world->camera in the OpenCV convention
        # (+X right, +Y down, +Z forward). Our CameraParams.T_world_cam is
        # camera->world in MuJoCo/OpenGL convention (+X right, +Y up, -Z
        # forward), and ``self._transform`` is ``world_from_gs`` (places the
        # gaussians' own frame into the MuJoCo world). So the gaussian->camera
        # transform is:  gl_to_cv * (world<-cam)^-1 * (world<-gs)
        #              =  gl_to_cv * cam_from_world * world_from_gs.
        gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
        viewmat_np = gl_to_cv @ np.linalg.inv(cam.T_world_cam) @ self._transform
        viewmat = torch.from_numpy(viewmat_np).float().unsqueeze(0).to(self._device)
        K = torch.from_numpy(cam.K).float().unsqueeze(0).to(self._device)

        # ``colors`` is either baked per-gaussian RGB ``(N, 3)`` (DC-only
        # asset: fast path, nothing to evaluate) or raw SH coefficients
        # ``(N, K, 3)`` (asset with view-dependent color). gsplat evaluates
        # the coefficients against per-gaussian view directions when
        # ``sh_degree`` is set; the loaders guarantee K == (degree + 1)^2.
        colors = s["colors"]
        sh_degree = int(np.sqrt(colors.shape[1])) - 1 if colors.dim() == 3 else None

        # rasterization returns (render_colors, render_alphas, meta). With
        # render_mode="RGB+D", render_colors is (B, H, W, 4): [..., :3] =
        # alpha-premultiplied RGB (already accumulated over the splats),
        # [..., 3] = ACCUMULATED (alpha-weighted) depth in meters, which must
        # be divided by alpha to give metric depth.
        render_colors, render_alphas, _ = rasterization(
            means=s["means"],
            quats=s["quats"],
            scales=s["scales"],
            opacities=s["opacities"],
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=cam.width,
            height=cam.height,
            near_plane=cam.znear,
            far_plane=cam.zfar,
            render_mode="RGB+D",
            sh_degree=sh_degree,
            rasterize_mode=self._rasterize_mode,
        )
        out = render_colors[0]  # (H, W, 4)
        rgb = out[..., :3].clamp(0, 1).cpu().numpy() * 255.0
        alpha = render_alphas[0, ..., 0].clamp(0, 1).cpu().numpy().astype(np.float32)
        # Composite the splat over a neutral fill, so un-observed regions
        # (zenith/edges) read as a plain ceiling/sky rather than black.
        # gsplat's RGB is already premultiplied by the accumulated alpha, so
        # the over-operator is ``rgb + (1 - alpha) * fill`` -- multiplying by
        # alpha again would weight the splat by alpha^2 and darken every
        # partial-coverage pixel. (No-op when fully covered.)
        rgb = rgb + self._bg_fill[None, None, :] * (1.0 - alpha[..., None])
        rgb_np = np.clip(rgb, 0, 255).astype(np.uint8)
        # Alpha-normalize the accumulated depth to metric depth. Without the
        # division, soft (partial-alpha) regions report depth biased toward
        # the camera by exactly their alpha, and the compositor's z-test then
        # wrongly occludes real foreground pixels.
        accum_depth = out[..., 3].cpu().numpy().astype(np.float32)
        depth_np = accum_depth / np.maximum(alpha, 1e-6)
        # Pixels with (essentially) no gaussian contribution are promoted to
        # zfar so they lose the depth test against any MuJoCo foreground. The
        # emptiness test keys on alpha, not the raw depth: after the division
        # an alpha~0 pixel would otherwise report ``accum/eps`` -- an
        # arbitrary near-camera phantom occluder.
        depth_np = np.where((alpha < 1e-4) | (depth_np <= cam.znear), cam.zfar, depth_np)
        return rgb_np, depth_np


# --------------------------------------------------------------------------- #
# Downloadable 3DGS scene presets (like MuJoCo-GS-Web's scene gallery)
# --------------------------------------------------------------------------- #

# Real trained 3DGS scenes hosted on HuggingFace (standard INRIA .ply layout).
# ``bonsai`` is an indoor plant-on-a-table room ~= a "tabletop" scene; the
# others are outdoor. Users can also upload their own .ply (e.g. a World Labs
# Marble export re-saved as .ply). Per-preset provenance (source, training
# iteration, license) lives in :data:`GSPLAT_SCENE_PROVENANCE`.
GSPLAT_SCENES = {
    "tabletop (indoor room)": (
        "https://raw.githubusercontent.com/Vector-Wangel/MuJoCo-GS-Web/main/assets/environments/tabletop/scene.spz"
    ),
    "bonsai (indoor tabletop)": (
        "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/point_cloud/iteration_30000/point_cloud.ply"
    ),
    "bicycle (outdoor)": (
        "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/"
        "bicycle/point_cloud/iteration_30000/point_cloud.ply"
    ),
    "stump (outdoor)": (
        "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/stump/point_cloud/iteration_30000/point_cloud.ply"
    ),
}

# Provenance for every preset in :data:`GSPLAT_SCENES` (same keys). The .ply
# presets are the fully-trained 30k-iteration checkpoints of the official
# INRIA 3D Gaussian Splatting models for the Mip-NeRF 360 scenes -- not the
# visibly-undertrained iteration_7000 ones this table used to point at.
GSPLAT_SCENE_PROVENANCE: dict[str, dict[str, str]] = {
    "tabletop (indoor room)": {
        "source": "Vector-Wangel/MuJoCo-GS-Web (curated tabletop environment, sh_degree=0 .spz)",
        "iteration": "n/a (curated export)",
        "license": "MIT (https://github.com/Vector-Wangel/MuJoCo-GS-Web/blob/main/LICENSE)",
    },
    "bonsai (indoor tabletop)": {
        "source": "INRIA 3DGS pre-trained models (Mip-NeRF 360 'bonsai'), mirrored at hf.co/datasets/dylanebert/3dgs",
        "iteration": "30000",
        "license": "Gaussian-Splatting License (research/evaluation; https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)",
    },
    "bicycle (outdoor)": {
        "source": "INRIA 3DGS pre-trained models (Mip-NeRF 360 'bicycle'), mirrored at hf.co/datasets/dylanebert/3dgs",
        "iteration": "30000",
        "license": "Gaussian-Splatting License (research/evaluation; https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)",
    },
    "stump (outdoor)": {
        "source": "INRIA 3DGS pre-trained models (Mip-NeRF 360 'stump'), mirrored at hf.co/datasets/dylanebert/3dgs",
        "iteration": "30000",
        "license": "Gaussian-Splatting License (research/evaluation; https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)",
    },
}


def gsplat_scene_names() -> list[str]:
    """Names of the built-in downloadable 3DGS scenes."""
    return list(GSPLAT_SCENES.keys())


# Per-scene alignment for the LIVE "skybox" backdrop (GsplatBackground(skybox=True)).
# Captured 3DGS scenes carry no canonical up-axis, so the PCA up-sign is authored
# per scene -- the reference (MuJoCo-GS-Web) likewise hand-authors each scene's
# alignment. Keyed by the scene *slug* (first token of the name):
#   tabletop -> MuJoCo-GS-Web's purpose-built room .spz (open floor, clean from
#     every angle -- the recommended scene); bonsai -> object-centric indoor plant
#     (good from hero/oblique angles only); stump -> outdoor clearing.
#   ``bicycle`` is intentionally excluded: it's an overcast outdoor capture that
#   renders as white haze + floaters from every angle (poor as a backdrop).
#   Re-measured on the fully-trained 30k checkpoint (four-view orbit at the
#   arm's eye level, auto up-sign): 33-66% of pixels near-white per view vs
#   bonsai's 0-5%, so the haze is the capture's overcast sky, not the earlier
#   checkpoint's undertraining -- the exclusion stands.
GSPLAT_SKYBOX_ALIGN: dict[str, dict[str, Any]] = {
    # tabletop is a three.js **Y-up** .spz of a kitchen. Align with the KNOWN
    # up-axis (GS +Y) -- PCA mis-picks a horizontal axis and tilts the room ~36 deg
    # (so the arm ends up perpendicular to a wall, not the bench). yaw=180 faces
    # the counter toward the camera; floor_z=-0.8 drops the room so the ~0.8 m
    # counter-top lands at world z=0 (the arm's base), seating the arm ON the
    # bench. own_floor hides MuJoCo's grid so the photoreal floor shows.
    "tabletop": {
        "up_axis": (0.0, 1.0, 0.0),
        "major_axis": (1.0, 0.0, 0.0),
        "up_sign": 1.0,
        "yaw_deg": 180.0,
        "metric": True,
        "floor_z": -0.8,
        "center": (0.0, -0.4),
        "own_floor": True,
        "clip_below": -1.0,
    },
    "bonsai": {"up_sign": -1.0, "yaw_deg": 0.0},
    "stump": {"up_sign": 1.0, "yaw_deg": 0.0},
}


def gsplat_skybox_scene_names() -> list[str]:
    """Names of scenes curated to look good as a LIVE 3DGS skybox backdrop."""
    return [n for n in GSPLAT_SCENES if n.split(" ")[0] in GSPLAT_SKYBOX_ALIGN]


def gsplat_skybox_align_for(name_or_slug: str) -> dict[str, Any]:
    """Authored skybox alignment for a scene name/slug. Empty dict (=> best-effort
    auto up-sign) when the scene isn't curated (e.g. an uploaded .ply)."""
    slug = Path(str(name_or_slug)).stem.split(" ")[0]
    return dict(GSPLAT_SKYBOX_ALIGN.get(slug, {}))


def download_gsplat_scene(name: str, cache_dir: str | Path | None = None) -> Path:
    """Download (and cache) a preset 3DGS scene; return its local path.

    The cached file keeps the source URL's extension (``.spz`` or ``.ply``) so
    the loader can dispatch correctly.

    Args:
        name: a key of :data:`GSPLAT_SCENES`.
        cache_dir: where to cache (default ``~/.cache/strands_robots/gsplat_scenes``).

    Returns:
        Local path to the cached scene file.
    """
    import urllib.request

    if name not in GSPLAT_SCENES:
        raise KeyError(f"Unknown scene {name!r}. Known: {list(GSPLAT_SCENES)}")
    url = GSPLAT_SCENES[name]
    cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "strands_robots" / "gsplat_scenes"
    cache.mkdir(parents=True, exist_ok=True)
    slug = name.split(" ")[0]
    ext = ".spz" if url.lower().split("?")[0].endswith(".spz") else ".ply"
    dest = cache / f"{slug}{ext}"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    logger.info("Downloading 3DGS scene %r -> %s", name, dest)
    tmp = dest.with_suffix(ext + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    logger.info("Downloaded %s (%.0f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def _upright_view_transform(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, viewpoint) -- a rotation standing the scene upright (PCA up ->
    +Z) and a viewpoint at the scene centroid -- for baking a panorama."""
    c = means.mean(axis=0)
    X = means - c
    cov = (X.T @ X) / max(1, len(X))
    _, evecs = np.linalg.eigh(cov)  # ascending; col0 = smallest variance ~= up
    up = evecs[:, 0]
    major = evecs[:, 2]
    z = up / (np.linalg.norm(up) + 1e-9)
    x = major - (major @ z) * z
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)  # rows map gs-axis -> upright-world axis
    return R, c


def bake_gsplat_panorama(
    ply_path: str | Path,
    out_path: str | Path | None = None,
    face_size: int = 640,
    equi_w: int = 2048,
    equi_h: int = 1024,
    device: str = "cuda",
) -> Path:
    """Render a 3DGS ``.ply`` into an equirectangular panorama image.

    Renders 6 cube faces (90 deg FOV) outward from the scene centroid in the
    scene's upright frame, then reprojects them into an equirectangular image
    using the *same* spherical convention :class:`PanoramaBackground` samples
    with. The result is a clean, camera-consistent skybox-style backdrop that
    "just works" without per-camera viewpoint alignment (the trade-off is no
    parallax -- the backdrop sits at infinity).

    Baking is expensive (six gaussian-splat renders), so a non-empty output
    file short-circuits the whole pass. The default output path therefore
    encodes the geometry that determines the pixels -- ``equi_w``, ``equi_h``
    and ``face_size`` -- as ``<stem>_pano_<equi_w>x<equi_h>_f<face_size>.jpg``.
    Without that, every bake of one scene shared a single ``<stem>_pano.jpg``
    and a later call asking for a different resolution silently returned the
    first call's image: the caller's ``equi_w`` / ``equi_h`` / ``face_size``
    were accepted and then dropped. An explicit ``out_path`` is caller-owned
    and is honored verbatim -- the caller named the file, so the caller owns
    what it holds.

    Returns the path to the written panorama ``.jpg`` (cached next to the ply
    unless ``out_path`` says otherwise).

    Raises:
        ValueError: if ``face_size``, ``equi_w`` or ``equi_h`` is not a positive
            whole number. These knobs are forwarded to
            :func:`~strands_robots.rendering.ibl.render_environment_map`, which
            owns their domain, so this bake checks that same domain rather than
            restating it -- and checks it *here* because two things read the
            values before that renderer does (see below).
    """
    # Both of those reads happen before ``render_environment_map`` is reached, so
    # its own refusal arrives too late to be the only one.
    #
    # The default path is composed from these values and that name is the cache
    # key, so an integral float must not spell a second file for pixels already
    # baked: the shared domain accepts ``face_size=640.0``, which composed
    # ``<stem>_pano_2048x1024_f640.0.jpg`` and re-baked a warm
    # ``..._f640.jpg`` beside it -- the silent-no-op failure the geometry in this
    # name exists to prevent, reintroduced by spelling. ``int()`` normalizes it
    # away, as ``environment_map_cache_path`` does for the same reason.
    #
    # The splat load is what an unusable resolution would otherwise pay for
    # first, and without the ``sim-gs`` extra installed the load does not merely
    # delay the refusal, it replaces it: ``equi_w=0`` reported a missing ``torch``
    # and advised installing it, which fixes nothing and names no resolution.
    from .ibl import _resolution_error, render_environment_map

    if text := _resolution_error("bake_gsplat_panorama", face_size=face_size, equi_w=equi_w, equi_h=equi_h):
        raise ValueError(text)
    face_size, equi_w, equi_h = int(face_size), int(equi_w), int(equi_h)
    ply_path = Path(ply_path)
    if out_path is not None:
        out = Path(out_path)
    else:
        out = ply_path.with_name(f"{ply_path.stem}_pano_{equi_w}x{equi_h}_f{face_size}.jpg")
    if out.exists() and out.stat().st_size > 0:
        return out

    # Load splats once; place the scene upright with the viewpoint at origin.
    base = GsplatBackground(ply_path=ply_path, device=device)
    base._load()
    assert base._splats is not None  # populated by _load()
    means = base._splats["means"].detach().cpu().numpy()
    R, viewpoint = _upright_view_transform(means)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ viewpoint  # world_from_gs: centroid -> origin, upright
    base._transform = T

    # The six-cube-face render + equirect reprojection is shared with the
    # world-frame environment-map bake (issue #2323), imported above with the
    # resolution domain it owns. This bake keeps its own viewpoint (the scene
    # centroid, in the unaligned upright frame above).
    pano = render_environment_map(
        base,
        origin_world=(0.0, 0.0, 0.0),
        face_size=face_size,
        equi_w=equi_w,
        equi_h=equi_h,
    )

    from PIL import Image as _Image

    _Image.fromarray(pano).save(out, quality=88)
    logger.info("Baked GS panorama -> %s", out)
    return out


def _fit_backdrop_transform(means: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Fit a ``world_from_gs`` SE(3)+scale that stands a captured scene upright,
    scales it to ~``radius`` m, and centres it on ``center``.

    Heuristics (captured scenes carry no canonical frame):
      * **Up axis** = the smallest-variance PCA axis of the gaussian positions
        (a room is wide + deep but short -> the thin axis ~= the floor normal).
      * **Scale** so the in-plane (horizontal) extent ~= ``radius``.
      * **Centre** the scene centroid at ``center``.

    Approximate by design -- exposed for tuning. Returns a 4x4 matrix mapping
    gaussian coords -> MuJoCo world coords.
    """
    c = means.mean(axis=0)
    X = means - c
    # Robust extent: use a percentile to ignore far "floater" gaussians.
    cov = (X.T @ X) / max(1, len(X))
    evals, evecs = np.linalg.eigh(cov)  # ascending eigenvalues; columns = axes
    up = evecs[:, 0]  # smallest variance ~= floor normal
    horiz_major = evecs[:, 2]  # largest in-plane axis
    # Orthonormal world-from-gs basis: gs `up` -> +Z, gs major -> +X.
    z = up / (np.linalg.norm(up) + 1e-9)
    x = horiz_major - (horiz_major @ z) * z
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    R_gs_to_world = np.stack([x, y, z], axis=0)  # rows map gs-axis -> world axis
    # Horizontal radius (95th pct of in-floor-plane distance) -> scale.
    horiz = np.linalg.norm((X @ np.stack([evecs[:, 2], evecs[:, 1]], axis=1)), axis=1)
    r95 = float(np.percentile(horiz, 95)) or 1.0
    s = float(radius) / r95
    T = np.eye(4)
    T[:3, :3] = s * R_gs_to_world
    T[:3, 3] = np.asarray(center, float) - (s * R_gs_to_world) @ c
    return T


def _auto_up_sign(means: np.ndarray) -> float:
    """Best-effort guess of the PCA up-axis sign (which way is "up").

    The floor is a dense, thin slab; project the gaussians onto the PCA up-axis
    and assume the densest layer is the floor -> world-up points away from it.
    Reliable enough for a rough upload preview; curated presets override this
    via :data:`GSPLAT_SKYBOX_ALIGN`.
    """
    c = means.mean(axis=0)
    X = means - c
    cov = (X.T @ X) / max(1, len(X))
    _, evecs = np.linalg.eigh(cov)
    u = X @ evecs[:, 0]
    hist, edges = np.histogram(u, bins=64)
    peak = 0.5 * (edges[int(hist.argmax())] + edges[int(hist.argmax()) + 1])
    # Densest slab (floor) on the +u side -> up is -u -> sign -1, else +1.
    return -1.0 if peak > float(np.median(u)) else 1.0


def _fit_skybox_transform(
    means: np.ndarray,
    up_sign: float = 1.0,
    yaw_deg: float = 0.0,
    radius: float = 2.5,
    center: tuple = (0.05, 0.05),
    floor_z: float = -0.3,
    floor_pct: float = 2.0,
    metric: bool = False,
    up_axis: tuple | None = None,
    major_axis: tuple | None = None,
) -> np.ndarray:
    """Fit a ``world_from_gs`` (4x4) for the live skybox backdrop.

    Stands the scene upright (its up-axis x ``up_sign`` -> world +Z), applies an
    extra ``yaw_deg`` about +Z, then either scales the horizontal extent to
    ~``radius`` m (captured scenes carry no metric scale) or -- when
    ``metric=True`` -- keeps the scene's own metric scale (use this for
    already-metric assets like the ``.spz`` exports). Places the ``floor_pct``
    percentile of world-z at ``floor_z`` (set this to *minus the height of the
    surface the arm should rest on* to seat the arm on a table). Centres the
    horizontal centroid at ``center``.

    The up-axis is the PCA smallest-variance axis by default (works for casual
    captures), but for assets with a *known* convention (e.g. a three.js Y-up
    ``.spz``) pass an explicit ``up_axis`` (and optional ``major_axis`` for the
    in-plane orientation) -- PCA can otherwise pick a horizontal axis and tilt
    the whole scene. See :class:`GsplatBackground`.
    """
    c = means.mean(axis=0)
    X = means - c
    cov = (X.T @ X) / max(1, len(X))
    _, evecs = np.linalg.eigh(cov)  # ascending; col0 = smallest var
    up = np.asarray(up_axis, dtype=float) if up_axis is not None else evecs[:, 0]
    major = np.asarray(major_axis, dtype=float) if major_axis is not None else evecs[:, 2]
    z = up_sign * up / (np.linalg.norm(up) + 1e-9)
    x = major - (major @ z) * z
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)  # rows map gs-axis -> world axis

    t = np.deg2rad(yaw_deg)
    ct, st = np.cos(t), np.sin(t)
    Rz = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ R

    # Horizontal (in-floor-plane) extent -> scale: world x/y components of R.
    horiz = np.linalg.norm(X @ R[:2].T, axis=1)
    r95 = float(np.percentile(horiz, 95)) or 1.0
    s = 1.0 if metric else float(radius) / r95
    M = s * R

    pts = X @ M.T
    floor_zz = float(np.percentile(pts[:, 2], floor_pct))
    b = np.array([center[0], center[1], float(floor_z) - floor_zz], dtype=float)

    T = np.eye(4)
    T[:3, :3] = M
    T[:3, 3] = b - M @ c
    return T


# --------------------------------------------------------------------------- #
# SPZ (Niantic Gaussian SPlat) reader -- pure numpy, no extra deps.
# This is the format MuJoCo-GS-Web ships its curated scenes in (e.g. the
# "tabletop" environment). Spec transcribed from the `spz` rust crate.
# --------------------------------------------------------------------------- #

_SPZ_MAGIC = 0x5053474E  # "NGSP"
_SPZ_COLOR_SCALE = 0.15


def _decode_spz_rotations(rot: np.ndarray, smallest_three: bool) -> np.ndarray:
    """``rot`` (N, 4|3) uint8 -> (N, 4) quaternion in WXYZ order (gsplat/INRIA)."""
    N = rot.shape[0]
    xyzw = np.zeros((N, 4), np.float32)  # [x, y, z, w]
    if smallest_three:  # version 3
        comp = (
            rot[:, 0].astype(np.uint32)
            | (rot[:, 1].astype(np.uint32) << 8)
            | (rot[:, 2].astype(np.uint32) << 16)
            | (rot[:, 3].astype(np.uint32) << 24)
        )
        i_largest = (comp >> 30).astype(np.int64)
        c_mask = np.uint32((1 << 9) - 1)
        inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
        c = comp.copy()
        ssq = np.zeros(N, np.float32)
        for i in (3, 2, 1, 0):  # non-largest comps consume 10 bits each, high index first
            active = i_largest != i
            mag = (c & c_mask).astype(np.float32)
            negbit = (c >> 9) & np.uint32(1)
            val = inv_sqrt2 * mag / float(c_mask)
            val = np.where(negbit == 1, -val, val).astype(np.float32)
            xyzw[active, i] = val[active]
            ssq[active] += (val * val)[active]
            c = np.where(active, c >> 10, c)
        xyzw[np.arange(N), i_largest] = np.sqrt(np.maximum(0.0, 1.0 - ssq)).astype(np.float32)
    else:  # version 2: "first three" + reconstructed w
        xyz = rot[:, :3].astype(np.float32) * np.float32(1.0 / 127.5) - 1.0
        xyzw[:, :3] = xyz
        xyzw[:, 3] = np.sqrt(np.maximum(0.0, 1.0 - (xyz * xyz).sum(axis=1)))
    return xyzw[:, [3, 0, 1, 2]].copy()  # -> WXYZ


def _load_spz_splats(spz_path: Path, device: str) -> dict[str, Any]:
    """Load a Niantic ``.spz`` (versions 2 & 3) into the same dict layout as
    :func:`_load_ply_splats`, plus an ``"antialiased"`` bool (header flags
    bit 0: the asset was trained with mip-splatting AA and must be rasterized
    with ``rasterize_mode="antialiased"``). ``sh_degree=0`` assets get baked
    DC color (``colors`` as ``(N, 3)`` RGB); assets with higher-order SH
    decode the trailing coefficient block into ``colors`` as ``(N, K, 3)``
    raw SH so the rasterizer can evaluate view-dependent color."""
    import gzip
    import struct

    require_optional("torch", extra="sim-gs", purpose=_GSPLAT_PURPOSE)
    import torch

    with gzip.open(str(spz_path), "rb") as f:
        raw = f.read()
    magic, version, num_points = struct.unpack_from("<iii", raw, 0)
    sh_degree, frac_bits, flags, _reserved = struct.unpack_from("<BBBB", raw, 12)
    if magic != _SPZ_MAGIC:
        raise ValueError(f"{spz_path}: bad SPZ magic {magic:#x}")
    if version not in (2, 3):
        raise ValueError(f"{spz_path}: unsupported SPZ version {version}")

    N = num_points
    smallest3 = version >= 3
    pos_bytes = 9  # 24-bit fixed point (version 1 float16 is not produced in practice)
    rot_bytes = 4 if smallest3 else 3

    off = 16
    pos = np.frombuffer(raw, np.uint8, count=N * pos_bytes, offset=off)
    off += N * pos_bytes
    alpha = np.frombuffer(raw, np.uint8, count=N, offset=off)
    off += N
    col = np.frombuffer(raw, np.uint8, count=N * 3, offset=off).reshape(N, 3)
    off += N * 3
    scl = np.frombuffer(raw, np.uint8, count=N * 3, offset=off).reshape(N, 3)
    off += N * 3
    rot = np.frombuffer(raw, np.uint8, count=N * rot_bytes, offset=off).reshape(N, rot_bytes)
    off += N * rot_bytes
    # Trailing SH block: (sh_degree+1)^2 - 1 coefficients per channel, one byte
    # each, coefficient-major with the color channel fastest-varying (spz spec).
    sh_rest: np.ndarray | None = None
    if sh_degree > 0:
        n_rest = (sh_degree + 1) ** 2 - 1
        want = N * n_rest * 3
        if len(raw) - off < want:
            raise ValueError(
                f"{spz_path}: header claims sh_degree={sh_degree} "
                f"({want} SH bytes) but only {len(raw) - off} bytes remain"
            )
        sh_bytes = np.frombuffer(raw, np.uint8, count=want, offset=off).reshape(N, n_rest, 3)
        # unquantizeSH: byte -> (byte - 128) / 128 (spz spec).
        sh_rest = (sh_bytes.astype(np.float32) - 128.0) / 128.0
        off += want

    # positions: 24-bit little-endian signed fixed point / 2^frac_bits
    p = pos.reshape(N, 3, 3).astype(np.int32)
    fixed = p[:, :, 0] | (p[:, :, 1] << 8) | (p[:, :, 2] << 16)
    fixed = np.where(fixed >= 0x800000, fixed - 0x1000000, fixed)
    means = fixed.astype(np.float32) / float(1 << frac_bits)

    scales = np.exp(scl.astype(np.float32) / 16.0 - 10.0)
    opac = alpha.astype(np.float32) / 255.0
    f_dc = (col.astype(np.float32) / 255.0 - 0.5) / _SPZ_COLOR_SCALE
    if sh_rest is not None:
        # Raw SH coefficients (DC first): the rasterizer evaluates these
        # per-view (see GsplatBackground.render).
        colors = np.concatenate([f_dc[:, None, :], sh_rest], axis=1)
    else:
        # DC-only fast path: bake the DC term to per-gaussian RGB.
        colors = np.clip(0.5 + 0.28209479177387814 * f_dc, 0.0, 1.0)
    quats = _decode_spz_rotations(rot, smallest3)

    logger.info(
        "Loaded SPZ %s: v%d, %d splats, sh_degree=%d, antialiased=%s",
        spz_path.name,
        version,
        N,
        sh_degree,
        bool(flags & 0x1),
    )

    def to_t(a: np.ndarray, dt: Any = None) -> Any:
        return torch.from_numpy(np.ascontiguousarray(a)).to(dt or torch.float32).to(device)

    return {
        "means": to_t(means),
        "scales": to_t(scales),
        "quats": to_t(quats),
        "opacities": to_t(opac),
        "colors": to_t(colors),
        "antialiased": bool(flags & 0x1),
    }


def _load_ply_splats(ply_path: Path, device: str) -> dict[str, Any]:
    """Minimal Gaussian-splat .ply loader.

    Supports the standard 3DGS PLY layout (means as ``x y z``, scales as
    ``scale_0 scale_1 scale_2`` in log-space, rotations as ``rot_0..rot_3``
    quaternions, opacity as ``opacity``, SH DC color as ``f_dc_0..2``, and
    optional higher-order SH as ``f_rest_*``).

    Assets without ``f_rest_*`` take the DC-only fast path: the DC term is
    baked to per-gaussian RGB (``colors`` as ``(N, 3)``). Assets that carry
    higher-order SH return ``colors`` as raw coefficients ``(N, K, 3)`` (DC
    first) so the rasterizer can evaluate view-dependent color per frame.
    """
    require_optional("torch", extra="sim-gs", purpose=_GSPLAT_PURPOSE)
    require_optional("plyfile", extra="sim-gs", purpose=".ply Gaussian-splat loading")
    import torch
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data

    means = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1)
    opac = np.array(v["opacity"])
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1)

    # Higher-order SH: the INRIA layout flattens per-point (3, K-1) coefficient
    # blocks channel-major, so f_rest_{c * (K-1) + j} is channel c, coeff j.
    n_rest_fields = sum(1 for name in v.dtype.names if name.startswith("f_rest_"))
    if n_rest_fields:
        if n_rest_fields % 3:
            raise ValueError(f"{ply_path}: {n_rest_fields} f_rest_* fields is not divisible by 3 channels")
        n_rest = n_rest_fields // 3
        degree = int(np.sqrt(n_rest + 1)) - 1
        if (degree + 1) ** 2 - 1 != n_rest:
            raise ValueError(
                f"{ply_path}: {n_rest} SH coefficients per channel does not "
                f"match any degree ((deg+1)^2 - 1 for integer deg)"
            )
        rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest_fields)], axis=-1)
        rest = rest.reshape(-1, 3, n_rest).transpose(0, 2, 1)  # -> (N, K-1, 3)
        colors = np.concatenate([sh_dc[:, None, :], rest], axis=1).astype(np.float32)
    else:
        # DC-only fast path: SH DC -> linear RGB
        # (see https://github.com/graphdeco-inria/gaussian-splatting).
        SH_C0 = 0.28209479177387814
        colors = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0)
    # Sigmoid opacity, exp scale.
    opac = 1.0 / (1.0 + np.exp(-opac))
    scales = np.exp(scales)

    def to_t(a: np.ndarray, dt: Any = None) -> Any:
        return torch.from_numpy(np.ascontiguousarray(a)).to(dt or torch.float32).to(device)

    return {
        "means": to_t(means),
        "scales": to_t(scales),
        "quats": to_t(quats),
        "opacities": to_t(opac),
        "colors": to_t(colors),
    }
