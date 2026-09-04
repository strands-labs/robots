# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid (simulation foreground + photoreal background) compositor.

This is the library home of the depth-aware compositing that the
``examples/mujoco_gs`` and ``examples/isaac_gs`` demos previously each
re-implemented (issue #1537): a per-pixel z-compare between a simulation
backend's ``(rgb, depth)`` foreground and a :class:`BackgroundRenderer`'s
``(rgb, depth)`` backdrop::

    +---------+        +---------------------+        +-----------+
    | backend |  RGB,  |                     |        |   final   |
    | get_    |--D-->  | per-pixel z-compare |  --->  |  composite|
    | frame   |        |                     |        |   frame   |
    +---------+        +---------------------+        +-----------+
        ^                       ^
        |                       | RGB, D
        |               +-------+-------+
        |               | Background    |
        +-- camera ---->| (panorama or  |
            params      |   gsplat)     |
                        +---------------+

Per-pixel rule: ``foreground_depth < background_depth`` -> foreground wins.
The compositor is pure numpy over the two public backend APIs
(:meth:`SimEngine.get_frame` / :meth:`SimEngine.get_camera_params`), so it is
backend-agnostic: any engine that implements those works unchanged.

Depth convention: backgrounds report "at infinity" pixels as
``depth >= zfar``; foreground pixels with no geometry (sky) are expected to
come back either pinned to the far clip (MuJoCo) or as ``0`` / non-finite
(Isaac's RTX annotator) -- both are treated as background. Scenes intended for
full-frame photoreal backdrops usually want ``create_world(ground_plane=False)``
(or an example-side floor-hiding shim) so the sim floor doesn't fight the
background's own floor.

Background caching: the backdrop is rendered once per camera and reused while
only the robot moves, since ``BackgroundRenderer.render(cam)`` depends on
nothing but the camera. "Per camera" means per distinct
:class:`~strands_robots.rendering.camera.CameraParams` -- pose, intrinsics,
image size **and both clip planes** -- because the backdrop is built from all
of them: the panorama parks its whole depth buffer at ``cam.zfar`` and the
gsplat backdrop hands ``cam.znear`` / ``cam.zfar`` to the rasterizer. That
matters because a backend can move the clip planes on its own: MuJoCo derives
both from ``model.stat.extent``, which the compiler recomputes from the scene
bounds, so any scene change (``add_object``, ``attach_bodies``,
``load_scene``) moves them while a fixed named camera keeps its pose and its
intrinsics. Callers therefore do not have to invalidate anything after a scene
change; :meth:`HybridCompositor.clear_caches` remains available to drop the
entries outright.

Concurrency contract: :meth:`HybridCompositor.render` calls straight into the
engine's ``get_frame`` on the calling thread -- thread-affinity is the
*backend's* contract (MuJoCo caches GL renderers per-thread; Isaac renders
must run on the SimulationApp thread). Call from a consistent thread, or
marshal calls yourself (see ``examples/mujoco_gs/compositor.py`` for a
single-render-thread wrapper).
"""

from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass, fields
from typing import Any, Protocol

import numpy as np

from ..utils import positive_whole_number_error
from .backgrounds import BackgroundRenderer, PanoramaBackground
from .camera import CameraParams
from .color import linear_to_srgb, relative_luminance, srgb_to_linear

logger = logging.getLogger(__name__)


def _feather_pixels_error(value: Any) -> str | None:
    """Error text when ``value`` is not a usable feather radius.

    A feather radius is a whole number of pixels, with ``0`` meaning "no
    blend". Nothing else can be honored: :func:`feather_mask` builds a
    ``2 * radius + 1`` box kernel, so a fractional or non-finite radius has no
    kernel and a negative one has no edge to soften. Those values used to be
    coerced with ``max(0, int(value))``, which turned every one of them into
    ``0`` - silently disabling the very seam blend the caller asked for.
    ``bool`` is rejected explicitly: it is an ``int`` subclass whose ``True``
    would act as a 1-pixel radius.

    Args:
        value: The caller-supplied radius.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    message = f"HybridCompositor: feather_pixels must be a whole number of pixels >= 0, got {value!r}."
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message
    numeric = float(value)
    # ``isfinite`` first: ``int(nan)`` raises, so short-circuit before the
    # integrality check below.
    if not math.isfinite(numeric) or numeric != int(numeric) or numeric < 0:
        return message
    return None


def _depth_epsilon_error(value: Any) -> str | None:
    """Error text when ``value`` is not a usable no-geometry depth threshold.

    The threshold is compared against the foreground depth buffer
    (``fg_depth > depth_epsilon`` selects "the simulation saw geometry here"),
    so only a finite, non-negative distance in meters can be honored:

    * ``nan`` makes that comparison ``False`` for every pixel, so the whole
      simulation foreground reads as empty and the composite is the background
      alone - the robot silently disappears from the frame.
    * ``inf`` discards every pixel for the same reason.
    * a negative threshold admits the no-hit pixels the parameter exists to
      exclude (Isaac's RTX annotator reports them as ``0``), painting
      simulation sky over the background.

    ``0`` is a legitimate setting: it means "only an exactly-zero depth counts
    as no geometry". ``bool`` is rejected explicitly - a truth value is not a
    distance, and ``True`` would act as a 1 m threshold.

    Args:
        value: The caller-supplied threshold, in meters.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    message = f"HybridCompositor: depth_epsilon must be a finite distance in meters >= 0, got {value!r}."
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        return message
    return None


def _shadow_plane_z_error(value: Any) -> str | None:
    """Error text when ``value`` is not a usable shadow-catcher plane height.

    The height is a world-frame z coordinate in meters, compared against the
    analytic per-pixel plane depth (:func:`plane_depth`), so only a finite
    number can be honored: a non-finite plane intersects no ray and the
    shadow pass would silently never fire. ``None`` (feature off) is handled
    by the caller, not here. ``bool`` is rejected explicitly -- a truth value
    is not a height.

    Args:
        value: The caller-supplied plane height, in meters.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    message = f"HybridCompositor: shadow_plane_z must be a finite world-frame height in meters, got {value!r}."
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message
    if not math.isfinite(float(value)):
        return message
    return None


def _shadow_plane_tolerance_error(value: Any) -> str | None:
    """Error text when ``value`` is not a usable plane-match depth tolerance.

    Catcher pixels are recognized by ``|fg_depth - plane_depth| <= tolerance``,
    so only a finite distance in meters ``> 0`` can be honored: ``0`` (or a
    negative value) matches no pixel even on an exact plane (float depth is
    never bit-exact), and a non-finite tolerance matches *every* valid
    foreground pixel -- the robot itself would be read as shadow and erased.

    Args:
        value: The caller-supplied tolerance, in meters.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    message = f"HybridCompositor: shadow_plane_tolerance must be a finite distance in meters > 0, got {value!r}."
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        return message
    return None


def _shadow_min_factor_error(value: Any) -> str | None:
    """Error text when ``value`` is not a usable shadow darkening floor.

    The floor is a multiplicative factor on linear background light, so only
    a finite number in ``[0, 1]`` can be honored: below ``0`` a shadow would
    invert the background's sign, above ``1`` it would *brighten*, and a
    non-finite floor makes every clip degenerate. ``bool`` is rejected
    explicitly -- ``True`` would read as "no darkening at all".

    Args:
        value: The caller-supplied factor floor.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    message = f"HybridCompositor: shadow_min_factor must be a number in [0, 1], got {value!r}."
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        return message
    return None


def plane_depth(cam: CameraParams, plane_z: float) -> np.ndarray:
    """Per-pixel camera-frame z-depth of the horizontal plane ``z == plane_z``.

    Casts one ray per pixel (same optical convention as the backgrounds: +X
    right, +Y up, -Z forward) and intersects it with the world-frame
    horizontal plane at height ``plane_z``. Pure camera math -- this is how
    the shadow-catcher pass recognizes "this foreground pixel is the catcher
    plane" without the compositor knowing anything about scene geometry.

    Args:
        cam: pinhole camera parameters at the target image size.
        plane_z: world-frame height of the plane, meters.

    Returns:
        ``(H, W) float32`` z-depth in meters; ``np.inf`` where the ray never
        hits the plane (parallel to it, or the hit is behind the camera).
    """
    H, W = cam.height, cam.width
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    Kinv = np.linalg.inv(cam.K)
    homo = np.stack([u, v, np.ones_like(u)], axis=-1)  # (H, W, 3)
    dirs_cam = homo @ Kinv.T  # image-plane rays at unit z-depth
    dirs_cam[..., 1] *= -1.0  # image v grows down -> camera +Y up
    dirs_cam[..., 2] *= -1.0  # OpenGL: -Z forward
    # World-frame ray p(s) = origin + s * (R @ dirs_cam); because dirs_cam has
    # camera-frame z == -1, the parameter ``s`` *is* the camera-frame z-depth.
    R = cam.T_world_cam[:3, :3]
    origin_z = float(cam.T_world_cam[2, 3])
    dz_world = dirs_cam @ R.T[:, 2]  # world-z component of each ray
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (float(plane_z) - origin_z) / dz_world
    depth = np.where(np.isfinite(s) & (s > 0.0), s, np.inf)
    return depth.astype(np.float32)


# Decimal places each ``CameraParams`` matrix is rounded to before it enters a
# background cache key, so float jitter in a pose the caller did not move does
# not bust the cache. Scalars are keyed exactly: they arrive from the backend as
# a single multiply and reproduce bit-for-bit, and a spurious *miss* only costs
# one background pass while a spurious *hit* returns a frame rendered for
# another camera.
_CACHE_KEY_MATRIX_DECIMALS = {"T_world_cam": 4, "K": 3}
_CACHE_KEY_DEFAULT_DECIMALS = 6


def _background_cache_key(camera_name: str, background_name: str, cam: CameraParams) -> tuple[Any, ...]:
    """Identify a cached ``BackgroundRenderer.render`` result.

    ``render(cam)`` reads nothing but ``cam``, so the key covers **every**
    :class:`CameraParams` field rather than a hand-listed subset: the
    equirectangular backdrop fills its whole depth buffer with ``cam.zfar``
    and the gsplat backdrop hands ``cam.znear`` / ``cam.zfar`` to the
    rasterizer as its clip planes, so a camera whose clip planes moved is a
    different render even when its pose, intrinsics and image size are
    untouched. Deriving the key from the dataclass rather than listing fields
    means a field added to :class:`CameraParams` later participates on
    arrival instead of silently widening what one entry stands for.

    Args:
        camera_name: the camera as named to the engine. Two cameras can share
            a pose (a duplicated MJCF camera), so the name is part of the key.
        background_name: :attr:`BackgroundRenderer.name` of the renderer that
            produced the entry, so a hot-swapped backdrop cannot be served
            from the previous one's entry.
        cam: the parameters ``render`` was, or would be, called with.

    Returns:
        A hashable key. Matrix fields are rounded (see
        ``_CACHE_KEY_MATRIX_DECIMALS``) and keyed by bytes; scalar fields are
        keyed exactly.
    """
    parts: list[Any] = [camera_name, background_name]
    for field in fields(cam):
        value = getattr(cam, field.name)
        if isinstance(value, np.ndarray):
            decimals = _CACHE_KEY_MATRIX_DECIMALS.get(field.name, _CACHE_KEY_DEFAULT_DECIMALS)
            parts.append(np.round(value, decimals).tobytes())
        else:
            parts.append(value)
    return tuple(parts)


class FrameSource(Protocol):
    """The slice of ``SimEngine`` the compositor needs (structural typing)."""

    def get_frame(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return an ``(rgb, depth)`` pair for the named camera."""

    def get_camera_params(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> CameraParams:
        """Return the intrinsics/extrinsics for the named camera."""


@dataclass
class CompositeFrame:
    """Output of :meth:`HybridCompositor.render`.

    Attributes:
        rgb: ``(H, W, 3) uint8`` final composited image.
        foreground_rgb: ``(H, W, 3) uint8`` simulation-only render, for debugging.
        background_rgb: ``(H, W, 3) uint8`` background-only render, for
            debugging. When a shadow catcher is configured
            (``shadow_plane_z``), this is the background *with the caught
            shadow applied* -- i.e. the layer actually composited under the
            foreground.
        foreground_mask: ``(H, W) bool`` ``True`` where the foreground won the
            depth test (i.e. a simulated object is visible). Shadow-catcher
            pixels are never part of the mask -- the catcher plane modulates
            the background instead of painting over it.
        depth: ``(H, W) float32`` foreground depth in meters (the simulation
            depth, since the foreground is what the user usually cares about
            measuring).
        camera: :class:`CameraParams` used to render this frame. Every array
            above is ``(camera.height, camera.width)``, so ``camera.K``
            describes them - :meth:`HybridCompositor.render` refuses a layer
            at any other size rather than returning a frame its own camera
            cannot describe.
    """

    rgb: np.ndarray
    foreground_rgb: np.ndarray
    background_rgb: np.ndarray
    foreground_mask: np.ndarray
    depth: np.ndarray
    camera: CameraParams


class HybridCompositor:
    """Render and composite simulation + photoreal background frames.

    Works against any engine exposing the public raw-frame APIs
    ``get_frame`` / ``get_camera_params`` (MuJoCo, Isaac; Newton renders RGB
    only and therefore cannot composite -- see :meth:`render`).

    Args:
        sim: a live engine implementing :class:`FrameSource`
            (e.g. ``strands_robots.simulation.Simulation``).
        background: any :class:`BackgroundRenderer`. Defaults to a procedural
            panorama so hybrid rendering works out of the box with zero ML deps.
        default_width: image width if not overridden per call, a positive
            whole number of pixels (the shared media domain - see
            :func:`~strands_robots.utils.positive_whole_number_error`).
            ``None`` defers to the engine's per-camera configuration.
        default_height: image height if not overridden per call, same domain.
        feather_pixels: width (in pixels) of a soft edge blend between
            foreground and background to hide the offscreen-renderer's
            anti-aliasing seam. A whole number of pixels; ``0`` disables
            feathering. Default ``1``.
        depth_epsilon: foreground depth at or below this (meters) is treated
            as "no geometry" -- those pixels show the background. Isaac's RTX
            depth annotator reports no-hit pixels as ``0`` (or non-finite);
            MuJoCo pins them to the far clip. Both extremes read as background.
            A finite distance in meters, ``>= 0``.
        shadow_plane_z: world-frame height (meters) of a shadow-catcher plane
            the *scene* renders (an untextured plane the robot casts shadows
            onto). Foreground pixels whose depth matches the analytic depth of
            that plane (:func:`plane_depth`) are read as the catcher: they are
            excluded from the foreground mask (the plane itself is never
            painted) and their relative darkening -- the shadow the simulator
            rendered onto the plane -- is multiplied onto the background in
            linear light, so the robot grounds itself with a contact shadow on
            the photoreal backdrop. ``None`` (default) disables the pass.
            A finite height in meters.
        shadow_plane_tolerance: how close (meters, along the ray's z-depth) a
            foreground depth must be to the analytic plane depth to count as
            the catcher. A finite distance ``> 0``. Default ``0.01``.
        shadow_min_factor: floor on the shadow's multiplicative darkening, in
            ``[0, 1]`` -- ``0`` allows pitch-black shadows, ``1`` disables
            darkening. Default ``0.25``, so contact shadows read clearly
            without punching a black hole into the backdrop.
        blend_in_linear: when ``True``, the fg/bg seam blend (and nothing
            else) is computed in linear light instead of on gamma-encoded
            bytes -- see :mod:`strands_robots.rendering.color` for the
            pipeline contract. Fully-foreground and fully-background pixels
            are byte-identical either way. Default ``False`` (keeps the
            historical byte-space blend).

    Raises:
        ValueError: if any option cannot be honored -- a ``feather_pixels``
            that is not a whole pixel count ``>= 0``, a ``depth_epsilon`` that
            is not a finite distance ``>= 0`` (a non-finite threshold discards
            the entire foreground, so the robot would vanish from the frame),
            a ``default_width`` / ``default_height`` that is not a positive
            whole number, a non-finite ``shadow_plane_z``, a
            ``shadow_plane_tolerance`` that is not a finite distance ``> 0``
            (``inf`` would read the robot itself as shadow), a
            ``shadow_min_factor`` outside ``[0, 1]``, or a
            ``blend_in_linear`` that is not a ``bool``. Every one of these
            was previously coerced or clamped into a plausible-but-different
            render.

    Example:

        >>> from strands_robots.simulation import Simulation
        >>> from strands_robots.rendering import HybridCompositor
        >>> sim = Simulation()
        >>> sim.create_world()
        >>> sim.add_robot("arm", data_config="so101")
        >>> sim.add_camera("front", position=[0.4, -0.5, 0.3], target=[0.0, 0.0, 0.1])
        >>> sim.step(20)
        >>> frame = HybridCompositor(sim).render(camera_name="front")
        >>> frame.rgb.dtype
        dtype('uint8')
    """

    def __init__(
        self,
        sim: FrameSource,
        background: BackgroundRenderer | None = None,
        default_width: int | None = None,
        default_height: int | None = None,
        feather_pixels: int = 1,
        depth_epsilon: float = 1e-4,
        shadow_plane_z: float | None = None,
        shadow_plane_tolerance: float = 0.01,
        shadow_min_factor: float = 0.25,
        blend_in_linear: bool = False,
    ) -> None:
        if text := _feather_pixels_error(feather_pixels):
            raise ValueError(text)
        if text := _depth_epsilon_error(depth_epsilon):
            raise ValueError(text)
        if shadow_plane_z is not None and (text := _shadow_plane_z_error(shadow_plane_z)):
            raise ValueError(text)
        if text := _shadow_plane_tolerance_error(shadow_plane_tolerance):
            raise ValueError(text)
        if text := _shadow_min_factor_error(shadow_min_factor):
            raise ValueError(text)
        if not isinstance(blend_in_linear, bool):
            # Truthiness would silently honor e.g. a misplaced string; the
            # option changes the blend math, so it must be an explicit bool.
            raise ValueError(f"HybridCompositor: blend_in_linear must be a bool, got {blend_in_linear!r}.")
        for label, size in (("default_width", default_width), ("default_height", default_height)):
            # ``None`` means "defer to the engine"; any supplied size must be a
            # size the engine can render, checked here rather than at the first
            # render so the error names the constructor argument.
            if size is not None and (size_text := positive_whole_number_error(size, label, "HybridCompositor")):
                raise ValueError(size_text)
        self.sim = sim
        self.background: BackgroundRenderer = background or PanoramaBackground()
        # Normalize to plain ints/floats: these flow into the requested camera
        # size and into status/cache keys, so a np.int64 must not leak through.
        self.default_width = None if default_width is None else int(default_width)
        self.default_height = None if default_height is None else int(default_height)
        self.feather_pixels = int(feather_pixels)
        self.depth_epsilon = float(depth_epsilon)
        self.shadow_plane_z = None if shadow_plane_z is None else float(shadow_plane_z)
        self.shadow_plane_tolerance = float(shadow_plane_tolerance)
        self.shadow_min_factor = float(shadow_min_factor)
        self.blend_in_linear = bool(blend_in_linear)
        # Cache of background renders keyed by the camera name, the background
        # name and every ``CameraParams`` field (see
        # :func:`_background_cache_key`). The background only changes when the
        # *camera* does, not when the robot moves -- so during a live motion we
        # recompute only the cheap sim foreground, not the expensive
        # panorama/gsplat pass.
        self._bg_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}

    # ----- main API ----- #

    def render(
        self,
        camera_name: str = "default",
        width: int | None = None,
        height: int | None = None,
    ) -> CompositeFrame:
        """Render one depth-composited frame of the current sim state.

        Args:
            camera_name: the camera to render, as named to the engine.
            width: image width in pixels, a positive whole number. ``None``
                falls back to ``default_width``, then to the engine's own
                per-camera configuration.
            height: image height in pixels, same domain.

        Returns:
            The composited :class:`CompositeFrame`.

        Raises:
            ValueError: if ``width`` or ``height`` is supplied but is not a
                positive whole number of pixels.
            RuntimeError: if the backend's ``get_frame`` returns no depth
                buffer (e.g. the Newton backend, which renders RGB only) --
                compositing without depth would silently paint the background
                over/under the wrong pixels, and silent wrong output is
                forbidden. Use a depth-capable backend (MuJoCo, Isaac).
                Also if any layer (either ``get_frame`` buffer, or either
                buffer from the :class:`BackgroundRenderer`) is not the size
                the resolved camera declares: the returned frame reports that
                camera, so a layer at another size cannot be composited into
                the image its ``K`` describes.
        """
        for label, size in (("width", width), ("height", height)):
            if size is not None and (text := positive_whole_number_error(size, label, "HybridCompositor.render")):
                raise ValueError(text)
        # Read the requested size by membership, not truthiness: ``width or
        # self.default_width`` read a supplied ``0`` as "not supplied" and fell
        # back to the default size, so a size no engine can render returned a
        # frame at a different one.
        cam = self.sim.get_camera_params(
            camera_name,
            width=self.default_width if width is None else int(width),
            height=self.default_height if height is None else int(height),
        )
        fg_rgb, fg_depth = self.sim.get_frame(camera_name, width=cam.width, height=cam.height)
        if fg_depth is None:
            raise RuntimeError(
                f"Backend {type(self.sim).__name__} returned no depth buffer for camera "
                f"{camera_name!r}; depth-aware compositing requires a depth-capable backend "
                "(MuJoCo or Isaac). Newton renders RGB only."
            )
        fg_rgb = np.asarray(fg_rgb)
        if fg_rgb.ndim == 3 and fg_rgb.shape[2] == 4:
            fg_rgb = fg_rgb[..., :3]
        fg_rgb = fg_rgb.astype(np.uint8, copy=False)
        fg_depth = np.asarray(fg_depth, dtype=np.float32)

        bg_rgb, bg_depth = self._background_for(cam, camera_name)

        # Every layer must be the size ``cam`` declares, because ``cam`` is
        # what the returned frame reports: its ``K`` places the principal
        # point at (cam.width / 2, cam.height / 2), and consumers project into
        # an image of that size. Truncating to the shortest layer instead
        # (``h = min(fg.shape[0], bg.shape[0])``) returned a composite ``cam``
        # cannot describe - a 64x64 frame whose reported principal point
        # (160, 120) lies outside it - at a size the caller never asked for
        # and with nothing saying so. Refused for the reason the no-depth
        # refusal above states: silent wrong output is forbidden. It is also
        # the disposition the foreground's own producer already applies, since
        # ``IsaacSimEngine.get_frame`` raises on a size it cannot render
        # "rather than silently dropping the requested size".
        for label, remedy, *layers in (
            (
                "foreground",
                f"{type(self.sim).__name__}.get_frame must return arrays of the size it is asked for",
                ("rgb", fg_rgb),
                ("depth", fg_depth),
            ),
            (
                "background",
                f"{self.background.name!r} (a BackgroundRenderer) must return "
                "(cam.height, cam.width) arrays for the CameraParams it is given",
                ("rgb", bg_rgb),
                ("depth", bg_depth),
            ),
        ):
            for buffer_name, layer in layers:
                got_h, got_w = layer.shape[0], layer.shape[1]
                if (got_h, got_w) != (cam.height, cam.width):
                    raise RuntimeError(
                        f"HybridCompositor.render: the {label} {buffer_name} for camera "
                        f"{camera_name!r} is {got_w}x{got_h}, not the {cam.width}x{cam.height} "
                        f"the camera declares. The composited frame is described by its "
                        f"camera parameters, whose principal point is "
                        f"({cam.K[0, 2]:.1f}, {cam.K[1, 2]:.1f}) - a layer at another size "
                        f"cannot be composited into that image. {remedy}."
                    )

        # Foreground pixels are valid where the sim saw real geometry:
        # finite depth above epsilon (Isaac reports no-hit as 0 / inf) and
        # short of the far clip (MuJoCo pins sky to zfar).
        valid_fg = np.isfinite(fg_depth) & (fg_depth > self.depth_epsilon) & (fg_depth < cam.zfar * 0.999)

        catcher = None
        if self.shadow_plane_z is not None:
            # Shadow-catcher pass (issue #2323): foreground pixels whose depth
            # matches the analytic depth of the configured plane ARE the
            # catcher plane the scene renders. They never win the composite --
            # the photoreal backdrop owns that surface -- but the shading the
            # simulator rendered onto them carries the robot's cast shadow.
            plane_d = plane_depth(cam, self.shadow_plane_z)
            catcher = valid_fg & np.isfinite(plane_d) & (np.abs(fg_depth - plane_d) <= self.shadow_plane_tolerance)
            valid_fg = valid_fg & ~catcher

        winner = valid_fg & (fg_depth + 1e-3 < bg_depth)

        if catcher is not None and catcher.any():
            # A shadow is a ratio of received light, so it multiplies in
            # linear space (see strands_robots.rendering.color). Reference
            # brightness = the catcher's own typical unshadowed level (75th
            # percentile), so unshadowed plane pixels clip to factor 1 and
            # only genuinely darker pixels darken the backdrop.
            lum = relative_luminance(srgb_to_linear(fg_rgb))
            ref = float(np.percentile(lum[catcher], 75.0))
            if ref > 1e-6:
                factor = np.ones(lum.shape, dtype=np.float32)
                factor[catcher] = np.clip(lum[catcher] / ref, self.shadow_min_factor, 1.0)
                if self.feather_pixels > 0:
                    # Soften the shadow field the same way the fg/bg seam is
                    # softened, so the catcher's silhouette (plane edge,
                    # robot-contact edge) doesn't print as a hard outline.
                    factor = 1.0 - feather_mask(1.0 - factor, self.feather_pixels)
                shadowed_lin = srgb_to_linear(bg_rgb) * factor[..., None]
                bg_rgb = np.clip(linear_to_srgb(shadowed_lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)

        if self.feather_pixels > 0:
            alpha = feather_mask(winner, self.feather_pixels)
        else:
            alpha = winner.astype(np.float32)

        alpha = alpha[..., None]
        if self.blend_in_linear:
            # Seam blending is light arithmetic, so do it on linear light and
            # re-encode. alpha in {0, 1} pixels round-trip byte-identically.
            blended = alpha * srgb_to_linear(fg_rgb) + (1.0 - alpha) * srgb_to_linear(bg_rgb)
            composite_u8 = np.clip(linear_to_srgb(blended) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        else:
            composite = alpha * fg_rgb.astype(np.float32) + (1.0 - alpha) * bg_rgb.astype(np.float32)
            composite_u8 = np.clip(composite, 0, 255).astype(np.uint8)

        return CompositeFrame(
            rgb=composite_u8,
            foreground_rgb=fg_rgb,
            background_rgb=bg_rgb,
            foreground_mask=winner,
            depth=fg_depth,
            camera=cam,
        )

    def _background_for(self, cam: CameraParams, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(bg_rgb, bg_depth)`` for ``cam``, cached by camera."""
        key = _background_cache_key(camera_name, self.background.name, cam)
        cached = self._bg_cache.get(key)
        if cached is None:
            bg_rgb, bg_depth = self.background.render(cam)
            bg_rgb = np.asarray(bg_rgb)
            if bg_rgb.ndim == 3 and bg_rgb.shape[2] == 4:
                bg_rgb = bg_rgb[..., :3]
            cached = (bg_rgb.astype(np.uint8, copy=False), np.asarray(bg_depth, dtype=np.float32))
            # Keep the cache tiny -- only a handful of cameras are ever live.
            if len(self._bg_cache) > 16:
                self._bg_cache.clear()
            self._bg_cache[key] = cached
        return cached

    # ----- convenience ----- #

    def set_background(self, background: BackgroundRenderer) -> None:
        """Hot-swap the background renderer (e.g. from a UI dropdown)."""
        logger.info("HybridCompositor: switching background %s -> %s", self.background.name, background.name)
        self.background = background
        self._bg_cache.clear()

    def clear_caches(self) -> None:
        """Drop every cached background render.

        Not needed to track the camera: entries are keyed by the full
        :class:`CameraParams` -- pose, intrinsics, image size and both clip
        planes -- so a camera that moved, including one whose clip planes a
        scene recompile moved, is a miss rather than a stale hit. Use this to
        release the memory, or to force a re-render after mutating a background
        renderer in place (a swapped renderer is already handled by
        :meth:`set_background`).
        """
        self._bg_cache.clear()


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Soft-blend the mask edges by ``radius`` pixels using a box blur.

    Implementation: separable 1D ``np.convolve(..., mode='valid')`` on each
    axis over an edge-replicated pad. Runs in tens of ms for a 640x480 mask --
    well below a live demo's per-frame budget -- and avoids pulling in
    ``scipy.ndimage`` for one call.
    """
    if radius <= 0:
        return mask.astype(np.float32)
    m = mask.astype(np.float32)
    k = 2 * radius + 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    # Pad with edge replication so the blur doesn't pull in zeros at the borders.
    mp = np.pad(m, radius, mode="edge")
    blur_y = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), 0, mp)
    blur = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), 1, blur_y)
    return np.clip(blur, 0.0, 1.0).astype(np.float32)
