# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image-based lighting derived from the photoreal background (issue #2323).

The hybrid composite's foreground robot is lit by whatever lights the
simulation scene authors, while the background is a captured real scene --
lighting the robot with two unrelated hardcoded lights is the single biggest
"pasted-on" cue. This module derives the lighting *from the background
itself*:

* :func:`render_environment_map` / :func:`bake_environment_map` -- render an
  equirectangular environment map of the background as seen **from the robot's
  position in the world frame**, by rasterizing six 90-degree cube faces
  through the background's own :meth:`render` (so a
  :class:`~strands_robots.rendering.backgrounds.GsplatBackground`'s curated
  skybox alignment is applied -- unlike
  :func:`~strands_robots.rendering.backgrounds.bake_gsplat_panorama`, which
  bakes an unaligned backdrop from the scene centroid). The result feeds a
  dome light (USD ``DomeLight`` latlong texture), so the robot is lit by the
  room it stands in.
* :func:`derive_key_light` -- estimate the dominant light direction and color
  from an environment map, for aiming the shadow-casting key light the same
  way the captured scene's light actually falls.

The bake is renderer-agnostic: any :class:`BackgroundRenderer` works
(a :class:`PanoramaBackground` bakes back to (a resampling of) its own
panorama). Everything here except ``background.render`` itself is pure numpy.

Direction convention (shared with ``PanoramaBackground`` and
``bake_gsplat_panorama``): world ``+Z`` up; a direction ``(theta, phi)`` maps
to equirect ``u = (theta + pi) / 2pi`` (so ``+X`` is the image-center column)
and ``v = 0.5 - phi / pi`` (so ``+Z`` is the top row).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.utils import boolean_flag_error, positive_whole_number_error

from .camera import CameraParams
from .color import relative_luminance, srgb_to_linear

logger = logging.getLogger(__name__)

__all__ = [
    "KeyLightEstimate",
    "bake_environment_map",
    "derive_key_light",
    "environment_map_cache_path",
    "render_environment_map",
]


# The six outward cube faces as (forward, up) world directions. The +/-Z
# (zenith/nadir) faces need a different up vector -- +Z is parallel to the
# default up.
_CUBE_FACES: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
)


def _face_camera(
    fwd: np.ndarray,
    up: np.ndarray,
    origin: np.ndarray,
    face_size: int,
    znear: float,
    zfar: float,
) -> tuple[CameraParams, np.ndarray, np.ndarray]:
    """Build the 90-degree-FOV pinhole camera for one cube face.

    Returns ``(cam, right, cam_up)`` -- the camera plus the face's in-plane
    basis vectors, which the equirect reprojection needs to invert the
    projection.
    """
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    u = np.cross(right, fwd)
    Twc = np.eye(4)
    Twc[:3, :3] = np.stack([right, u, -fwd], axis=1)  # OpenGL: -Z = forward
    Twc[:3, 3] = origin
    f = 0.5 * face_size  # 90 deg FOV -> focal = size/2
    K = np.array([[f, 0.0, face_size / 2], [0.0, f, face_size / 2], [0.0, 0.0, 1.0]])
    cam = CameraParams(K=K, T_world_cam=Twc, width=face_size, height=face_size, znear=znear, zfar=zfar)
    return cam, right, u


def _resolution_error(
    context: str,
    *,
    face_size: Any,
    equi_w: Any,
    equi_h: Any,
) -> str | None:
    """Error text when a resolution knob cannot produce a usable map, else None.

    The three knobs that size the pixel grid share one domain, checked in one
    place so the render, the bake and the cache-path cannot disagree about which
    resolutions this module can honor.

    Only the domain is checked here, not the *quality* a resolution buys. A very
    coarse cube face is still a cube face: the reprojection scales by
    ``face_size - 1``, so 1, 2 and 3 each resolve almost nothing within a face
    (measured on a background with a gradient inside every face: 4, 4 and 5
    distinct colours in the whole map, growing smoothly from 4 upward). Where
    "too coarse" begins is a judgement about acceptable quality rather than a
    value the module cannot use, so it is left to the caller.

    Args:
        context: Calling function name, quoted in the message.
        face_size: Cube-face resolution in pixels.
        equi_w: Equirect width in pixels.
        equi_h: Equirect height in pixels.

    Returns:
        The refusal text, or ``None`` when every knob is usable.
    """
    for value, param in ((face_size, "face_size"), (equi_w, "equi_w"), (equi_h, "equi_h")):
        if text := positive_whole_number_error(value, param, context):
            return text
    return None


def _equirect_directions(equi_w: int, equi_h: int, half_texel: bool = False) -> np.ndarray:
    """Per-texel world-frame unit ray directions of an equirect grid.

    Matches ``PanoramaBackground``'s sampling convention: ``u`` in ``[0, 1]``
    maps to azimuth ``theta`` in ``[-pi, pi]`` (``+X`` at the center column)
    and ``v`` in ``[0, 1]`` top-to-bottom maps to elevation ``phi`` in
    ``[pi/2, -pi/2]`` (``+Z`` at the top row).

    ``half_texel`` samples at texel *centers* instead of the top-left grid
    line. The reprojection keeps the grid-line convention (bit-compatible
    with the panorama bake it replaced); :func:`derive_key_light` needs
    centers so its directions pair exactly with its per-row solid-angle
    weights -- on the offset grid a uniform map's directions cancel to zero
    instead of leaving a spurious polar bias.
    """
    off = 0.5 if half_texel else 0.0
    jj, ii = np.meshgrid(np.arange(equi_w) + off, np.arange(equi_h) + off)
    theta = (jj / equi_w) * 2 * np.pi - np.pi
    phi = np.pi / 2 - (ii / equi_h) * np.pi
    dx = np.cos(phi) * np.cos(theta)
    dy = np.cos(phi) * np.sin(theta)
    dz = np.sin(phi)
    return np.stack([dx, dy, dz], axis=-1)


def render_environment_map(
    background: object,
    origin_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
    face_size: int = 512,
    equi_w: int = 2048,
    equi_h: int = 1024,
    znear: float = 0.01,
    zfar: float = 1e3,
) -> np.ndarray:
    """Render the background into an equirectangular map seen from a point.

    Renders the six outward cube faces (90 deg FOV) from ``origin_world``
    through ``background.render`` and reprojects them onto the equirect grid.
    Because the background applies its own scene alignment inside ``render``,
    the result is the environment *as the world frame sees it* -- bake from
    the robot's position and the map is exactly what should light the robot.

    Args:
        background: any :class:`~strands_robots.rendering.backgrounds.BackgroundRenderer`.
        origin_world: world-frame viewpoint to bake from (e.g. a point just
            above the robot's base).
        face_size: cube-face resolution in pixels.
        equi_w: output equirect width in pixels.
        equi_h: output equirect height in pixels.
        znear: near clip distance handed to the background, meters.
        zfar: far clip distance handed to the background, meters.

    Returns:
        ``(equi_h, equi_w, 3) uint8`` equirectangular environment map.

    Raises:
        ValueError: if any resolution knob is not a positive whole number.
            Checked before the six background renders,
            which are GPU-bound for a
            :class:`~strands_robots.rendering.backgrounds.GsplatBackground`.
    """
    if text := _resolution_error("render_environment_map", face_size=face_size, equi_w=equi_w, equi_h=equi_h):
        raise ValueError(text)
    # Normalize to plain ints: the shared domain accepts an integral float and a
    # NumPy integer, and both index the grid below.
    face_size, equi_w, equi_h = int(face_size), int(equi_w), int(equi_h)
    origin = np.asarray(origin_world, dtype=np.float64).reshape(3)
    face_imgs: list[np.ndarray] = []
    face_bases: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fwd_t, up_t in _CUBE_FACES:
        fwd = np.asarray(fwd_t, dtype=np.float64)
        cam, right, u = _face_camera(fwd, np.asarray(up_t, dtype=np.float64), origin, face_size, znear, zfar)
        rgb, _ = background.render(cam)  # type: ignore[attr-defined]
        face_imgs.append(np.asarray(rgb, dtype=np.float32))
        face_bases.append((fwd, right, u))

    dirs = _equirect_directions(equi_w, equi_h)
    pano = np.zeros((equi_h, equi_w, 3), np.float32)
    best = np.full((equi_h, equi_w), -1e9, np.float32)
    for img, (fwd, right, u) in zip(face_imgs, face_bases):
        d_f = dirs @ fwd
        sel = d_f > 1e-6  # rays in this face's hemisphere
        # Pick the face with the largest forward component per pixel.
        take = sel & (d_f > best)
        if not take.any():
            continue
        s = dirs[take] / d_f[take][:, None]  # project to image plane (z=1)
        u_img = s @ right
        v_img = s @ u
        inside = (np.abs(u_img) <= 1.0) & (np.abs(v_img) <= 1.0)
        col = np.clip(((u_img + 1) * 0.5 * (face_size - 1)).astype(int), 0, face_size - 1)
        row = np.clip(((1 - (v_img + 1) * 0.5) * (face_size - 1)).astype(int), 0, face_size - 1)
        idx = np.where(take)
        ri, ci = idx[0][inside], idx[1][inside]
        pano[ri, ci] = img[row[inside], col[inside]]
        best[ri, ci] = d_f[take][inside]
    return np.clip(pano, 0, 255).astype(np.uint8)


def environment_map_cache_path(
    scene_path: str | Path,
    origin_world: tuple[float, float, float],
    face_size: int = 512,
    equi_w: int = 2048,
    equi_h: int = 1024,
) -> Path:
    """Default cache path for a baked environment map, next to its scene file.

    A warm output file short-circuits :func:`bake_environment_map`, so the
    path must identify the pixels it holds: the bake origin and every
    resolution knob change the image, and a path that ignored them would make
    a second caller's parameters a silent no-op (the exact failure
    ``bake_gsplat_panorama`` documents for its own cache).

    Args:
        scene_path: the background's scene file (``.ply`` / ``.spz`` /
            panorama image); the map is cached beside it.
        origin_world: the world-frame bake viewpoint, encoded into the name
            at millimeter precision.
        face_size: cube-face resolution in pixels.
        equi_w: equirect width in pixels.
        equi_h: equirect height in pixels.

    Returns:
        The cache path (a ``.png`` -- environment maps feed light sampling,
        so JPEG block artifacts are not welcome).

    Raises:
        ValueError: if any resolution knob is not a positive whole number. The name
            encodes the resolutions, so a knob this module refuses to render must
            not be named a cache entry.
    """
    if text := _resolution_error("environment_map_cache_path", face_size=face_size, equi_w=equi_w, equi_h=equi_h):
        raise ValueError(text)
    # Normalize before formatting: this name IS the cache key, so an integral
    # float must not spell a second file for pixels already baked.
    face_size, equi_w, equi_h = int(face_size), int(equi_w), int(equi_h)
    scene_path = Path(scene_path)
    ox, oy, oz = (float(c) for c in origin_world)
    stem = f"{scene_path.stem}_env_{equi_w}x{equi_h}_f{face_size}_o{ox:+.3f}_{oy:+.3f}_{oz:+.3f}"
    return scene_path.with_name(f"{stem}.png")


def bake_environment_map(
    background: object,
    out_path: str | Path,
    origin_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
    face_size: int = 512,
    equi_w: int = 2048,
    equi_h: int = 1024,
) -> Path:
    """Bake :func:`render_environment_map` to an image file, with caching.

    Baking is six background renders (GPU-bound for a
    :class:`~strands_robots.rendering.backgrounds.GsplatBackground`), so a
    non-empty ``out_path`` short-circuits the whole pass. The caller owns the
    path -- name it with :func:`environment_map_cache_path` so the name
    encodes the origin and resolutions that determine the pixels.

    Args:
        background: any :class:`~strands_robots.rendering.backgrounds.BackgroundRenderer`.
        out_path: where to write the equirect image (format from the suffix).
        origin_world: world-frame viewpoint to bake from.
        face_size: cube-face resolution in pixels.
        equi_w: output equirect width in pixels.
        equi_h: output equirect height in pixels.

    Returns:
        The path to the written (or cached) environment map.

    Raises:
        ValueError: if any resolution knob is not a positive whole number.
            Checked before the cache probe, so an
            unusable resolution never writes a file the short-circuit would then
            serve on every later call.
    """
    if text := _resolution_error("bake_environment_map", face_size=face_size, equi_w=equi_w, equi_h=equi_h):
        raise ValueError(text)
    face_size, equi_w, equi_h = int(face_size), int(equi_w), int(equi_h)
    out = Path(out_path)
    if out.exists() and out.stat().st_size > 0:
        return out
    env = render_environment_map(
        background,
        origin_world=origin_world,
        face_size=face_size,
        equi_w=equi_w,
        equi_h=equi_h,
    )
    from PIL import Image

    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(env).save(out)
    logger.info("Baked environment map -> %s", out)
    return out


@dataclass(frozen=True)
class KeyLightEstimate:
    """Dominant light estimated from an environment map.

    Attributes:
        direction: world-frame unit vector pointing *toward* the light (the
            direction the light comes from, not the direction it travels).
        color: linear-light RGB chromaticity of the light, normalized so the
            largest channel is ``1.0`` (feed a light's color input directly;
            scale brightness with the light's own intensity knob).
        azimuth_deg: ``direction``'s azimuth in degrees (``atan2(y, x)``).
        elevation_deg: ``direction``'s elevation above the horizon in degrees.
    """

    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    azimuth_deg: float
    elevation_deg: float


def derive_key_light(
    env_map: np.ndarray,
    brightest_fraction: float = 0.02,
    upper_hemisphere: bool = True,
) -> KeyLightEstimate:
    """Estimate the dominant light direction + color of an environment map.

    Takes the brightest ``brightest_fraction`` of texels by linear luminance,
    weights each by luminance times its solid angle (equirect rows shrink by
    ``cos(elevation)``), and averages their world directions and linear
    colors. Deliberately an estimate: it aims the *key* light; the rest of
    the environment lights the robot through the dome texture itself.

    Args:
        env_map: ``(H, W, 3)`` equirect image, ``uint8`` or float in
            ``[0, 1]``, in the direction convention of this module.
        brightest_fraction: fraction of texels (by luminance) treated as "the
            light", in ``(0, 1]``.
        upper_hemisphere: search above the horizon only (default). A bright
            floor/table under the bake point easily out-weighs the actual
            light sources (measured on the ``tabletop`` preset: the full
            sphere derives an elevation of -76 deg -- a key light from
            *underneath*), and below-horizon radiance is bounce, which the
            dome texture already provides as fill. Pass ``False`` to search
            the full sphere. It selects a *posture* rather than scaling a
            quantity, so it is checked against
            :func:`~strands_robots.utils.boolean_flag_error` rather than read
            by truthiness: ``"false"`` is a truthy string, so reading it that
            way would search the hemisphere the caller asked to leave out.

    Returns:
        The :class:`KeyLightEstimate`.

    Raises:
        ValueError: if ``brightest_fraction`` is outside ``(0, 1]``, if
            ``upper_hemisphere`` is not a boolean, if ``env_map`` is not an
            ``(H, W, 3)`` image, or if no dominant direction exists in the
            searched region (bright texels cancel out, or the searched
            hemisphere is black), in which case there is no key light worth
            authoring and the caller should keep its default lighting.
    """
    if not isinstance(brightest_fraction, (int, float)) or isinstance(brightest_fraction, bool):
        raise ValueError(
            f"derive_key_light: brightest_fraction must be a number in (0, 1], got {brightest_fraction!r}."
        )
    frac = float(brightest_fraction)
    if not np.isfinite(frac) or frac <= 0.0 or frac > 1.0:
        raise ValueError(
            f"derive_key_light: brightest_fraction must be a number in (0, 1], got {brightest_fraction!r}."
        )
    # Checked beside brightest_fraction rather than read by truthiness below:
    # every spelling of "off" a caller reaches for is a truthy string, so
    # upper_hemisphere="false" would search the hemisphere it asks to skip,
    # while an undeclared falsy value (None, 0, "") would silently select the
    # full sphere this argument's own docstring warns aims a key light from
    # underneath. Both choose a search region, not a magnitude, so the flag is
    # checked rather than parsed.
    if text := boolean_flag_error(upper_hemisphere, "upper_hemisphere", "derive_key_light"):
        raise ValueError(text)
    env = np.asarray(env_map)
    if env.ndim != 3 or env.shape[2] != 3:
        raise ValueError(f"derive_key_light: env_map must be an (H, W, 3) image, got shape {env.shape}.")

    linear = srgb_to_linear(env)
    lum = relative_luminance(linear)
    H, W = lum.shape
    dirs = _equirect_directions(W, H, half_texel=True)
    # Solid angle of an equirect texel shrinks toward the poles.
    phi = np.pi / 2 - ((np.arange(H) + 0.5) / H) * np.pi
    solid = np.cos(phi)[:, None]

    region = dirs[..., 2] > 0.0 if upper_hemisphere else np.ones_like(lum, dtype=bool)
    searched = lum[region]
    if searched.size == 0 or float(searched.max()) <= 0.0:
        raise ValueError(
            "derive_key_light: no light in the searched region (the map is black "
            + (
                "above the horizon -- pass upper_hemisphere=False to search the full sphere, or "
                if upper_hemisphere
                else "; "
            )
            + "keep the caller's default key light)."
        )
    threshold = float(np.quantile(searched, max(0.0, 1.0 - frac)))
    sel = region & (lum >= threshold)
    weights = (lum * solid)[sel]
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("derive_key_light: environment map is black -- no light to derive.")
    mean_dir = (dirs[sel] * weights[:, None]).sum(axis=0) / total
    norm = float(np.linalg.norm(mean_dir))
    if norm < 1e-3:
        raise ValueError(
            "derive_key_light: the brightest texels cancel out (no dominant direction); "
            "keep the caller's default key light."
        )
    direction = mean_dir / norm
    mean_color = (linear[sel] * weights[:, None]).sum(axis=0) / total
    peak = float(mean_color.max())
    color = mean_color / peak if peak > 0 else np.ones(3, np.float32)
    azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])))
    elevation = float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
    return KeyLightEstimate(
        direction=(float(direction[0]), float(direction[1]), float(direction[2])),
        color=(float(color[0]), float(color[1]), float(color[2])),
        azimuth_deg=azimuth,
        elevation_deg=elevation,
    )
