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

Concurrency contract: :meth:`HybridCompositor.render` calls straight into the
engine's ``get_frame`` on the calling thread -- thread-affinity is the
*backend's* contract (MuJoCo caches GL renderers per-thread; Isaac renders
must run on the SimulationApp thread). Call from a consistent thread, or
marshal calls yourself (see ``examples/mujoco_gs/compositor.py`` for a
single-render-thread wrapper).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .backgrounds import BackgroundRenderer, PanoramaBackground
from .camera import CameraParams

logger = logging.getLogger(__name__)


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
        background_rgb: ``(H, W, 3) uint8`` background-only render, for debugging.
        foreground_mask: ``(H, W) bool`` ``True`` where the foreground won the
            depth test (i.e. a simulated object is visible).
        depth: ``(H, W) float32`` foreground depth in meters (the simulation
            depth, since the foreground is what the user usually cares about
            measuring).
        camera: :class:`CameraParams` used to render this frame.
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
        default_width: image width if not overridden per call. ``None`` defers
            to the engine's per-camera configuration.
        default_height: image height if not overridden per call.
        feather_pixels: width (in pixels) of a soft edge blend between
            foreground and background to hide the offscreen-renderer's
            anti-aliasing seam. ``0`` disables feathering. Default ``1``.
        depth_epsilon: foreground depth at or below this (meters) is treated
            as "no geometry" -- those pixels show the background. Isaac's RTX
            depth annotator reports no-hit pixels as ``0`` (or non-finite);
            MuJoCo pins them to the far clip. Both extremes read as background.

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
    ) -> None:
        self.sim = sim
        self.background: BackgroundRenderer = background or PanoramaBackground()
        self.default_width = default_width
        self.default_height = default_height
        self.feather_pixels = max(0, int(feather_pixels))
        self.depth_epsilon = float(depth_epsilon)
        # Cache of background renders keyed by (camera_name, W, H, background
        # name) + a rounded hash of the camera pose/intrinsics. The background
        # only changes when the *camera* moves, not when the robot does -- so
        # during a live motion we recompute only the cheap sim foreground, not
        # the expensive panorama/gsplat pass.
        self._bg_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}

    # ----- main API ----- #

    def render(
        self,
        camera_name: str = "default",
        width: int | None = None,
        height: int | None = None,
    ) -> CompositeFrame:
        """Render one depth-composited frame of the current sim state.

        Raises:
            RuntimeError: if the backend's ``get_frame`` returns no depth
                buffer (e.g. the Newton backend, which renders RGB only) --
                compositing without depth would silently paint the background
                over/under the wrong pixels, and silent wrong output is
                forbidden. Use a depth-capable backend (MuJoCo, Isaac).
        """
        cam = self.sim.get_camera_params(
            camera_name, width=width or self.default_width, height=height or self.default_height
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

        # Align shapes defensively (foreground and background should both
        # match the camera resolution, but guard against off-by-one).
        h = min(fg_rgb.shape[0], bg_rgb.shape[0])
        w = min(fg_rgb.shape[1], bg_rgb.shape[1])
        fg_rgb, fg_depth = fg_rgb[:h, :w], fg_depth[:h, :w]
        bg_rgb, bg_depth = bg_rgb[:h, :w], bg_depth[:h, :w]

        # Foreground pixels are valid where the sim saw real geometry:
        # finite depth above epsilon (Isaac reports no-hit as 0 / inf) and
        # short of the far clip (MuJoCo pins sky to zfar).
        valid_fg = np.isfinite(fg_depth) & (fg_depth > self.depth_epsilon) & (fg_depth < cam.zfar * 0.999)
        winner = valid_fg & (fg_depth + 1e-3 < bg_depth)

        if self.feather_pixels > 0:
            alpha = feather_mask(winner, self.feather_pixels)
        else:
            alpha = winner.astype(np.float32)

        alpha = alpha[..., None]
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
        """Return ``(bg_rgb, bg_depth)`` for ``cam``, cached by camera pose."""
        pose_key = (
            camera_name,
            cam.width,
            cam.height,
            self.background.name,
            # Round the pose so tiny float jitter doesn't bust the cache.
            np.round(cam.T_world_cam, 4).tobytes(),
            np.round(cam.K, 3).tobytes(),
        )
        cached = self._bg_cache.get(pose_key)
        if cached is None:
            bg_rgb, bg_depth = self.background.render(cam)
            bg_rgb = np.asarray(bg_rgb)
            if bg_rgb.ndim == 3 and bg_rgb.shape[2] == 4:
                bg_rgb = bg_rgb[..., :3]
            cached = (bg_rgb.astype(np.uint8, copy=False), np.asarray(bg_depth, dtype=np.float32))
            # Keep the cache tiny -- only a handful of cameras are ever live.
            if len(self._bg_cache) > 16:
                self._bg_cache.clear()
            self._bg_cache[pose_key] = cached
        return cached

    # ----- convenience ----- #

    def set_background(self, background: BackgroundRenderer) -> None:
        """Hot-swap the background renderer (e.g. from a UI dropdown)."""
        logger.info("HybridCompositor: switching background %s -> %s", self.background.name, background.name)
        self.background = background
        self._bg_cache.clear()

    def clear_caches(self) -> None:
        """Drop cached background renders (call after a scene rebuild)."""
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
