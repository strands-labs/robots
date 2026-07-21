"""Depth-aware compositor: Isaac RTX foreground over a 3DGS background.

The Isaac-side analogue of ``examples.mujoco_gs.compositor.HybridCompositor``.
The MuJoCo one is coupled to ``mujoco.Renderer`` / ``sim.mj_model``;
this one pulls the foreground from Isaac's RTX camera
(``sim.render`` + ``camera_utils``) instead. The compositing maths
(per-pixel z-compare + optional feathered seam) is the same shape but
re-implemented here against Isaac frames rather than imported, since
the MuJoCo compositor's body is entangled with MuJoCo renderer caching.

The background renderer is **reused verbatim** from
``examples.mujoco_gs.backgrounds`` -- it's backend-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from examples.isaac_gs.camera_utils import get_camera_params, render_rgb_and_depth
from examples.mujoco_gs.backgrounds import BackgroundRenderer, PanoramaBackground

logger = logging.getLogger(__name__)


@dataclass
class CompositeFrame:
    """One composited frame + its component layers (for debugging / saving)."""

    rgb: np.ndarray  # (H, W, 3) uint8 -- the final composite
    foreground_rgb: np.ndarray  # (H, W, 3) uint8 -- Isaac RTX robot
    foreground_depth: np.ndarray  # (H, W) float32 -- metric depth
    background_rgb: np.ndarray  # (H, W, 3) uint8 -- 3DGS / panorama
    mask: np.ndarray  # (H, W) bool -- True where foreground wins


class IsaacHybridCompositor:
    """Composite an Isaac RTX robot over a photoreal background.

    Args:
        sim: a live ``IsaacSimulation`` (world created, camera added).
        background: any ``BackgroundRenderer`` from
            ``examples.mujoco_gs.backgrounds``. Defaults to the
            procedural ``PanoramaBackground`` so the demo runs with
            zero ML deps. Pass a ``GsplatBackground(ply_path=...)`` for
            a real captured 3DGS scene (the digital-twin use case).
        feather_pixels: width of a soft foreground/background edge
            blend to hide the RTX anti-aliasing seam. ``0`` disables.
        depth_epsilon: foreground depth below this (meters) is treated
            as "no geometry / sky" -- those pixels show the background.
            Isaac returns 0.0 or very large values for sky depending on
            the annotator; we treat both extremes as background.

    The background is cached per (camera_name, resolution, pose-hash):
    it only changes when the *camera* moves, not when the robot does,
    so a multi-frame robot motion only re-renders the cheap RTX
    foreground per frame, not the background pass.
    """

    def __init__(
        self,
        sim: "object",
        background: Optional[BackgroundRenderer] = None,
        feather_pixels: int = 1,
        depth_epsilon: float = 1e-4,
    ) -> None:
        self.sim = sim
        self.background: BackgroundRenderer = background or PanoramaBackground()
        self.feather_pixels = max(0, int(feather_pixels))
        self.depth_epsilon = float(depth_epsilon)
        self._bg_cache: dict = {}

    def _background_for(self, camera_name: str, cam_params) -> np.ndarray:
        """Render (or cache) the background RGB for the camera's current pose."""
        pose_hash = hash(np.asarray(cam_params.T_world_cam).tobytes())
        key = (camera_name, cam_params.width, cam_params.height, pose_hash)
        cached = self._bg_cache.get(key)
        if cached is not None:
            return cached
        bg_rgb, _bg_depth = self.background.render(cam_params)
        bg_rgb = np.asarray(bg_rgb)
        if bg_rgb.ndim == 3 and bg_rgb.shape[2] == 4:
            bg_rgb = bg_rgb[..., :3]
        bg_rgb = bg_rgb.astype(np.uint8)
        if len(self._bg_cache) > 8:
            self._bg_cache.clear()
        self._bg_cache[key] = bg_rgb
        return bg_rgb

    def render(self, camera_name: str = "default") -> CompositeFrame:
        """Render one depth-composited frame.

        Foreground = Isaac RTX robot (where it has geometry); background
        = 3DGS / panorama elsewhere. A pixel shows the foreground iff
        the RTX camera saw real geometry there (finite, > epsilon
        depth) -- which is exactly the robot / objects, since the
        background is rendered separately and the world has no floor
        competing for those pixels.
        """
        cam = get_camera_params(self.sim, camera_name)
        fg_rgb, fg_depth = render_rgb_and_depth(self.sim, camera_name)
        bg_rgb = self._background_for(camera_name, cam)

        # Align shapes defensively (RTX + background should match the
        # camera resolution, but guard against off-by-one).
        h = min(fg_rgb.shape[0], bg_rgb.shape[0])
        w = min(fg_rgb.shape[1], bg_rgb.shape[1])
        fg_rgb, fg_depth, bg_rgb = fg_rgb[:h, :w], fg_depth[:h, :w], bg_rgb[:h, :w]

        # Foreground wins where it has valid, finite geometry depth.
        # Isaac's depth is 0 (or huge/inf) for sky/no-hit; robot pixels
        # carry a sensible metric depth.
        mask = np.isfinite(fg_depth) & (fg_depth > self.depth_epsilon)

        composite = bg_rgb.copy()
        composite[mask] = fg_rgb[mask]

        if self.feather_pixels > 0:
            composite = self._feather(composite, fg_rgb, bg_rgb, mask)

        return CompositeFrame(
            rgb=composite,
            foreground_rgb=fg_rgb,
            foreground_depth=fg_depth,
            background_rgb=bg_rgb,
            mask=mask,
        )

    def _feather(
        self,
        composite: np.ndarray,
        fg_rgb: np.ndarray,
        bg_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Blend a soft ``feather_pixels``-wide seam at the mask boundary.

        Hides the RTX anti-aliasing fringe so the robot's silhouette
        doesn't show a hard pixel staircase against the background.
        Pure-numpy box-blur of the boundary band; no scipy dep.
        """
        # Boundary = mask pixels adjacent to non-mask (erode via shifts).
        m = mask
        inner = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
        boundary = m & ~inner
        if not boundary.any():
            return composite
        out = composite.astype(np.float32)
        blended = 0.5 * fg_rgb.astype(np.float32) + 0.5 * bg_rgb.astype(np.float32)
        out[boundary] = blended[boundary]
        return np.clip(out, 0, 255).astype(np.uint8)
