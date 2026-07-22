# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid (MuJoCo + photoreal background) compositor.

This is the analogue of the in-browser MuJoCo-GS-Web hybrid renderer
(``mujoco_wasm`` + ``@sparkjsdev/spark``), implemented as a tiny Python
component on top of ``strands_robots.simulation.Simulation``:

    +--------+        +---------------------+        +-----------+
    | MuJoCo |  RGB,  |                     |        |   final   |
    |  step  |--D-->  | per-pixel z-compare |  --->  |  composite|
    +--------+        |                     |        |   frame   |
                      +---------------------+        +-----------+
        ^                       ^
        |                       | RGB, D
        |               +-------+-------+
        |               | Background    |
        +-- camera ---->| (panorama or  |
            params      |   gsplat)     |
                        +---------------+

Per-pixel rule: ``foreground_depth < background_depth`` → foreground wins.
This gives the same "physics objects correctly occlude / are occluded by GS
geometry" property that MuJoCo-GS-Web shows off.

Stateless (apart from a cached :class:`BackgroundRenderer`); safe to call
from a Gradio callback or a Strands tool.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .backgrounds import BackgroundRenderer, PanoramaBackground
from .camera_utils import CameraParams, get_camera_params, render_rgb_and_depth

if TYPE_CHECKING:
    from strands_robots.simulation import Simulation

logger = logging.getLogger(__name__)


@dataclass
class CompositeFrame:
    """Output of :meth:`HybridCompositor.render`.

    Attributes:
        rgb: ``(H, W, 3) uint8`` final composited image.
        foreground_rgb: ``(H, W, 3) uint8`` MuJoCo-only render, for debugging.
        background_rgb: ``(H, W, 3) uint8`` background-only render, for debugging.
        foreground_mask: ``(H, W) bool`` ``True`` where the foreground won the
            depth test (i.e. a MuJoCo object is visible).
        depth: ``(H, W) float32`` foreground depth in meters (the MuJoCo
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
    """Render and composite MuJoCo + photoreal background frames.

    Args:
        sim: a live ``strands_robots.simulation.Simulation``.
        background: any :class:`BackgroundRenderer`. Defaults to a procedural
            panorama so the demo runs out of the box.
        default_width: image width if not overridden per call.
        default_height: image height if not overridden per call.
        feather_pixels: width (in pixels) of a soft edge blend between
            foreground and background to hide the offscreen-renderer's
            anti-aliasing seam. ``0`` disables feathering. Default ``1``.

    Example:

        >>> from strands_robots.simulation import Simulation
        >>> sim = Simulation()
        >>> sim.create_world()
        >>> sim.add_robot("arm", data_config="so101")
        >>> sim.add_object("cube", shape="box", position=[0.2, 0.2, 0.05],
        ...                color=[1, 0, 0, 1])
        >>> sim.add_camera("front", position=[0.4, -0.5, 0.3],
        ...                target=[0.0, 0.0, 0.1])
        >>> sim.step(20)
        >>> compositor = HybridCompositor(sim)
        >>> frame = compositor.render(camera_name="front")
        >>> frame.rgb.shape
        (480, 640, 3)
    """

    def __init__(
        self,
        sim: Simulation,
        background: BackgroundRenderer | None = None,
        default_width: int = 640,
        default_height: int = 480,
        feather_pixels: int = 1,
    ) -> None:
        self.sim = sim
        self.background: BackgroundRenderer = background or PanoramaBackground()
        self.default_width = int(default_width)
        self.default_height = int(default_height)
        self.feather_pixels = max(0, int(feather_pixels))
        # ALL MuJoCo rendering runs on this single dedicated thread. MuJoCo's
        # EGL/GL contexts are thread-affine, and touching them from multiple
        # threads (the Gradio worker poll + the agent's tool thread) segfaults
        # at the C level. Funnelling every render through one worker thread is
        # the canonical fix: one EGL context, created and used on one thread,
        # for the process lifetime. Callers (.render()) submit work and block
        # for the result, so the public API stays synchronous.
        self._render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mjrender")
        # Cache of background renders keyed by (camera_name, W, H) + a hash of
        # the camera pose. The background only changes when the *camera* moves,
        # not when the robot does — so during a live motion we recompute only
        # the cheap MuJoCo foreground, not the expensive panorama/gsplat pass.
        self._bg_cache: dict = {}
        # Cache of CameraParams keyed by (camera_name, W, H). The demo cameras
        # are static, so we compute intrinsics/extrinsics once (which calls
        # mj_forward — a WRITE to mj_data) and reuse them. This removes
        # mj_forward from the per-frame render path, so rendering (which only
        # *reads* mj_data) never races a concurrent sim.step() *write* on the
        # agent thread. Cleared on scene reset.
        self._cam_cache: dict = {}
        # Cache of mujoco.Renderer objects keyed by (W, H), only ever touched
        # on the render-executor thread (so a plain dict is safe). Reusing a
        # renderer instead of allocating a GL framebuffer every frame is the
        # single biggest speedup for live streaming (~130 ms → ~11 ms/frame).
        self._renderer_cache: dict = {}
        # Original alpha of any floor geoms, so we can hide them (set alpha 0)
        # for backgrounds that bring their own photoreal floor, then restore.
        self._orig_floor_alpha: dict = {}
        self._apply_floor_visibility()

    def _renderer_for(self, width: int, height: int):
        """Return a cached ``mujoco.Renderer`` for ``(W, H)``.

        Only called on the render-executor thread, so no locking is needed.
        """
        import mujoco

        key = (int(width), int(height))
        r = self._renderer_cache.get(key)
        if r is None:
            r = mujoco.Renderer(self.sim.mj_model, height=int(height), width=int(width))
            if len(self._renderer_cache) > 4:
                for old in self._renderer_cache.values():
                    try:
                        old.close()
                    except Exception:  # pragma: no cover
                        pass
                self._renderer_cache.clear()
            self._renderer_cache[key] = r
        return r

    # ----- main API ----- #

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
            cached = self.background.render(cam)
            # Keep the cache tiny — only the few demo cameras are ever used.
            if len(self._bg_cache) > 16:
                self._bg_cache.clear()
            self._bg_cache[pose_key] = cached
        return cached

    def render(
        self,
        camera_name: str = "default",
        width: int | None = None,
        height: int | None = None,
    ) -> CompositeFrame:
        """Render the current sim state through the compositor.

        Thread-safe: the actual MuJoCo/GL work is submitted to the single
        render thread and this call blocks for the result, so it can be
        invoked from any thread (Gradio worker, agent tool thread, …).
        """
        W = int(width or self.default_width)
        H = int(height or self.default_height)
        return self._render_executor.submit(self._render_sync, camera_name, W, H).result()

    def _viz_option(self):
        """Return the sim's benchmark viz option (``MjvOption``) if present.

        Benchmark adapters (e.g. LIBERO) stash a cleaned-up
        ``mujoco.MjvOption`` on ``sim._world._backend_state["viz_option"]`` that
        hides collision geoms and site markers — without it our separate
        renderer paints the green/blue debug patches over the robot. Plain
        scenes (the SO-101 demo) have no such option, so this returns ``None``
        and rendering uses MuJoCo defaults (which look correct there).
        """
        try:
            state = getattr(self.sim._world, "_backend_state", None)
            if isinstance(state, dict):
                return state.get("viz_option")
        except Exception:  # pragma: no cover
            pass
        return None

    def _apply_floor_visibility(self) -> None:
        """Hide built-in floor geoms (the robot's ``arm/floor`` and any MuJoCo
        ``ground`` plane) by setting their alpha to 0 when the active background
        supplies its own photoreal floor; restore them otherwise.

        Alpha is render-only, so the floor still collides (the cube keeps
        resting on it) — it just stops painting MuJoCo's blue/white grid over
        the GS scene's floor.
        """
        try:
            import mujoco

            model = self.sim.mj_model
            hide = bool(getattr(self.background, "own_floor", False))
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                is_floor = name == "ground" or name == "floor" or name.endswith("/floor")
                if not is_floor:
                    continue
                if gid not in self._orig_floor_alpha:
                    self._orig_floor_alpha[gid] = float(model.geom_rgba[gid, 3])
                model.geom_rgba[gid, 3] = 0.0 if hide else self._orig_floor_alpha[gid]
        except Exception:  # pragma: no cover — cosmetics only
            logger.warning("Could not toggle floor visibility.", exc_info=True)

    def _render_sync(self, camera_name: str, W: int, H: int) -> CompositeFrame:
        """The real render — only ever runs on the render-executor thread."""
        cam_key = (camera_name, W, H)
        cam = self._cam_cache.get(cam_key)
        if cam is None:
            # Computed once per static camera (calls mj_forward). Safe here at
            # warm-up / between motions; cached so the per-frame path never
            # writes mj_data.
            cam = get_camera_params(self.sim, camera_name=camera_name, width=W, height=H)
            self._cam_cache[cam_key] = cam
        renderer = self._renderer_for(W, H)
        fg_rgb, fg_depth = render_rgb_and_depth(
            self.sim, camera_name, W, H, renderer=renderer, scene_option=self._viz_option()
        )
        bg_rgb, bg_depth = self._background_for(cam, camera_name)

        # Foreground "wins" where it is in front of the background, OR where
        # the panorama-style background reports infinite depth and the
        # foreground is finite (i.e. there's any MuJoCo geometry there).
        # We treat MuJoCo's "saw the sky" pixels (depth pinned to zfar) as
        # background.
        valid_fg = fg_depth < (cam.zfar * 0.999)
        winner = (fg_depth + 1e-3 < bg_depth) & valid_fg

        if self.feather_pixels > 0:
            alpha = _feather_mask(winner, self.feather_pixels)
        else:
            alpha = winner.astype(np.float32)

        alpha = alpha[..., None]
        composite = alpha * fg_rgb.astype(np.float32) + (1.0 - alpha) * bg_rgb.astype(np.float32)
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        return CompositeFrame(
            rgb=composite,
            foreground_rgb=fg_rgb,
            background_rgb=bg_rgb,
            foreground_mask=winner,
            depth=fg_depth,
            camera=cam,
        )

    # ----- convenience ----- #

    def set_background(self, background: BackgroundRenderer) -> None:
        """Hot-swap the background renderer (useful from a Gradio dropdown)."""
        logger.info("HybridCompositor: switching background %s → %s", self.background.name, background.name)
        self.background = background
        self._bg_cache.clear()
        # Show/hide the built-in MuJoCo floor depending on whether the new
        # background brings its own photoreal floor.
        self._apply_floor_visibility()

    def clear_caches(self) -> None:
        """Drop cached backgrounds and MuJoCo renderers.

        Call this after the scene is rebuilt (``reset``) — the cached
        renderers reference the previous ``mj_model`` and would be stale.
        The renderer ``close()`` runs on the render thread (where the GL
        context lives).
        """
        self._bg_cache.clear()
        self._cam_cache.clear()

        def _close_renderers():
            for r in self._renderer_cache.values():
                try:
                    r.close()
                except Exception:  # pragma: no cover
                    pass
            self._renderer_cache.clear()

        self._render_executor.submit(_close_renderers).result()

    def close(self) -> None:
        """Release the render thread and its GL resources."""
        try:
            self.clear_caches()
        except Exception:  # pragma: no cover
            pass
        self._render_executor.shutdown(wait=True)


def _feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Soft-blend the mask edges by ``radius`` pixels using a box blur.

    Implementation: separable 1D ``np.convolve(..., mode='same')`` on each
    axis. Runs in tens of ms for a 640×480 mask — well below the demo's
    per-frame budget — and we avoid pulling in scipy.ndimage just for one
    call.
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
    return np.clip(blur, 0.0, 1.0)
