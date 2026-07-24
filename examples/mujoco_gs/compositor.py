# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo-flavored wrapper around the library hybrid compositor.

The depth-aware compositing itself (per-pixel z-compare + background cache +
feathered seam) now lives in :class:`strands_robots.rendering.HybridCompositor`
(issue #1537), driven through the public ``sim.get_frame`` /
``sim.get_camera_params`` APIs -- so this example no longer reaches into
``sim.mj_model`` / ``sim.mj_data`` or maintains its own ``mujoco.Renderer``
cache (the backend caches GL renderers per-thread and applies the benchmark
viz option internally).

What stays example-side is the MuJoCo demo glue:

* **A single render thread.** MuJoCo's EGL/GL contexts are thread-affine, and
  this demo renders from several threads (the Gradio worker poll + the agent's
  tool thread). Funnelling every render through one worker thread means one
  GL context for the process lifetime; callers block for the result, so the
  public API stays synchronous.
* **Floor visibility.** Backgrounds that bring their own photoreal floor
  (``GsplatBackground(own_floor=True)``) hide MuJoCo's grid floor by zeroing
  the floor geoms' alpha (render-only -- collision still works).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from strands_robots.rendering import BackgroundRenderer, CompositeFrame
from strands_robots.rendering import HybridCompositor as _LibraryHybridCompositor

if TYPE_CHECKING:
    from strands_robots.simulation import Simulation

logger = logging.getLogger(__name__)

__all__ = ["CompositeFrame", "HybridCompositor"]


class HybridCompositor(_LibraryHybridCompositor):
    """Library :class:`~strands_robots.rendering.HybridCompositor` + MuJoCo demo glue.

    Args:
        sim: a live ``strands_robots.simulation.Simulation``.
        background: any :class:`BackgroundRenderer`. Defaults to a procedural
            panorama so the demo runs out of the box.
        default_width: image width if not overridden per call.
        default_height: image height if not overridden per call.
        feather_pixels: soft foreground/background edge width (``0`` disables).
    """

    def __init__(
        self,
        sim: Simulation,
        background: BackgroundRenderer | None = None,
        default_width: int = 640,
        default_height: int = 480,
        feather_pixels: int = 1,
    ) -> None:
        super().__init__(
            sim,
            background=background,
            default_width=default_width,
            default_height=default_height,
            feather_pixels=feather_pixels,
        )
        # ALL MuJoCo rendering runs on this single dedicated thread (GL
        # contexts are thread-affine; see module docstring).
        self._render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mjrender")
        # Original alpha of any floor geoms, so we can hide them (set alpha 0)
        # for backgrounds that bring their own photoreal floor, then restore.
        self._orig_floor_alpha: dict = {}
        self._apply_floor_visibility()

    # ----- main API ----- #

    def render(
        self,
        camera_name: str = "default",
        width: int | None = None,
        height: int | None = None,
    ) -> CompositeFrame:
        """Render the current sim state through the compositor.

        Thread-safe: the actual MuJoCo/GL work is submitted to the single
        render thread and this call blocks for the result, so it can be
        invoked from any thread (Gradio worker, agent tool thread, ...).
        """
        render = super().render  # bound library implementation
        return self._render_executor.submit(render, camera_name, width, height).result()

    def _apply_floor_visibility(self) -> None:
        """Hide built-in floor geoms (the robot's ``arm/floor`` and any MuJoCo
        ``ground`` plane) by setting their alpha to 0 when the active background
        supplies its own photoreal floor; restore them otherwise.

        Alpha is render-only, so the floor still collides (the cube keeps
        resting on it) -- it just stops painting MuJoCo's blue/white grid over
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
        except Exception:  # pragma: no cover -- cosmetics only
            logger.warning("Could not toggle floor visibility.", exc_info=True)

    # ----- convenience ----- #

    def set_background(self, background: BackgroundRenderer) -> None:
        """Hot-swap the background renderer (useful from a Gradio dropdown)."""
        super().set_background(background)
        # Show/hide the built-in MuJoCo floor depending on whether the new
        # background brings its own photoreal floor.
        self._apply_floor_visibility()

    def close(self) -> None:
        """Release the render thread."""
        try:
            self.clear_caches()
        except Exception:  # pragma: no cover
            pass
        self._render_executor.shutdown(wait=True)
