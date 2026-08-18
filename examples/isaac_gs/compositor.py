"""Depth-aware compositor: Isaac RTX foreground over a 3DGS background.

The compositing itself is the library's backend-agnostic
:class:`strands_robots.rendering.HybridCompositor` (issue #1537): it pulls
the foreground through the public ``sim.get_frame`` / ``sim.get_camera_params``
APIs, so this module no longer re-implements the z-mask / cache / feather
maths against Isaac frames (the previous copy existed only because the old
MuJoCo compositor's body was entangled with MuJoCo renderer caching -- that
entanglement is gone).

``IsaacHybridCompositor`` remains as the example's named entry point; it only
pins the Isaac-appropriate default ``depth_epsilon`` (Isaac's RTX annotator
reports no-hit pixels as ``0`` or very large values -- both read as
background).

Note the composite frame's foreground mask is ``frame.foreground_mask``
(the library dataclass), not the old example-local ``frame.mask``.
"""

from __future__ import annotations

from typing import Optional

from strands_robots.rendering import BackgroundRenderer, CompositeFrame
from strands_robots.rendering import HybridCompositor as _LibraryHybridCompositor

__all__ = ["CompositeFrame", "IsaacHybridCompositor"]


class IsaacHybridCompositor(_LibraryHybridCompositor):
    """Composite an Isaac RTX robot over a photoreal background.

    Args:
        sim: a live ``IsaacSimulation`` (world created, camera added).
        background: any ``BackgroundRenderer`` from
            ``strands_robots.rendering``. Defaults to the procedural
            ``PanoramaBackground`` so the demo runs with zero ML deps. Pass a
            ``GsplatBackground(ply_path=...)`` for a real captured 3DGS scene
            (the digital-twin use case).
        feather_pixels: width of a soft foreground/background edge blend to
            hide the RTX anti-aliasing seam. ``0`` disables.
        depth_epsilon: foreground depth below this (meters) is treated as
            "no geometry / sky" -- those pixels show the background.

    Rendering must be driven from Isaac's main (SimulationApp) thread -- the
    same contract as ``sim.render`` (the demo app marshals renders through
    its main-thread queue).
    """

    def __init__(
        self,
        sim: "object",
        background: Optional[BackgroundRenderer] = None,
        feather_pixels: int = 1,
        depth_epsilon: float = 1e-4,
        **kwargs: object,
    ) -> None:
        super().__init__(
            sim,  # type: ignore[arg-type]
            background=background,
            feather_pixels=feather_pixels,
            depth_epsilon=depth_epsilon,
            # Forward everything else (shadow_plane_z, blend_in_linear, ...)
            # so the library compositor's options stay reachable -- a kwarg
            # accepted here but dropped before the library would be a silent
            # no-op (see AGENTS.md "Forward all advertised kwargs").
            **kwargs,  # type: ignore[arg-type]
        )
