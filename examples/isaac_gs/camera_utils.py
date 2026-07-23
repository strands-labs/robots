"""Isaac camera params + RGB/depth helpers -- now provided by the library.

The intrinsics/extrinsics query (including the fixed USD-prim -> OpenGL
optical-frame basis correction) and the raw-frame render are public backend
APIs since issue #1537:

* ``sim.get_camera_params(name)`` -> :class:`strands_robots.rendering.CameraParams`
  (applies the ``PRIM_TO_GL`` correction inside the Isaac backend, so the
  pose is already in the +X right / +Y up / -Z forward frame the background
  renderers expect).
* ``sim.get_frame(name)`` -> raw ``(rgb_uint8, depth_float32)`` without the
  agent-tool PNG envelope, raising (never blank frames) on degraded paths.

The thin wrappers below keep this example's historical call shape
(``fn(sim, camera_name)``); new code should call the ``sim`` methods
directly. No private ``sim._cameras[...].handle`` access remains.
"""

from __future__ import annotations

import numpy as np

# Backend-agnostic CameraParams from the library; kept under the example's
# historical alias so existing imports keep working.
from strands_robots.rendering import CameraParams as IsaacCameraParams

__all__ = ["IsaacCameraParams", "get_camera_params", "render_rgb_and_depth"]


def get_camera_params(sim: "object", camera_name: str) -> IsaacCameraParams:
    """Build :class:`IsaacCameraParams` for a camera added via ``add_camera``."""
    return sim.get_camera_params(camera_name)  # type: ignore[attr-defined]


def render_rgb_and_depth(sim: "object", camera_name: str) -> "tuple[np.ndarray, np.ndarray]":
    """Render the Isaac RTX foreground RGB + metric depth for a camera.

    Pixels with no geometry (sky / background) come back from Isaac as
    zero / very large / non-finite depth; the compositor treats those as
    "see the background through here".
    """
    rgb, depth = sim.get_frame(camera_name)  # type: ignore[attr-defined]
    assert depth is not None  # the Isaac backend always produces a depth buffer
    return rgb, depth
