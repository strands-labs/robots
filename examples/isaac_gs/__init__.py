"""Isaac Sim + 3D Gaussian Splatting hybrid render — strands-robots example.

The Isaac-Sim companion to ``examples/mujoco_gs``. Same core idea --
a simulated robot composited against a photoreal 3DGS (or procedural
panorama) background with per-pixel depth-aware occlusion -- but with
a deliberately different *motivation*:

* ``mujoco_gs`` exists because MuJoCo's renderer isn't photoreal, so
  it composites the robot against a 3DGS scene to *gain*
  photorealism.
* Isaac Sim's RTX renderer is **already** photoreal. So this example
  isn't about fixing a renderer -- it's about the **digital-twin /
  real2sim** use case: dropping an RTX-rendered *simulated* robot
  into a **real-world-captured 3DGS environment** with correct
  depth-aware occlusion, so the sim robot looks like it's standing in
  the captured real scene.

Reuse
-----
The background renderers (``PanoramaBackground`` procedural default,
``GsplatBackground`` for real ``.ply`` / ``.spz`` captures) are
backend-agnostic and **reused verbatim** from ``examples.mujoco_gs.backgrounds``
-- they only need a pinhole ``CameraParams`` (intrinsics + world pose)
and numpy, nothing MuJoCo-specific. This example supplies those
``CameraParams`` from the Isaac RTX camera instead of MuJoCo's
``mj_data``, and z-composites Isaac's RGB + metric depth over the
background.

Runtime dependencies
--------------------
Needs the Phase-2 camera + render wiring:

* `PR #61 <https://github.com/strands-labs/robots-sim/pull/61>`_ --
  ``add_camera`` constructs the ``omni.isaac.sensor.Camera`` (+ depth
  annotator) this example pulls intrinsics / pose / depth from.
* `PR #62 <https://github.com/strands-labs/robots-sim/pull/62>`_ --
  ``render`` returns the RGB + metric-depth frames the compositor
  needs.

Until those merge, ``render`` returns blank frames on a stock build.
"""

from __future__ import annotations

# Eager re-exports are safe here: none of these modules import omni (or any
# other Isaac dependency) at module level -- heavy imports are lazy inside the
# functions that need them, so `import examples.isaac_gs` stays cheap.
from examples.isaac_gs.camera_utils import (
    IsaacCameraParams,
    get_camera_params,
    render_rgb_and_depth,
)
from examples.isaac_gs.compositor import IsaacHybridCompositor
from examples.isaac_gs.scene import build_default_scene

__all__ = [
    "IsaacCameraParams",
    "get_camera_params",
    "render_rgb_and_depth",
    "IsaacHybridCompositor",
    "build_default_scene",
]
