"""Newton FEM-cloth + frictional grasp library (issue #5).

A reusable cloth representation + grasp stack built on Newton physics
(MuJoCo-Warp / Warp, NVIDIA's announced future physics engine for Isaac Lab),
verified on the DGX Spark (aarch64 / GB10) from stock pip wheels.

Why it exists: PhysX particle cloth cannot hold a frictional grasp (rigid grippers
penetrate + slip — NVIDIA forum 332704; cloth↔articulation attachment unsupported —
IsaacLab #4291), which forced issue #2's spring peel-off grip workaround. Newton's
VBD cloth holds a *pure friction* pinch through a sustained drag — no attachment,
no tabs, no springs — so the grasp itself is the physics, exactly as on the real
robot.

Modules:
  bedsheet  — the bed-cover cloth model (FEM triangle grid, bed-recipe parameters)
  gripper   — force-limited pinch gripper (bounded applied wrenches; honest forces)
  sensing   — short-range proximity sensing + grasp decision (bundled copy of the
              issue-#2 sensor layer; representation-agnostic)
"""

from .bedsheet import BedsheetParams, add_bedsheet  # noqa: F401
from .gripper import ForceLimitedPinch, PinchParams  # noqa: F401
from .sensing import GraspDecision, HandClothSensor  # noqa: F401
