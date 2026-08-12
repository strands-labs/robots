### Quality: pin the not-reached half of the motion-primitive result envelopes

`MotionPrimitivesCore` owns the `move_to` / `set_gripper` / `rotate_wrist` result
envelopes so every backend answers identically, and its own test module says so -
but it drove none of the three. The reached half was reached through the MuJoCo
mixin; the not-reached half of both converging primitives was reached nowhere,
and that is the half an agent acts on: it decides whether to retry with a larger
step budget, reading `reached`, `steps`, `position_error_m` and `ik_residual_m`
from the json block rather than the sentence.

Pinned in the module that owns them, plus the reachability and the remedy on a
live world - a solvable pose the servo cannot reach inside a two-step budget is
refused with its residuals, and the identical call with a real budget converges.
Also pins the default `_get_registry_robot` seam, which every shipped caller
overrides, so the body a new backend adapter inherits is exercised. Tests only;
no behaviour change.
