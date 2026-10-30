### Fixed: an agent or dashboard can hold a body where it is, even a joint resting past its limit

A joint can come to rest a hair past its hard range: MuJoCo joint limits are soft, and on a Feetech arm
`counts_to_rad` has no clamp, so a servo past its calibrated sweep reads outside `[lo, hi]`. The operator
gate (`confirm_act`, and every agent-tool and dashboard write through it) graded the whole target against
`range` with a 1e-9 tolerance, so after any legal move that saturated a joint, holding the body where it is
(`arm.confirm_act(arm.observe().joints.positions)`) was refused, naming a joint the caller was only echoing.
The limit check now grades each joint against its range widened to include the body's measured position:
holding there, or moving back toward the range, is allowed, while a target that drives a joint further out
than it already sits is still refused. The body is read once, only when the plain check would refuse.
