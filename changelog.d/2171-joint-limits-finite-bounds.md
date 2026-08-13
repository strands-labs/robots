### Fixed: a non-finite `joint_limits` bound refuses the ROS 2 / RTPS bridge instead of dropping every inbound command

`RosTelemetryBase._validate_joint_limits` exists so a malformed bound surfaces at
bridge construction rather than as a silent mid-run rejection, but its only
ordering check was `low > high` -- and every comparison against `nan` is `False`,
so `(-1.9, nan)` passed. The `low <= pos <= high` test in `_command_action` was
then `False` for every position, so a bridge that reported a clean construction
dropped **every** inbound `joint_command` for that joint. `(inf, inf)` and
`(-inf, -inf)` passed the ordering check the same way and admitted nothing
either, and an `int` past the float64 range escaped as a bare `OverflowError`
rather than the documented `ValueError`.

Each bound now goes through the shared `strands_robots.utils.finite_number_error`
domain before the ordering comparison, so a non-finite bound is refused at
construction with the same wording as its other callers. Both hardware bridges
inherit the validator, so one rule covers the rclpy and pure-RTPS transports.
Note this narrows the accepted domain: a half-infinite pair such as `(-1.9, inf)`
did admit in-range commands and is now refused -- a `{motor: (min, max)}` clamp
range is a bounded interval, and an unbounded joint is expressed by omitting it.
