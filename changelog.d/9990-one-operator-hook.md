### Changed: one operator hook, `Robot(operator=)`, replaces `confirm=` and `approve=`

A real robot now takes one callable, `operator=<callable(Move | RunPlan) -> bool>`. A single command from an agent
tool, the dashboard, a mesh peer or a ROS topic asks it with a `Move`: the robot, the targets by joint name in the
robot's units, the written vector (read-only), who sent it (`source`: `agent`, `dashboard`, `mesh`, `ros`), the ramp
time and, for a target outside `safe`, the reason; `str(move)` is one line. An agent's run asks it once with its
`RunPlan`, as `approve=` did. `Move` and `Operator` are exported from `strands_robots.robot.run` (next to `RunPlan`)
and `strands_robots.robot.safety`. `use_mesh(mesh, operator=)` and `ros.connect(..., operator=)` follow.

What stays the same: without `operator=` a real robot refuses agent moves and agent runs; a run's yes binds the
approved files and Hub commit; chat and voice never answer; the e-stop asks nobody and `resume` still needs
`approved_by`; `operator=` is hardware-only. The dashboard's click (`X-Confirm`) is still the operator's yes over HTTP.

Migration: `Robot(..., confirm=c, approve=a)` becomes
`Robot(..., operator=lambda r: c(r.vector) if isinstance(r, Move) else a(r))`. The old keywords raise a
`TypeError` that says `use operator=`. `use_ros(approve=)` (a ROS graph write) is unchanged.
