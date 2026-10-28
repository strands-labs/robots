### Added: a run starts from the policy's start pose; a Feetech write has a default joint speed cap

**Start pose.** A policy may name where its episodes begin: `start_pose` (in its own units; a mapping of joint — or
LeRobot `joint.pos` — to value, or a full vector) and `start_seconds` (default 3). `Policy("zenoh://…",
start_pose={"shoulder_pan": 20.0})` sets it; left out, a server whose descriptor carries `start_pose` is believed.
`Robot.run` / `Simulation.run` ramp there (smoothstep, `reach`) before step one; an agent's run passes every ramp
target through the same safety chain as its steps; `run.started` carries the pose. A chunked model trained from one
rest pose plans its first chunk from there instead of from wherever the arm was left (last night: a 52° first-row jump).

**Speed cap.** `FeetechDriver(max_speed=2.0)` (rad/s, `None` to opt out) caps every write to `max_speed × dt` from
the last goal — or, on the first write after construction or torque-off, from a read the write takes itself — so a far goal becomes a ramp and the
next write continues from where the goal got to; trained motion (well under 2 rad/s) is written as given.
`extras()["max_speed"]`, `extras()["slewed"]` report it. `strands_robots/robot/drivers/slew.py` holds the rule for
any driver. `Robot.act(far_target)` on hardware therefore no longer snaps: `act(target, seconds=)` reaches a pose.
