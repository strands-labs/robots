### Added: a mesh device runs a named skill on itself, so no control loop crosses the network

`Mesh.mount(robot, skills={name: policy})` gives a device its own catalogue, and the new `skill` and `skill_status`
rpcs start a skill on the device by name and report on it. Before, `use_mesh` ran the policy in the caller and drove
the robot with one observe and one act rpc per control step: 5.7 steps per second over AWS IoT Core for a 50 Hz walk.
The skill is an agent's run on the device (its safety chain on every step, a real robot's own `approve=`); `stop` and
`estop` end it; while it runs the caller's watchdog pauses its quiet clock, and a caller whose mesh vanishes ends it
and the device applies `on_disconnect`. `use_mesh` lists each device's skills, starts one through `run`, reads it
with a new `status` action, and refuses to drive a real robot on another machine that offers no skills.
