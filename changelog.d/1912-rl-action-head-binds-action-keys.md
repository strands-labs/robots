### Fixed: the RL action head is sized and named from the robot's action keys

`SimEnv` sized `num_actions` from `robot_joint_names`, but `SimEnv.step` sends a
numeric **vector** and `SimEngine.send_action` binds a vector positionally to
`robot_action_keys`. Both docstrings already drew that line -
`robot_joint_names` names the `observation.state` vector and says
"Action-vector binding ... uses `robot_action_keys` instead", and
`robot_action_keys` warns "a caller must not assume it has the same width" -
but the env read the joint list.

The two lists coincide only when a robot's actuator set matches its joint set,
and two shipped shapes make them disagree: a tendon-driven gripper (one
actuator over two mimic finger joints, so the builtin MuJoCo `panda` has nine
joints and eight action keys) and a Newton floating base (a 6-DoF free joint
that is a joint with no commandable scalar). On both, the head was one output
too wide, so every `send_action` was refused with a structured width error -
and `step` does not read that result. Measured on `panda`, 60 `env.step` calls
wrote no target, left `joint1` at exactly `0.000000`, and still banked 60.0
reward; a rollout of any length could be collected for a robot that never
moved. Sized from the action keys the same 60 steps drive `joint1` to 0.900.
`action_dim` is unchanged as the explicit override, and a robot whose actuators
match its joints is unaffected.

`PpoTrainer` and `FastSacTrainer` wrote the same list into `policy_meta.json`
as `joint_names`, beside `num_actions`. That field names what those outputs
drive, which a joint list cannot do once the widths differ, so it is now
`action_keys` and `len(action_keys) == num_actions` holds. `GymSimEnv`'s
`action_space` and `VecSimEnv`'s width both derive from `SimEnv.num_actions`
and inherit the fix.

`robot_joint_names` itself is unchanged: it remains the joint list and still
names policy state keys. Whether it should narrow on the Newton backend is a
separate question about observation naming, tracked in #1912.
