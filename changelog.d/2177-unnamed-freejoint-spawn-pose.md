### Fixed: a floating base whose `<freejoint/>` is unnamed keeps its spawn pose when a robot is added

`add_robot` composes a robot into the live scene with `spec.recompile`, whose
state transfer is positional: the new buffers receive the old values for the
indices both models share, and the entries past the old size are whatever the
fresh allocation happened to contain. `_recompile_preserving_state` defined that
tail for `ctrl` and `act` but documented `qpos` as compiler-defined, which it is
not -- so a joint the recompile grew took its pose from leftover memory.

Every other pass over a new robot's state is keyed by joint NAME: the
robot-scoped reset walks `robot.joint_ids`, and `keyframe=` is applied by short
joint name. An UNNAMED `<freejoint/>` -- the standard MJCF idiom for a floating
base, and what the Unitree Go2 and LeKiwi ship -- appears in neither, so nothing
downstream could repair the tail either.

Measured on `go2` spawned with `keyframe="home"`: with one robot already in the
world the tail happened to hold `qpos0` and the base stood at `z=0.445`; with two
it held zeros, including an all-zero quaternion, and the base started on the
floor. The leg hinges were applied by name in both cases, so the only symptom of
a dropped base was a quadruped lying down under a reported
`{"status": "success"}` -- and the free joint is the one part of a keyframe no
controller can restore, because a joint-space hold at the same angles cannot lift
a base that never stood up.

`_recompile_preserving_state` now defines every buffer it grows -- `qpos` from
`qpos0`, and `qvel` alongside the existing `ctrl`/`act` -- before the forward
pass at the end of the recompile reads them. The write covers only the entries
past the old size, so a settled object and a parked arm keep the state the
positional transfer carried over.
