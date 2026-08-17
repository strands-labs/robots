### Fixed

`add_robot(keyframe=...)` now applies the actuator command a MuJoCo `<keyframe>`
pairs with its pose, and `reset()` restores both. Previously only `key_qpos` was
read, so a gravity-loaded arm stood at its home configuration with every actuator
commanded to the zero configuration and sagged off home on the first steps -
measured at 1.5008 rad on the built-in `panda` `home` keyframe, repeated
identically after every `reset()`. 28 of the 31 built-in robots that ship a
`<keyframe>` declare a non-zero `ctrl` in it. The keyed `ctrl`/`act` are applied
verbatim (a servo setpoint, a motor torque and a stateful actuator's activation
each reach their own actuator) and matched by name under the robot's namespace,
so another robot's setpoints are untouched. A keyframe that declares no `ctrl` is
unaffected.
