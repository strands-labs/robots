### Added: `list_driver_coverage()` - which drivers can build each robot, joined from both registries

Two registries answer "what can build this robot for real": a registry entry's
`hardware.lerobot_type` names a lerobot robot type, and the native-driver
registry holds the classes driver packages register. `list_native_drivers()`
reports the second and `get_hardware_type()` the first, so the group defined by
*both* absences - the simulation-only robots, which is the one group a
driver-coverage gap list is made of - was reported by nothing and got assembled
by hand. A hand-assembled join goes stale in one direction only: a robot that
gains a driver keeps reading as a gap. Measured on the shipped registry, a
hand-written gap list of sixteen robots named five that were already reachable -
`open_duck_mini` through `FeetechDriver`, and `openarm`, `bi_openarm`,
`rebot_b601` and `bi_rebot_b601` through their declared lerobot types.

`list_driver_coverage()` is that join, derived on every call: canonical robot
name to the driver names able to build it, for every registered robot. Both
reported names are `DRIVER_CHOICES` values, so an entry reads as the set of
`driver=` arguments that work, and an empty tuple is the driver gap. It reports
what the registries *declare*, needing no optional dependency and no hardware -
whether lerobot is installed is a property of the environment, not of the robot.
Coverage is not resolution: a robot both drivers can build is reported as both,
and `resolve_driver` still picks one.

Resolution precedence, the native-driver refusal and every registry entry are
unchanged. `has_hardware`'s docstring described only the `lerobot_type` half of
the block it reads, which was already wrong for the Reachy Mini and the Microduck
- both declare only `hardware.driver`, because lerobot has no robot type for them
- so it now describes the block and points at the join for the complete answer.
