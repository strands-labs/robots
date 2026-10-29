### Fixed: under `units=`, a policy may return a `{joint: value}` mapping

`arm.run(lambda obs: {"shoulder_pan": 60.0}, units="lerobot")` now converts each named joint from the policy's units
to radians and holds the joints it leaves out at their measured position, the same rule as without `units=`. Before,
it died with a bare `TypeError: float() argument must be ... not 'dict'`. A `JointState` is accepted the same way; an
unknown or LeRobot `'<joint>.pos'` key is refused by name.
