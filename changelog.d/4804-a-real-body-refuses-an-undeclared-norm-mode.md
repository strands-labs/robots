### Fixed: a real body refuses a LeRobot checkpoint that does not say its norm mode

A LeRobot checkpoint stores no norm mode, so `Policy("<checkpoint>")` took the registry's label for the robot,
`so101-range-gripper-percent-v1` for an SO-101, while LeRobot has defaulted SO arms to degrees since 0.4.4, the
pinned commit included. A degrees checkpoint on a real arm was read in the wrong units: commanded to +90 degrees,
shoulder_lift landed 20.4 degrees off and the full-turn wrist_roll 72 degrees off (on the repo's fake Feetech bus,
read back with LeRobot's own conversion). On a hardware `Robot`, a LeRobot checkpoint given neither `norm=` nor
`semantics=`, or an empty one, is now refused when `run` or `start` attaches it, before the checkpoint is fetched or
loaded and before anything reaches the bus, and the error names both options and both modes. A robot from
`Mesh.connect()` counts as hardware even when the device is simulated, and a spec string alone (`from_spec`) cannot
declare a mode. A simulated robot keeps the registry's default.
