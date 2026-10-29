### Fixed: a LeRobot checkpoint's units are checked against its own saved stats, not assumed

`Policy("<owner>/<repo>")` read every checkpoint in the robot's registry convention (`so101`: range mode) without
looking at the checkpoint. An SO-101 ACT trained in degrees (`robotfuel/act_so101_t16b`, saved `wrist_roll` min
-155.6) ran as range mode and drove `wrist_roll` toward its limit with nothing said. `server_config` (so a local
`Policy` and a served one alike) now reads the checkpoint's saved normalizer min/max and refuses, before a session,
when a joint's numbers are ones LeRobot's normalizer cannot emit in the assumed mode (range modes are
clamped to their range; degrees spans one motor turn, +-180), naming the joint and the
`norm=` its data does fit. Explicit `semantics=` / `norm=` are checked the same way; a checkpoint that saved no
min/max is not checked.
