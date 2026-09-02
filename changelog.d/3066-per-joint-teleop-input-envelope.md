### Fixed: a teleop frame's joints are bounded in the unit each one declares, not by one shared scalar

`validate_input_frame` bounded every key in a teleop frame by a single scalar,
`DEFAULT_INPUT_VALUE_ABS = 720.0`. One frame does not carry one unit: lerobot
normalises Feetech positions per motor, so a shipped SO-100 class arm declares
`DEGREES` for its five arm joints and `RANGE_0_100` for its gripper. A scalar
therefore has to be loose enough for the widest joint, which leaves it slack on
the narrowest - the gripper's fully-open command is `100` and it was bounded at
`720`, seven times its own full scale. `validate_input_frame` already took a
`value_abs_by_key` seam for exactly this and nothing populated it.

This was slackness, not a false refusal: `720` is already degree-scaled, so real
frames were accepted before and still are. Measured on that arm, the five degree
joints stay at `720` and only the gripper tightens, to `200`.

The unit is read from the robot the frame is about to be applied to, never from
the frame - the bound that constrains a sender must not be chosen by that sender.
New `bus_access.motor_norm_modes` reads `bus.motors[name].norm_mode` through the
same wrapper-or-driver resolution as `joint_read_source`, and `InputReceiver`
resolves the envelope once from its own follower on the first frame, so a frame
carrying its own `norm_modes` or `unit` changes nothing.

`INPUT_ENVELOPE_FULL_SCALES` is derived from the existing default rather than
restated (`DEFAULT_INPUT_VALUE_ABS / 360.0`), so the degree row *is* the old
default expressed through the new rule and a retune of the default moves every
unit together. Two properties hold by construction: an unrecognised `norm_mode`
is absent from the mapping and so keeps the scalar envelope instead of being
widened to the most permissive row, and a per-joint row only ever tightens
(`min(scalar, per_key)`), so an operator who narrows
`STRANDS_MESH_INPUT_VALUE_ABS` is not handed the envelope back by a declared
unit. Both frame spellings of one motor are bounded (`gripper` and the
`<motor>.pos` action key lerobot publishes).

`mesh/security.py` still imports nothing from lerobot - the mode spellings are
held as plain strings, and the guard is an import-graph assertion rather than a
check on the strings, so the module stays importable and testable with lerobot
absent.

`input_frame_slew_violation`'s `max_slew_by_key` seam is deliberately left
unpopulated: it bounds a speed, and the same full-scale derivation would put a
percent gripper's ceiling below what the leader's own servos produce across that
joint's travel, which would be a false-refusal regression rather than a slack
removal.
