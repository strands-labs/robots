### Fixed: an inbound `joint_command` position must be finite, not merely readable

`RosTelemetryBase._command_action` is the one inherited parser both hardware
bridges and the simulation bridge deliver `/<robot>/joint_command` into.  It
refused a position `float()` could not read and deliberately let a `nan`
through, on this stated ground: "a nan position is a readable number that
`send_action` refuses naming the joint".

That is a claim about `send_action`, and the parser is shared by subclasses whose
`send_action` is a different function.  It held for `SimRosBridge`, whose host
does refuse a non-finite action value naming the key.  It did not hold for
`HardwareRosBridge` or `HardwareRtpsBridge`: those reach `bus_access.write_action`,
which takes the bus lock and delegates, and lerobot's follower writes
`Goal_Position` with no finiteness check.  lerobot bounds a normalized position
in `MotorsBus._unnormalize` with `min(100.0, max(-100.0, val))`, and `nan`
compares false against both bounds, so `max` keeps its first argument and the
clamp resolves the joint to an end stop instead of refusing it.

Measured through the real `FeetechMotorsBus._unnormalize` on an SO-101's own
default norm modes, with a shoulder calibrated `800..3200`: a position of `10.0`
reached the wire as `2120`, a `nan` as `800` (the joint's `range_min`) and an
`inf` as `3200` (its `range_max`) - each of them with zero warnings logged.  So
a `joint_command` carrying a `nan` drove a real arm to an end stop under a
`success` envelope: `_drive_from_command` warns only when `send_action` raises or
answers `status="error"`, and it did neither.  The `joint_limits` branch below
does catch it, because `not (low <= nan <= high)` is true, but `joint_limits` is
optional and defaults to `None` on both bridges - so the default
`ros2_bridge=True` configuration had nothing between the wire and the clamp.

The parser now refuses a non-finite position WHOLE, which is the disposition its
two siblings in the same method already have: a name/position length mismatch and
an out-of-range value are both rejected entire rather than partially applied.  It
asks `finite_number_error`, the same domain the `joint_limits` bounds a few lines
above already use, so the accepted set cannot drift between the two.  The
refusal names the bridge, the joint and the value.

What is unchanged: every position that was accepted still is.  The domain is
asked about the *coerced* value, so `True` is judged as `1.0` rather than as a
`bool` the domain would reject; `0.0`, a negative position, an `int` and a
numeric string all still resolve, and an empty keep-alive sample is still dropped
silently.  The non-numeric branch keeps its own report, and a declared
`joint_limits` still bounds an in-range position and still refuses an
out-of-range one.
