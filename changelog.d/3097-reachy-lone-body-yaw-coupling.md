### Fixed: a Reachy Mini body-only turn is held to the head-body coupling limit

`HEAD_BODY_YAW_DELTA_LIMIT_DEG` bounds `head_yaw - body_yaw`, and
`envelope_error` only reached it when one action carried both names. Every motion
verb that sends one member routed around it: `reachy_body_turn` sends `body_yaw`
alone, `reachy_look` omits `body_yaw` at its `None` default, and an action naming
a head axis other than the yaw (`{"head_pitch": 10, "body_yaw": 160}`) commands
the head yaw to zero without spelling it, so the gate did not see that pair
either. `reachy_body_turn(yaw=160)` against a head at 0 reported `success` for a
body that reaches 65 degrees and stops.

The limit is the daemon's own. Its default kinematics solves a head pose through
`inverse_kinematics_safe(pose, body_yaw, max_relative_yaw=65 deg,
max_body_yaw=160 deg)` - the two figures this envelope carries - and it never
refuses an over-twist: it keeps the twist inside the limit by moving the body,
holding the head pose as the primary task. So an out-of-limit request does not
fail, it succeeds with a body yaw the caller did not ask for, which is the silent
substitution the envelope exists to refuse.

That makes the two directions different events, and only one of them is a defect.
A lone `head_yaw` of 180 is honored - the body turns to 115 under it, nothing of
the caller's is substituted, and refusing it would refuse the head verb its own
range. A lone `body_yaw` of 160 against a head target of 0 is replaced by one 95
degrees short. `ReachyDriver.send_action` now checks the coupling on a body-only
turn as well as on a pair, against the head yaw it last commanded - which is
exactly what the daemon is still targeting, and known rather than estimated
because the head command is a whole pose every time. The bound follows that
target rather than being a fixed range, so a body turn to 160 with the head
already round at 100 is 60 degrees of twist and still legal.

Unknown is not guessed. Before any head pose is commanded, and after
`play_move`, `wake_up`, `goto_sleep` or `set_motors` - the daemon re-pins its own
head target to wherever the head physically is when torque returns - the target
is forgotten and the coupling is skipped, because a turn refused against a stale
target is a turn the robot could have made.

The descriptions a model reads back out of its own schema follow the same split.
`reachy_look` can omit `body_yaw`, so it states that the pairwise limit applies
only when both are sent, and now says why a lone large `yaw` is not an unchecked
twist: the body turns under the head to serve it. `reachy_body_turn` sends
`body_yaw` alone, where the limit *is* applied, so it names the counterpart it is
measured against instead of claiming an exemption. The suite grades those two
obligations separately, derived per verb from what one call actually carries, so
a stale exemption fails as loudly as an unqualified promise. The Device Connect
`_reject_unusable` keeps its scope paragraph but no longer says the coverage is
absent: the native driver has it, and this surface's `body` RPC stays per-axis
because keeping no record of the pose its `look` RPC commanded is the invariant
that puts the limit out of reach there.
