### Fixed

- **Local teleop now holds a leader frame to a per-joint slew bound, using a separate local default
  wide enough for shipped hardware.**
  `teleoperate(publish=True)` drives a local follower and, from the same `get_action()` stream, every
  remote one, but only the mesh path bounded how fast a single joint could be commanded to travel.
  The merged frame is now checked against the same helper the mesh path calls, but with its own default
  bound: `STRANDS_TELEOP_SLEW_ABS` defaults to 500 units/s (the mesh receive path keeps its own, looser bound),
  so the shipped SO hardware defaults (joints in degrees, gripper in 0-100 range) work without env-var
  tuning -- a calm 90 deg/s sweep, a half-second gripper close (200 units/s), and even the STS3215
  no-load max (~372 deg/s) all pass, while encoder glitches (>1000 units/s) are still caught. An
  over-speed frame is refused and counted in a new `slew_rejected` stat rather than clamped, since
  clamping would silently alter an actuator command. Refusals are counted apart from errors but still
  move the session off `success`, so a device whose units the bound does not expect cannot report a
  clean run while moving nothing. The baseline a joint is measured against is kept for as long
  as it can still refuse a frame at the bound in force, so a device that stops reporting for a while
  -- a disconnect, a USB re-enumerate -- has its first read back on reconnecting measured against where
  it left the follower rather than applied unchecked.
