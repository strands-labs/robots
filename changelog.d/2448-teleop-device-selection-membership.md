### Fixed

- **teleop**: read a teleop device selector by membership instead of by truthiness, so an
  empty selection is refused rather than widened to every attached device.
  `teleoperate(names=[])` - what a filter that matched nothing produces - connected and
  drove every attached leader and reported success, and `detach_teleop("")` detached the
  whole attached set, which also ended a running session because `detach_teleop` stops the
  loop once nothing is left to drive. `names` now goes through the shared name-list domain
  as well, so a single name passed as a bare string, a repeated name (polled twice per
  tick) and a one-shot iterator (consumed before the loop could poll it) are refused before
  any device is connected. `names=None`, a named subset and `detach_teleop(None)` are
  unchanged.
