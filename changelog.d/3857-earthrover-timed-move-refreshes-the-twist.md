### Fixed: a timed EarthRover `move` no longer stops early under the SDK watchdog

`EarthRoverDriver.move(duration_s=N)` sent one twist and slept `N` seconds before the trailing stop. The earth-rovers-sdk stops the rover on its own once no fresh command has been confirmed for `CONTROL_WATCHDOG_S` (3 s by default), so a 10 s move drove for about 3 s while the answer still reported `held_s=10, stopped=True`. The hold now re-sends the twist every 0.1 s (`MOVE_REFRESH_PERIOD_S`) until the deadline, then sends the zero twist, so the rover drives for the time you asked and the report is true.
