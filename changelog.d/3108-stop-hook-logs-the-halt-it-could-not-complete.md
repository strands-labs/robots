### Fixed: a `stop()` that could not halt the robot says so

`HardwareDriver.stop` is annotated `-> None` on all twelve shipped drivers, so it
carries no verdict - which makes its log the only place a halt it could not
complete can be recorded. Three drivers delegated to a verb that decides one and
threw the answer away: `BoosterDriver.stop` and `EarthRoverDriver.stop` dropped
`stop_task()`'s envelope, and `CrazyflieDriver.stop` dropped `land()`'s.

For the two ground robots the refusal is an ordinary hardware failure -
`BoosterDriver.move` reports a `RuntimeError` from `MoveCommand` as "the T1
refused the twist", and `EarthRoverDriver.send_action` reports a `/control` POST
that did not land. Because `stop` returns `None` and wrote nothing, a fleet
teardown had no surface anywhere - envelope, flag or log - saying the robot had
not stopped: a velocity-commanded rover holds its last command until another one
arrives, so an unsent zero twist leaves it driving.

Each hook now reads the envelope and logs a non-success naming what may still be
moving. `strands_robots.drivers.halt_failure_detail` renders the reason, because
the shipped halt verbs answer in two shapes - a refusal text, and a per-half
outcome dict naming which half of the T1's two-part halt failed - and a failure
it cannot parse still reports as a failure rather than as a landed halt. The
`stop` contract now states the obligation it always had, and a regression test
derives the rule over the whole fleet: a `stop` that calls one of its own
envelope-returning verbs must read what that verb answered.
