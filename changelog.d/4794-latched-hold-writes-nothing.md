### Fixed: while the e-stop is latched, a faulted run, a watchdog lapse and hold() write nothing

`apply_on_disconnect`, which a watchdog lapse and, since #4793, a faulted run call, sent a body whose controls all
hold (every registry robot) to `body.hold()` without checking the e-stop latch, and so did `Robot.hold()`. A
Feetech hold switches torque on, so a run whose policy faulted after the operator pressed e-stop energised a limp
SO-101. `hold()` now raises `EStopped` while latched, as `act()` does, and `apply_on_disconnect` writes nothing
while latched, since `estop()` already asked the body for its own stop. A run that faults while latched now returns
its fault instead of raising `EStopped` when a control declares `zero`, `safe` or `apply`.
