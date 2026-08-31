### Tests: the Microduck driver's degraded surface is graded, not just its happy path

`tests/drivers/microduck/test_microduck_driver_over_socket.py` drives the driver
against a real robotd socket and grades the path where everything works. The
other half -- every path where the driver refuses, degrades or gives up -- was
graded by nothing: 60 of `drivers/microduck.py`'s 354 statements uncovered, and
all of them on the failure side.

That is the wrong half to leave ungraded for a native hardware driver, because a
driver whose failure surface is unchecked is one that can report success about a
robot that did not move. The uncovered set included the two delegate-only
refusals that are the module's entire answer to "why does `mode="real"` not run a
policy"; the whole connect ladder below the version check (nothing listening, a
refused handshake, a hang-up mid-handshake, a declined `robot.subscribe`, a
best-effort `robot.health` that must not block the connect); the reader thread's
tolerance of a blank or unparseable line; the id-correlation table's timeout
accounting; and the `-> None` shutdown hook whose only observable is whether it
claims a halt robotd declined.

31 cells over that surface. Every socket cell drives a genuine `AF_UNIX` server
-- `MockRobotd` for the faithful protocol, a small scripted server for the
misbehaviours a faithful server by definition does not exhibit -- so the NDJSON
framing and the id correlation are exercised rather than mocked away. Measured
by mutation: 13 changes to the behaviours these cells grade, 12 detected, the
control clean and no sibling suite disturbed by any of them.

One docstring is corrected in the same change. `map_hardware_joints` justifies
its positional fallback as letting a robotd whose vector "grew or shrank" degrade
"to a partial read", and the two directions are not symmetric: a shorter vector
does yield a partial read, while a longer one yields a full 14-joint read in
which index 9 is no longer dropped, so every joint after the mouth is named one
position early. The behaviour is deliberately left alone -- which of those a
14-wide vector is, robotd having dropped the mouth itself or having dropped some
other joint, is not knowable from the width -- so the docstring now says what
each direction does instead of describing both as partial, and the tests pin the
15-wide contract and the short-vector partial read while claiming nothing about
the grown case. The 13th mutation is that behaviour change, and it fires nothing.

`drivers/microduck.py`: 60 -> 13 missing, 83% -> 96%. The statement count is
unchanged at 354, so the docstring correction added no production statements.
