### Fixed

A `_read_state` probe that fails is now reported instead of swallowed, once per
category. `Mesh._read_state` probes a robot driver defensively so that a flaky
read cannot kill the state thread, and all four probes ended in a bare
`except Exception: pass` -- so a probe that raised left no trace at any log level.

The consequence was not a missing log line. `_read_state` returns `None` when
nothing but `peer_id` and `t` survived and the state loop publishes nothing for a
`None`, so a peer whose joint probe raised stopped publishing state entirely
while its presence broadcast still advertised it as `connected`: on the wire and
in the log it was indistinguishable from a healthy peer with no joints to report.
Measured against the real method, a joint probe raising `RuntimeError` or
`ConnectionError` produced a `None` snapshot and zero log records, and a failing
task probe dropped the `task` section out of an otherwise healthy snapshot in
silence.

The state loop's own report could not cover it. `_read_state`'s top-level
statements are `Assign, AnnAssign, Try, Try, Try, Return`, so every statement
that can raise is already inside a probe `try` and `_state_loop`'s
`logger.debug("state tick error")` is unreachable for a probe failure -- it only
ever sees a failure of `publish` itself.

Each probe keeps `except Exception`: a flaky read must not kill the state thread,
so this is a recovery path and the breadth is correct. The defect was the
silence. The first failure of each category (`hw_joints`, `task_state`,
`sim_world`, `sim_joints`) is now logged at WARNING naming the category, the peer
and the exception's `repr` -- the type is in the line because it selects the
operator's next move, a contended serial port being a different job from an
uncalibrated arm. Later failures of that category drop to DEBUG, because the loop
retries at `STATE_HZ` (10.0) and would otherwise emit ten warnings a second.

What is published is unchanged; this changes the report, not the snapshot.
