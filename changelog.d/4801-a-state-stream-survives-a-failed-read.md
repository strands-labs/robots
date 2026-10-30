### Fixed: a state stream survives a failed read, reports it once and resumes at its rate

`Emitter._state_stream` had no `try` around a sample, so the first exception (a corrupted reply, a
servo that did not answer, an I/O error, a subscriber that raised) ended the robot's `state` thread,
with no `fault` event and no sample after the bus recovered; the mesh publishes this stream. The
stream now counts each failure in `Emitter.errors`, reports each run of failures as one `fault`
event (an error that cannot print itself is named by its type) and keeps sampling; `seq` still
advances, so a failed tick is a gap. After a failed read the next one is a full period away, so the
stream does not burst out the ticks it missed once the bus answers again. A stream whose reads keep
failing still takes the bus a period after each failure, so `estop()` on that robot can wait behind
a failed read, and a mesh's tenant-wide e-stop, which latches its robots one after another, can
latch the robots mounted after that one later.
