### Changed: a Feetech arm reads servo position only on each tick (2 bytes per servo, as LeRobot does); `read_motion=True` reads velocity and load too

`FeetechDriver.read()` asked every servo for position, velocity and load (6 bytes each) on every read, though the run
loop, the safety chain, recording and the LeRobot policies use positions only: 24 extra reply bytes per tick on an
SO-101, about 0.24 ms at 1 Mbaud. It now reads `Present_Position` alone and reports `velocities` and `efforts` as NaN,
the value a body gives for what it did not measure (the mesh sends it as `null`). `Robot("so101", port=...,
read_motion=True)` restores the 6-byte read; `extras()["read_motion"]` says which one a robot uses. An agent or voice tool's `observe` result sends an unmeasured velocity as `null`, so it stays
strict JSON. Every `state` subscriber (mesh stream, IoT
`state` shadow, Basic Ingest telemetry) gets an unmeasured velocity or effort as `None`.
