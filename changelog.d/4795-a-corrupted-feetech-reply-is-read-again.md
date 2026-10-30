### Fixed: one corrupted Feetech status packet no longer ends a run or an e-stop's hold

`FeetechDriver.read()` made a single `sync_read`, so one status packet with a bad checksum raised
`ProtocolError` out of whatever was reading: a run, including the read with which a faulted run puts
the body in its declared `on_disconnect` state, a watchdog lapse, and `estop()` after its latch, which
then wrote no hold and sent no `estop` event. `read()` now asks again up to `num_read_retries = 2`
more times, LeRobot's SO follower default, then raises. A servo that does not answer now costs three
read timeouts before `read()` raises, not one.
