### Fixed: a Feetech calibration whose counts do not fit the servo is refused at connect

`FeetechDriver` now checks the calibration it loads, from a LeRobot file or from `calibration=`: every joint's
`range_min` and `range_max` must lie in the encoder's 0..4095 counts, `homing_offset` in -2047..2047 (what the servo's
register holds and what `calibrate()` writes) and `drive_mode` must be 0 or 1. Anything else raises `ValueError` naming
the joint, the field and the file. Before, a span such as 0..65535 mapped a target inside the joint's limits to a goal
many turns past its end stop, and a count outside 16 bits failed mid-run with torque on. The check is
`Calibration.check_bus(resolution=, max_homing=, source=)`.
