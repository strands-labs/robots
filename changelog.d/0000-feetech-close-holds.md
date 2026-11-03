### Fixed: a Feetech `close()` holds the arm, and a degenerate calibration is refused

`close()` now writes `Goal_Position := Present_Position` under torque and leaves torque on, so
quitting a program never drops a gravity-loaded SO-10x arm; pass
`disable_torque_on_disconnect=True` for LeRobot's release-on-close. A `JointCalibration` whose
`range_min >= range_max` (an unswept joint) is refused at build instead of dividing by zero on
every read (K8).
