### Added: Robot(**driver_options) reaches every driver parameter, and a misspelt one is refused

`Robot(name, port=, **driver_options)` now hands any extra keyword to the driver unchanged: the feetech
bus takes `robot_id=` and `baud=`, the unitree and microduck links take `timeout=`. `make_driver` forwards
them and checks each against `driver_parameters` for that driver, so a misspelt option (`baudrate=` for
`baud=`) raises a `TypeError` naming what the driver takes instead of being silently dropped, and a new
inner parameter with no public path fails the pass-through test. A robot in a world has no bus and so takes
no driver options. Hardware camera specs are `{"index", "width", "height", "fps"}` (LeRobot's
`OpenCVCameraConfig` field names), checked at the first frame: a camera that delivers another size than
asked is refused, since every checkpoint is trained at one size.
