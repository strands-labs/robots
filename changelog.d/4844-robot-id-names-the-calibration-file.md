### Fixed: `Robot(id=)` names the LeRobot calibration file; an uncalibrated hardware arm warns

`Robot("so101", port=…, id="blue_follower")` now reads the calibration LeRobot wrote for that id
(`so_follower/blue_follower.json`): the Robot's `id` reaches a driver that takes `robot_id` (LeRobot's `--robot.id`)
unless the caller named one. Before, the id only named the twin and the mesh; the Feetech driver looked for
`so101_follower.json`, found other files, and ran uncalibrated with nothing said (joints read as a full turn, LeRobot
degrees off by the calibration). A hardware Feetech body that finds no calibration now warns, naming where it looked
and the three ways to give it one (`calibration=`, `robot_id=` / `Robot(id=)`, `calibrate()`).
