### Fixed: a Feetech arm's calibration is the file `Robot(id=)` names, recorded as LeRobot records it; inf and nan rates are refused; GLFW renders on macOS

`Robot("so101", port=..., id="my_arm")` passes its id to the Feetech driver as LeRobot's `--robot.id`, so it
loads `so_follower/my_arm.json`, and an id never loads another arm's file: with no `<id>.json` beside other
files, it loads none and a warning names them, so joints map raw full-turn counts. `calibrate()` saves over the
file it loaded (else `<lerobot_type>/<id>.json`), so the next connect reads the homing offsets the servos now
hold. Without an id the only file in a directory still loads, but it may be another arm's, so with `save=True`
(the default) `calibrate()` refuses to record over it or beside it and asks for `id=` or `calibration=`. With a
calibration loaded, `calibrate(interactive=True)` asks first, as LeRobot does; Enter keeps it and, unlike
LeRobot, writes nothing to the servos; `c` re-records, and the offsets are zeroed only after the middle pose is
confirmed. As in LeRobot, `wrist_roll` is not swept but given the full turn (0..4095), and a sweep that leaves
another joint without a range is refused before anything is saved; the servos then get back the homing offsets
they held, so the calibration loaded still maps the arm. `run(hz=)` and `subscribe("state", hz=)` refuse `inf`
and `nan` (both come out of `json.loads`), at which the bus was read and written with no pause.
`FeetechDriver.write()` refuses a NaN goal before torque goes on. With `MUJOCO_GL=glfw` on macOS a render runs
on the thread that asks, where #4789's one GL thread ended the process with SIGTRAP.
