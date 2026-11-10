### Fixed: a Feetech calibration whose span does not fit the joint's limits is refused, and `extras()` names it

`FeetechDriver` maps each joint's limits onto its calibration's `range_min..range_max`. A span inside the servo's
counts but much wider than the joint's limits, such as a full turn for a ±100° joint or another joint's span, put a
goal inside the limits past the end stop and scaled the speed cap with it. Now a span covering more than 1.5× the
counts the limits take on the servo raises `ValueError` naming the joint, the span and the file, at connect and in
`calibrate()` before it saves (the servos get their homing offsets back). A real SO-101 calibration sits at
1.0–1.25×; a narrower span still loads. `extras()["calibration_sha256"]` is the sha256 of the calibration's values,
the same for a file and `calibration=`, so a changed calibration shows up in a recording. The check is
`Calibration.check_limits(limits, resolution=, source=, max_scale=1.5)`; the hash is `Calibration.digest()`.
