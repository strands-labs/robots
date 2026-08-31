### Fixed: `raw_accel` slot two carries a gravity direction, not the accelerometer reading

`build_observation(..., gravity_source="raw_accel")` wrote the `base_acc` reading
into slot two of the observation vector unchanged.  Pollen's
`get_raw_accelerometer` does not: it negates the reading, normalises it, and
rotates `base_quat` instead when the magnitude is too small to carry a
direction.  So both `gravity_source` branches produce a *unit* gravity direction
in the base frame - they are two estimators of one quantity rather than two
quantities in different units, which is what makes a `raw_accel` export
interchangeable with an alpha export on the same robot.

Measured on a settled duck (`scene.xml`, `|accel| = 9.8100`): the two reference
branches agree to 1e-6, while the shipped write differed from the reference by
`max|d| = 10.060244` with all three components sign-flipped - a vector 9.81x too
long pointing the opposite way, in a slot the export was trained to receive a
unit direction in.  In free fall (`|accel| = 0`, where the accelerometer carries
no direction at all) the reference falls back to the rotation and the shipped
write produced a zero block.  Both wrong values were finite and both were the
documented width, so nothing downstream refused either.

`raw_accel_gravity` now sits beside `projected_gravity` as the single owner of
that estimator, and the raw-accel path requires `base_quat` as well as
`base_acc`: the degenerate-reading fallback is the rotation, so requiring the
orientation up front refuses a short observation dict on the first tick rather
than at the moment the robot leaves the ground.  Both readings now match the
reference at `max|d| = 0.000000`.

The `projected_gravity` default - every currently shipped alpha policy - is
unchanged and still reproduces its pre-change vector byte-for-byte.  This module
still never rescales the assembled vector; the fused `EmpiricalNormalization`
baked into the graph remains the only thing that does, and the one normalisation
here is the estimator's own unit-vector step.
