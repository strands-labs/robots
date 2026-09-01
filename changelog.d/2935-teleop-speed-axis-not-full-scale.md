### Changed: the teleop slew bound's margin is stated against recorded teleop, and reach's per-unit table is not reused on the speed axis

`input_value_abs_by_key` bounds how far one teleop command may *reach* in the
unit the receiving follower declares for each joint, because full scale is where
reach stops being addressable - lerobot clamps a command past it. The open
question was whether the *speed* bound, `DEFAULT_INPUT_SLEW_ABS`, should be
keyed on the same declaration. Measured, it must not be.

Full scale is not a speed. A frame-unit speed converts to servo travel through
the joint's *calibrated* range, which on a `RANGE_0_100` gripper is unrelated to
its full scale of 100. Granting speed the same multiple of full scale that reach
gets would bound a percent gripper at 400 units/s - below the ~406 units/s that
recorded leader-follower teleop actually commands - so the tighter number would
be bought by refusing real motion on the one joint a hand traverses full scale
fastest, while leaving the degree joints untouched.

The bound's own justification was also reasoning from the wrong ceiling. A
leader arm is back-driven by hand, so the Feetech STS3215 no-load speed
(~372 deg/s in the unit the frames carry) is a floor on what a real stream
contains, not a limit on it: across 176,293 recorded action frames the fastest
single commanded step is ~899 deg/s, 2.4x that figure. The default is unchanged
and still clears every one of those frames, but by 1.6x rather than the ~3.9x
the datasheet reading suggests, which is what a future retune has to spend.

No production behaviour changes. The measured margins, and the counterfactual
per-unit row that would refuse a recorded gripper step, are pinned by tests -
including at the receiver, where such a row would take effect.
