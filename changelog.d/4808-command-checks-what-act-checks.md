### Fixed: `Robot.command` refuses a NaN or wrong-length target before the body and clips into the limits, as `act` does

`Robot.command(targets)`, the write without a read that #4797 added, sent its vector as given. On a Feetech arm a
NaN or infinite target, or a vector of the wrong length, turned torque on before the driver refused it, and on an
uncalibrated arm a target past a joint's limit went out unclipped; in a world the NaN went into MuJoCo's controls,
and while one control is NaN MuJoCo drives every actuator in that world as if its control were 0. `command()` now
checks as `act()` does before anything reaches the body: `NonFiniteTarget` for NaN or inf, a `ValueError` for a
vector that is not one value per action key, and every joint clipped into its limits. It still never reads the
body. `coerce(..., measured=)` likewise refuses a `measured` vector of the wrong length, where it raised
`IndexError` or returned extra targets.
