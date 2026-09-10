### Fixed: a refused `control_frequency` or `action_horizon` no longer logs a cleanup error

`Robot("so100", mode="real", control_frequency=0)` raised the documented
`ValueError`, and then the half-built instance was finalised: `__del__` ran
`cleanup()` unconditionally, `cleanup()` read the shutdown latch the validators
had raised before, and the operator saw
`ERROR Cleanup error for so100: 'Robot' object has no attribute '_shutdown_event'`
beside the refusal they had actually caused. `__del__` now returns before
`cleanup()` when `__init__` never created the latch, because such an instance
holds nothing to release; a bring-up that fails after the latch exists still
runs the full teardown.
