### Fixed: a T1 verb refuses a vendor enum member the installed SDK lacks

`booster_robotics_sdk_python` is a vendor wheel pinned to the robot's firmware,
not a dependency this project resolves, so the *installed* build's vocabulary is
an input. `BoosterDriver` freezes two claims about it - `ROBOT_MODES` and the
keys of `CMD_TYPE_STATE_FIELD` - and both were handed to an SDK enum with a
bare `getattr`. A build that spells one differently therefore raised
`AttributeError` out of `change_mode()` and `send_action()`, past the
`{"status": "error", ...}` envelope every driver verb is contracted to return:
an agent saw a traceback instead of the one reason nothing else could give it.

Both lookups now go through `resolve_vendor_member`, which returns the member or
a refusal naming the enum, the member asked for, and the vocabulary the
installed build *does* declare - so the operator can pick a mode from the
refusal alone, and a `send_action` whose wire convention could not be resolved
publishes nothing rather than a frame built on a guess.

The vendor-truth cell that grades `ROBOT_MODES` against a real wheel checked two
of the five names, which is why the drift had no shape to be caught in; it now
grades every mode the driver offers, and pins that `kUnknown` is deliberately
outside the set (`GetMode` reports it, a host may not ask for it). The class is
closed rather than the two sites patched: a whole-package cell holds
`strands_robots/` to zero two-argument `getattr` calls on an SDK module.
