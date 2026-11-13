### Internal: the run loop's shared tick helpers live in a leaf module, so the import order of `robot/run/*` never matters

`robot/run/common.py` (the `emit`, `announce`, `approve_run`, `declared`, `rate` and `label` helpers that
`loop.py` and `lockstep.py` both import) now takes its `Bound`, `RunResult` and `RunPlan` names only under
`TYPE_CHECKING`. It imports nothing from its own package at load time, so no import order can enter `loop.py`
before a helper is defined. Behaviour and the public surface are unchanged.
