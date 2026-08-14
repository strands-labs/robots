### Fixed: the MuJoCo finalizer completes its teardown at interpreter shutdown

`MuJoCoSimEngine.cleanup()` opened with a function-local `import contextlib`.
A finalizer calls it during interpreter shutdown, where the import system is
already gone, so that first statement raised before any of the nine teardown
steps below it: `SimEngine.__del__`'s safety net released nothing - not the ROS
bridge, not the renderers, not the executor - and the only thing the operator
was told was a warning naming the interpreter (`sys.meta_path is None, Python
is likely shutting down`) rather than anything actionable. A script that called
`cleanup()` explicitly got the same warning per engine, reporting a failure on
the path where everything had succeeded.

`contextlib` is now imported at module scope, matching the nine other modules
in the package that do so - including `simulation/base.py`, which declares the
`cleanup`/`__del__` contract, and `mujoco/backend.py` beside it. A structural
test refuses a standard-library import inside any `cleanup`, `destroy` or
`__del__` in the package, and the behavioural half drives a child interpreter
to real shutdown and asserts the teardown steps are reached.
