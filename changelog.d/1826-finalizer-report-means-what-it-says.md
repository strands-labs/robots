### Fixed: the finalizer reports only a real cleanup failure, on every engine

`SimEngine.__del__` called `cleanup()` on any instance, so on one whose
`__init__` never finished it reported whichever attribute construction had not
reached yet as a cleanup failure. A plain caller typo reached this:
`IsaacSimulation(headles=False)` raised the intended `TypeError` and then, on
collection, also logged `Cleanup error during __del__: 'IsaacSimulation' object
has no attribute '_world_created'` - a second message naming an attribute the
failure does not turn on. `MuJoCoSimEngine` avoided that noise by overriding
`__del__` to swallow every exception silently, which also discarded real
failures, so the base class's own guarantee that a cleanup failure is logged did
not hold for the default backend. Concrete engines now declare
`self._init_complete = True` as the final statement of `__init__` and the shared
finalizer skips an instance that never acquired anything, rather than calling
`cleanup()` and silencing whatever it raises; the MuJoCo override is gone.

`IsaacSimulation.__repr__` no longer raises on such an instance. It is what a
traceback or a failing assertion renders, and `[AttributeError ... raised in
repr()]` was actively misdirecting: it named `_world_created` for a failure whose
real cause was a missing `_cameras`. It now reports the lifecycle fact and names
no attribute. `IsaacSimulation.cleanup`'s docstring no longer claims the class
has no `__del__` finalizer - it inherits one, and its reasons are reasons not to
*rely* on garbage collection.
