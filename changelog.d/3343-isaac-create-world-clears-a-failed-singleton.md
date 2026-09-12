### Fixed: a failed `create_world` no longer leaves a live World for the retry to inherit

`World` registers itself as `SimulationContext.instance()`, a process-wide
singleton. `create_world`'s failure path dropped the Python reference and nothing
else:

```python
except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:
    self._world = None          # releases nothing
    return {"status": "error", ...}
```

`destroy()` does the real teardown - `stop()`, `clear_instance()`, then drop - so
this was an asymmetry between two paths that release the same object.

It is not a leak that costs memory. It is a leak that returns a wrong answer to
the caller's obvious next move. Measured against the real
`isaacsim.core.api.World` on an A10G under Isaac Sim 6.0.1:

```
World(physics_dt=1/60)      -> registered as SimulationContext.instance()
del w1; gc.collect()        -> instance STILL alive
World(physics_dt=1/120)     -> the SAME object; get_physics_dt() is 1/60
clear_instance()
World(physics_dt=1/120)     -> a fresh world; get_physics_dt() is 1/120
```

A second construction returns the first object **and keeps the first call's
arguments**. So a caller whose `create_world` failed - and the documented causes
are real, a `physics_dt` the integrator cannot honour, a `set_gravity` shape Isaac
Sim 5.1 rejects - who then corrected the value and called `create_world` again,
got a world running the *failed attempt's* physics. The new result reported the
new value, because `world_info` is built from the config and the arguments and is
never read back off the world.

`destroy()` could not clean it up either: it early-returns on
`self._world is not None`, and that is `None`.

The failure path now performs the same teardown as `destroy`, under the same
narrow handler. `stop()` and `clear_instance()` can themselves raise on a
half-built world, and this path already has a failure to report, so a cleanup
error is logged and the original error is what reaches the caller - pinned in both
directions, including that the reference is dropped even when cleanup raised.

The regression test's fake `World` is a genuine singleton that keeps its first
kwargs, because that is the mechanism. A fake handing out a fresh object per call
let the retry assertions pass on the broken code - the retry got a clean world for
the wrong reason - which is how the first version of this test measured 1 pre-fix
failure where the honest fake measures 3.

Two properties of the teardown were wrong in the first version of this fix, both
found by adversarial review, and both silently restored the leak for the most
likely failure:

* **It must not be gated on `self._world`.** That name is bound only after
  `World(...)` *returns*, while the singleton registers inside
  `SimulationContext.__new__`. So every failure raised by the constructor itself -
  an unusable `physics_dt`, a device the host cannot provide, which is the
  likeliest cause here - left a registered instance that an
  `if self._world is not None` guard skipped entirely. `clear_instance` is a
  classmethod, so the local `World` symbol reaches the live instance whether or
  not this object ever held a reference to it.
* **`stop()` and `clear_instance()` need separate handlers.** `stop()` is the call
  that raises on a half-built world - `destroy()`'s own comment says so - and with
  both in one `try` a raising `stop()` skipped the `clear_instance()` that is the
  entire point. The first version's own test exercised exactly that input while
  asserting around the invariant rather than on it.

Both are now pinned by tests that fail when the single-`try`, `self._world`-gated
shape is restored, and the fake `World` registers itself in `__new__` and can fail
in `__init__`, because a fake that only fails later cannot reach either case.
