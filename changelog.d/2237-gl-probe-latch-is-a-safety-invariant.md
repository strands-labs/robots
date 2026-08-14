### Quality: the GL probe builds its renderer at most once per process, latched outside the cache

`tests/simulation/mujoco/_gl_probe.gl_available()` decides whether a MuJoCo
render test can run by constructing a 1x1 offscreen renderer and reporting
whether it worked. On a headless host that construction fails gracefully and
reports `False`, which is the whole point of the probe -- but a *second*
construction in the same process does not fail gracefully. It aborts the
interpreter uncatchably:

```
libc++abi: __cxa_guard_acquire detected recursive initialization: do you have a
function-local static variable whose initialization depends on that function?
Aborted (core dumped)
```

The answer lived only in the `functools.cache` on `gl_available`, and the
probe's own contract test clears that cache to exercise the
`ROBOT_TEST_MUJOCO=0` force-skip. Clearing it re-armed the hardware probe, so
the next caller re-ran the construction and took the process with it -- and the
abort surfaced in whichever test called `gl_available()` next, not in the test
that cleared the cache. `except Exception` cannot see an abort, so nothing
reported it as a GL problem at all.

The hardware answer is now latched in a module-level sentinel the cache cannot
reset, written *before* the construction is attempted so a graceful failure
cannot be handed a retry. `gl_available` still re-reads `ROBOT_TEST_MUJOCO` on a
cleared cache -- the force-skip contract is unchanged, and the force-skip no
longer consumes or poisons the latch either.

Pinned on a GL host as well as a headless one: a `mujoco.Renderer` stand-in that
fails if it is ever constructed proves a cleared cache reaches no second
construction, and a stand-in that fails on its *first* call proves a graceful
failure is latched rather than retried. No production behaviour changes.

The cases hold with `ROBOT_TEST_MUJOCO=0` already in the environment as well as
without it -- the configuration the skip reason points an operator at on a
known-bad runner, which is the host class this gating exists for. An autouse
fixture gives every case the unforced environment its assertions assume and
primes an unset latch rather than probing it, since that setting exists to keep
such a host from attempting GL at all. A child pytest run over the module pins
that configuration.
