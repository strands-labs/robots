### Fixed: the main-thread pump no longer reads or writes a tensor view PhysX has invalidated

The stale-view gate this PR threads through the Isaac backend guards two distinct
harms, and only the first had a sweep. Advancing `_sim_time` over a scene PhysX is
not simulating reports success for an action that never applied; that axis is
covered. The second is an articulation tensor **read or write** against an
invalidated view, which `remove_object` measured as hanging until a two-minute
timeout, and which otherwise raises a bare `Exception` ("Failed to get DOF
positions from backend") that the narrow
`(RuntimeError, ValueError, AttributeError, TypeError)` handlers wrapped around
these reads cannot catch - widening them to `except Exception` being what AGENTS.md
forbids.

Three surfaces performed such a read or write with no gate. None of them advances
`_sim_time`, so the existing sweep - keyed on a body holding both a `_sim_time +=`
and a `world.step(...)`, with render helpers explicitly out of scope - could not see
any of them:

- **`pump` step 3** refreshed the joint cache for every robot. This is the worst
  placement in the backend: it runs on the main thread, and `run_pump_forever` wraps
  it in `try`/`finally` with no `except`, so one escape ends the loop and takes the
  app down - the same whole-app failure the snapshot immediately above it was added
  to prevent - while the hang wedges a live UI session. It now skips the refresh,
  keeps the last good cache rather than clearing it, and reports once instead of
  once per ~50 ms tick.
- **`_converge_render`** read and then **wrote** (`set_joint_positions` plus
  `set_joint_velocities`), `max(1, n)` times per call, reached from pump's step 2 on
  the default idle preview path. It now drops the pose-hold and keeps rendering, so
  the preview stays live rather than freezing until the caller resets, and it
  re-reads the flag per iteration for the reason `step` re-checks per batch: the
  loop holds no lock, so a worker's dynamic add lands between two iterations.
- **`set_joint_positions`** reads the live vector before writing so a partial dict
  updates only the DOFs it names. It refuses through the shared helper, placed after
  the robot resolves as `send_action`'s gate is, so a bad `robot_name` still reports
  itself. Off the main thread this verb is queued onto the pump, which is how its
  escape reached `run_pump_forever` too.

The trigger is ordinary shipped usage rather than an engineered race: a worker
thread's `remove_object`, or `load_scene`'s per-episode reload, invalidates the view
and the very next pump tick performs the read.

Pinned by `tests/simulation/isaac/test_the_main_thread_pump_does_not_touch_a_stale_tensor_view.py`,
whose sweep is keyed on the articulation **operation** rather than on the clock, so
it answers for this second axis the way the existing sweep answers for the first.
Its exemption map names the four surfaces that are covered by a caller's gate or
by running inside `reset()` itself, and checks both that each cited guard still
consults the flag and that no exemption has outlived the surface it names. 13 of its
23 cells fail on the pre-fix tree.
