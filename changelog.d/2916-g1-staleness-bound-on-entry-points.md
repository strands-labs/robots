### Fixed: the FSM staleness bound is asked on entry-point paths too

PR #2916 landed the FSM refresher off the control-loop thread and a 1.0 s
staleness bound so the loop's per-step re-gate refuses a step on a cache the
refresher has not renewed. The bound sat inside `if not refresh:` in
`_check_motion_gates`, so it fired for the loop's per-step re-gate
(`refresh=False`) and not for the one-shot entry points
(`send_action` / `start_task` / `run_policy` admission, all `refresh=True`).

`_refresh_fsm_id`'s refused-reading branch deliberately keeps `_fsm_id` and
deliberately does not stamp `_fsm_read_at` (so a transient `CheckMode()`
failure does not clobber the last-known id), and the bound is what backs that
tolerance. But the tolerance only closes if every path that writes `rt/lowcmd`
asks the same question. Same driver, same cached reading, same age past the
same bound - the loop stops and `send_action` published. Measured pre-fix:
a driver whose `CheckMode()` starts failing after one good admission read
publishes at ages 1.5 x, 3 x and 6 x the bound; publication continues on an
arbitrarily stale FSM, refused only when `_fsm_read_at is None` (which the
OK branch always stamps in production, so that case is unreachable via
admission).

The fix asks the same question on both paths. On the entry-point path
(`refresh=True`), `age is None` is tolerated - the caller has already run
`_refresh_fsm_id()`, and `_fsm_id is not None` on that path implies the OK
branch stamped `_fsm_read_at`, so `age is None` is production-unreachable
and refusing on it would ripple failures into cells that assign `_fsm_id`
directly without going through admission. On the loop path (`refresh=False`)
the strict `age is None or age > bound` refusal is preserved - the loop path
is the backstop for a refresher that never runs.

New test file `test_g1_staleness_bound_is_asked_on_entry_points.py` (5
cells): direct pin that `send_action` refuses when the cached FSM is over
the bound; a companion cell that `send_action` succeeds inside the bound; a
regression cell that measures unboundedness at 1.5 x, 3 x and 6 x the bound
(before the fix all three publish, after the fix all three refuse); a pin
that `start_task`'s admission gate refuses on `scope="motion"` too; and a
production-invariant cell that the entry-point tolerance for `age is None`
holds while the loop path continues to refuse on it.

Refs harness#361 and #2765 (both remain open for the wire-side regime
question this PR does not answer).
