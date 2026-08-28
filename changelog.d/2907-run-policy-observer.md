### Added: a rollout can be watched without taking the hook that cancels it

`run_policy` accepts an `observer` callable that receives one `RunPolicyStarted`,
one `RunPolicyStep` per physically applied action, and one `RunPolicyEnded`.

`on_frame` looked like the seam for this and is not one. There is exactly one per
rollout and `SimEngine.run_policy` fills it from `_make_run_policy_hook`, so on
MuJoCo it already carries cooperative cancellation, the trajectory mirror,
rate-limited mesh step telemetry and the LeRobot dataset recorder, and on Isaac
and Newton the recording half of that. The facade does not accept one from the
caller at all. A consumer reaching for it through `PolicyRunner.run` directly does
not *add* observation - it replaces cancellation and recording with a visualiser,
and `stop_policy` stops working on that rollout.

The new lane sits beside that hook and reports the three things the hook's own
`(step, obs, action)` signature cannot carry:

- **`action_resolution`** - the backend's per-key `send_action` verdict, normalised
  to `full` / `partial` / `none` / `unknown`. `partial` is the case a single
  aggregate count hides, because the step *is* an error and the robot *did* move:
  a rollout driving one joint of six returns `status="success"` with
  `action_errors=0`.
- **`observation_is_chunk_reused`** - open-loop chunk replay feeds one observation
  to a whole chunk, so every action after the first acts on a stale snapshot
  unless an active recording forced a per-step refresh. Stated per step rather
  than left to a consumer to assume freshness it does not have.
- **`legacy_hook_outcome`** - `ok`, `cancelled`, `recording_error`, `error`, or
  `absent`.

**The step a cancellation aborts on is reported, and that is the point.** The
legacy hook runs *after* `send_action` and `step_count` is incremented *after* the
hook, so an action a cancelling or recording-failing hook aborts on has already
advanced the world while being excluded from `steps_used`, from the video cadence
and from the resolution denominator. A lane that inherited that boundary would be
silent about the one step a cancellation is usually debugged from.
`applied_action_index` counts what the world did, `legacy_step_index` mirrors what
the hook was told, and `RunPolicyEnded.applied_actions` exceeds
`legacy_steps_used` by one on exactly that path.

Emission is in a `finally` around the hook, which is also what keeps the hook's
own behaviour unchanged: each `except` re-raises exactly what it raised before, so
the exception type, its traceback and its `__cause__` reach the outer handler
identically - the `finally` merely runs first. Measured against the same suites at
the same commit, `on_frame` receives an identical argument sequence with and
without an observer, and the rollout applies an identical action sequence and
returns identical existing payload fields whether the observer is absent, healthy,
or raising on every event.

Failures are contained but never hidden. An observer exception cannot change the
rollout's outcome and never reaches the `max_onframe_failures` watchdog - that
exists so a recorder losing dataset frames cannot produce a silently empty
dataset (GH #117), which is not the same event as a visualiser that cannot draw.
`CooperativeStop` is contained too, and named explicitly: it is a `BaseException`
precisely so a hook's broad `except Exception` cannot swallow a cancellation, and
without naming it any observer could cancel a rollout it is only supposed to
watch. That is the whole reason the guard cannot simply be `except Exception`,
and it is why the clause is `except (CooperativeStop, Exception)` rather than
`except BaseException` - the smallest superset that keeps a stop from escaping an
observer. `KeyboardInterrupt`, `SystemExit`, `GeneratorExit` and
`asyncio.CancelledError` propagate, none of them being an `Exception` subclass: a
generator closed underneath a visualiser, or a task cancelled while one was
drawing, is a teardown rather than a drawing failure, and counting it as an
observer failure on a rollout that then ran to its full budget said the opposite.
Every contained failure increments `observer_failures`, reported in the result
json, so a stream with holes says it has them.

The lane is inert when unused: no clock is read, no id is minted and no sim-time
lookup is performed unless an observer is installed, and `observer_failures`
appears in the payload only for a rollout that had one, so the documented payload
is unchanged for every existing caller.

Payloads are borrowed rather than copied - the same objects the hook received -
because a per-step deep copy of an image-sized observation is the opposite of what
an observability lane should cost. Consumers treat them as read-only and snapshot
synchronously. Dispatch is synchronous on the rollout thread, so this is telemetry
and not a sandbox: a blocking observer blocks the robot, and that is documented
rather than defended against.

Scoped to `run_policy` (its `n_episodes > 1` path included, one lifecycle and one
`run_id` per episode) and `PolicyRunner.run`. `eval_policy`,
`evaluate_benchmark` and `run_multi_policy` are separate loops with different step
semantics and carry no observer, which the parameter documentation states rather
than leaving to be discovered.
