### Added: a rollout can be watched without taking the hook that cancels it

`run_policy` accepts an `observer` callable that receives one `RunPolicyStarted`,
one `RunPolicyStep` per completed `send_action` call, and one `RunPolicyEnded`.
A complete backend breakdown says what physically applied; a coarse error keeps
that state explicitly unknown.

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
  to `full` / `partial` / `none` / `unknown`. `partial` and `none` are emitted
  only from a valid complete per-key breakdown. A coarse atomic refusal is
  `unknown` with empty explicit key tuples: the input keys are not evidence of
  what reached physical state. Those refusals still feed the existing dead-rollout
  probe through the backend error text, so fail-fast is not weakened. Coarse
  steps are also excluded from aggregate action-rate denominators rather than
  being counted as confirmed misses; `action_errors` and the result text retain
  the uncertainty.
- **`observation_age_steps`** - the authoritative nonnegative count of completed
  rollout action attempts since the snapshot was sampled. Sync chunks report
  their chunk index; async prefetch carries the number of remaining old-chunk
  attempts through the swap and adds the new chunk index; an active recording
  refreshes every frame and reports zero. On an `unknown` action resolution it
  does not claim physical advancement.
- **`observation_is_chunk_reused`** - the narrower chunk-position signal: true
  only for a later action using the same chunk-start snapshot. It deliberately
  does not claim freshness; the first action after an async swap can have a
  positive age without yet reusing that snapshot inside its new chunk.
- **`legacy_hook_outcome`** - `ok`, `cancelled`, `recording_error`, `error`, or
  `absent`.

**The step a cancellation aborts on is reported, and that is the point.** The
legacy hook runs *after* `send_action` and `step_count` is incremented *after* the
hook, so an action a cancelling or recording-failing hook aborts on has already
advanced the world while being excluded from `steps_used`, from the video cadence
and from the resolution denominator. A lane that inherited that boundary would be
silent about the one step a cancellation is usually debugged from.
`applied_action_index == legacy_step_index` on every step, including the aborting
one: both identify the same zero-based action passed to the hook, and
`legacy_hook_outcome` identifies the abort. Terminal accounting uses a different
boundary, so `RunPolicyEnded.applied_actions` can exceed `legacy_steps_used` by
one on that path.

Emission is in a `finally` around the hook, which is also what keeps the hook's
own behaviour unchanged: each `except` re-raises exactly what it raised before, so
the exception type, its traceback and its `__cause__` reach the outer handler
identically - the `finally` merely runs first. Measured against the same suites at
the same commit, `on_frame` receives an identical argument sequence with and
without an observer, and the rollout applies an identical action sequence and
returns identical existing payload fields whether the observer is absent, healthy,
or raising on every event.

The event shape is observer schema version 2. Once `RunPolicyStarted` dispatch is
attempted, exactly one `RunPolicyEnded` dispatch is attempted for every Python
exit from control or result assembly. That includes `KeyboardInterrupt`,
`SystemExit`, `GeneratorExit`, `asyncio.CancelledError`, and ordinary assembly
errors; non-cooperative exceptions preserve identity and traceback and still
propagate. If Step or Ended dispatch raises another non-cooperative exception
while one is already unwinding from the legacy hook or rollout, the original
remains primary and the secondary is logged and attached as an exception note. A preflight refusal emits neither
event. A non-callable `observer` is one such refusal: the facade returns its
standard `run_policy` error and a direct `PolicyRunner.run` raises `ValueError`,
both before world/robot discovery, policy, hook, clock, id, inference, or action
side effects.

Ordinary failures are contained but never hidden. An observer `Exception` or
`CooperativeStop` cannot change the rollout's outcome and never reaches the
`max_onframe_failures` watchdog - that
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
lookup is performed unless an observer is installed. When installed, sim time is
read only from cached `_world.sim_time` or `_sim_time`; telemetry never calls
`get_state`. `observer_failures` appears in the payload only for a rollout that
had one, so the documented payload is unchanged for every existing caller.

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
