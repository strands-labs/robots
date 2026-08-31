### Added: `g1_get_task_status` verb for the driver's control-loop snapshot

`G1Driver.get_task_status` reports whether the driver's control loop is
rolling out a policy (or whether one has ever run), and if so the eight
fields `_ControlLoop.snapshot` writes: `running`, `steps`, `refusals`,
`elapsed_s`, the two budgets (`duration_budget_s` / `n_steps_budget`),
the loop's `hz`, and the two `fsm_refresh_hz` / `fsm_reads` fields that
name the FSM refresher thread filling the cache the re-gate reads. A
poller that misses the running window still sees the loop's terminal
snapshot: the driver stashes `_last_task_snapshot` under the admission
lock right before the `finally` clears `self._loop`, so every
self-terminating exit reason (`n_steps`, `duration`, `gate`, `policy`,
`publish`) round-trips to the caller instead of collapsing to "no task
has been started" once the thread joins.

The mesh's status wire consumes that envelope directly, but an agent
that wants the same read has to reach through the driver's Python
surface. This adds `strands_robots.tools.g1.g1_task_status.g1_get_task_status`
as the agent-facing side of that read: one duck-typed call on
`driver.get_task_status`, one reshape into a flat `@tool` dict, and a
`present` boolean that lets a caller tell the just-connected "no task
has been started" shape apart from a finished loop's stashed snapshot
without parsing the free-text `reason` string. Read-only; no bus is
touched, no locomotion client is opened, no FSM gate is consulted (the
driver's method itself is a snapshot read under `_task_admission`, and
neither "no task running" nor "loop finished with exit_reason=n_steps"
is a motion refusal). The verb is the third driver-instance-taking
entry in `strands_robots.tools.g1` after `g1_get_state` and `g1_battery`,
and follows the same duck-typed / SDK-load-hygiene contract those two do
(refs strands-labs/robots#358): the driver argument is typed `Any` to
keep the module out of the import cycle the driver's own `ensure_dds`
reach into this package would close, and `import
strands_robots.tools.g1.g1_task_status` pulls no `unitree_sdk2py`
submodule.

The tests in `tests/drivers/test_g1_task_status_reads_the_driver_envelope.py`
pin four shapes: the SDK-load-hygiene contract, the "no task has been
started" envelope reshaping to `present=False`, a running loop reshaping
to `present=True` with every snapshot field round-tripping verbatim, a
finished loop reshaping to `present=True running=False` (so a caller
reading `present` can tell it apart from the "no task" sentinel where
both are falsey), the single-read invariant (the verb calls the driver
exactly once so a caller reading `elapsed_s` cannot see two snapshots
on the same call), and the envelope's own `status` field reaching the
returned dict verbatim (so a future refusal on this driver path cannot
be masked to success by the verb). Every shape is read off the driver's
own writer rather than restated, so a rename on the driver side moves
both the write path and this verb together.

The verb refuses a wrong handle rather than dereferencing it. `driver` is a
live object typed `Any`, so the generated tool schema carries nothing telling
a model the argument cannot be synthesized, and the verb's first statement was
the accessor call: all six handle shapes a caller plausibly arrives with
(`None`, a robot *name*, an empty mapping, an integer, a list, and an object
carrying `get_task_status` as data rather than as a method) surfaced as
`AttributeError` or `TypeError` past the structured response, which is the one
thing an `AgentTool` handler must never do. `g1_get_task_status` now consults
the package's shared live-handle guard first and returns the error envelope
every `@tool` owes its caller, naming the verb, the `driver` parameter and the
type it received.

That guard gains one owner in the process. `snapshot_handle_refusal` already
bound the `_snapshot` accessor for the five sensor verbs; the envelope
construction and the four invariants behind it are now
`live_handle_refusal(verb, driver, accessor=..., reads=..., expected=...)`, and
`snapshot_handle_refusal` is that function bound to `_snapshot`. The five
existing verbs' refusal strings are byte-identical either side of the change,
so the factoring carries no behaviour with it.

The family sweep that is supposed to enforce this rule could not see the verb.
It discovered its population with `getattr(module, info.name)`, which reaches a
verb only when its function name equals its module name -- so `g1_state.py`'s
`g1_get_state` and `g1_task_status.py`'s `g1_get_task_status` both sat outside
it while the file documented a population "derived ... by signature rather than
naming it" that holds a new verb "to the rule the hour it lands". Discovery is
now by definition site (`__wrapped__.__module__`) and signature, independent of
what the module is called, and coroutine verbs are awaited rather than graded
as coroutine objects -- an unawaited verb is inside the population and still
ungraded. Two cells pin the blind spot shut: one names both verbs whose
function name differs from their module, and one refuses to let the population
consist only of verbs that agree with their module name, which would let the
name-keyed scan return unnoticed. Measured either side of the guard: the
widened sweep is 22 passed with it and 8 failed without it, so the rule is
load-bearing rather than decorative. The over-reach control that asserts a
`present` answer now runs against the AST-derived set of verbs that actually
call `snapshot_handle_refusal`, because the rest of the family answers a
different accessor and a different shape and carries its own healthy-handle
control in its own suite.
