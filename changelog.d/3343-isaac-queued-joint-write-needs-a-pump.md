### Fixed: a joint write queued with no pump to drain it is refused, not reported applied

`IsaacSimulation.set_joint_positions` applies the pose inline on the
`SimulationApp`-owning thread and otherwise queues it for `pump()`. `pump` is the
only consumer of that queue, and `run_pump_forever` is the only thing that runs
it - so a call from a worker thread with no pump engaged put the action into a
queue nobody reads and answered:

```python
{"status": "success", "content": [{"text": "Set joint positions (queued)."}]}
```

Measured from a worker thread with no pump: success reported, the articulation
still at its previous pose, one action left sitting in the queue.

This is the failure the method's own docstring says it avoids for bad *values*:

> Validation runs synchronously, before the write is queued for the main thread,
> so a rejected value is reported to the caller rather than raised on the pump
> thread - where the queued-action handler swallows it after this method has
> already answered `status="success"`.

A queue with no consumer is the same silence one level up, and strictly harder to
diagnose: there is not even a swallowed exception to find afterwards. The caller
sees a success, reads back the joint positions, gets the old pose, and has no
reason to suspect the write rather than their own indexing.

`set_joint_positions` was the **only** one of this backend's `_action_q`
producers without a pump guard. Every other main-thread-affine surface either
routes through `_marshal_main_thread_affine` or checks `_pump_running` directly,
as `get_body_state` does - so this was a gap in an otherwise uniform contract
rather than a missing convention.

The refusal is an error dict rather than the `RuntimeError`
`_marshal_main_thread_affine` raises for `reset`/`step`. Two reasons: this
surface's contract is the envelope throughout, and unlike those it could not
deadlock - it returned promptly having done nothing, which is precisely why the
silence is what needed closing. The message states that the pose was validated
and **not** applied, and names both remedies.

Nothing is left in the queue on refusal, which is pinned: putting the action and
*then* reporting an error would apply a pose the caller had been told failed, as
soon as anyone started a pump.

Value validation still runs first, also pinned. A bad joint name, a non-finite
value, an empty mapping and a length mismatch each keep their own diagnosis
instead of inheriting the threading one - reporting "start a pump" to someone who
passed a typo'd joint name would cost them a round before they saw the real
error.

`_pump_running` is now declared on the **class**, not only assigned in
`__init__`. Many test modules build a skeleton engine with
`IsaacSimulation.__new__` and seed only what the method under test reads, so a
guard reading this off `self` raised `AttributeError` in every one of them - a
failure *inside* the guard rather than the refusal it exists to make, naming an
attribute unrelated to what the test was about. It surfaced as two `[queued]`
cases in the cross-backend joint-state suite, which is a composition-only break:
this change alone is green, and so is that suite alone.

`False` is the default because it is the fail-**closed** direction: an engine that
has not been told a pump is running is treated as having none, so a queued write
is refused rather than stranded. `getattr(self, "_pump_running", True)` would
assume a consumer exists, which is the assumption the refusal was added to stop.

That cross-backend fixture now sets `_pump_running=True` for its queued cases,
which is the real deployment shape it means - a worker thread *plus* a running
pump. Without the pump there is no consumer, so the write it asserts lands only
because one exists.
