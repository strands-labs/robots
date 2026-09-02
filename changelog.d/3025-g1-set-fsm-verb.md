### Added: `g1_set_fsm` verb for the driver's SetFsmId write

`G1Driver.set_fsm` is the driver-side ``SetFsmId`` entry point: a caller
passes a target FSM id (``1`` Damp, ``500`` Start, ``501`` Walk, ``801``
BalanceExpert, ``3`` Sit, ``0`` ZeroTorque, ``2`` Squat2Stand, ``4``
Locomotion, ``706`` BalanceLie, ``802`` DampToBalance) and the driver
publishes the SDK's `LocoClient.SetFsmId` call over the same DDS singleton
`ensure_dds` opens, waits for the transition to settle, and reads the
driver's own live `fsm_id` back to surface the fsm-before / fsm-after / rc
round-trip the neon bundle's `g1_set_fsm` verb documented (a transition the
SDK refused silently still shows the fsm-after equal to fsm-before). The
`strands_robots.tools.g1.g1_fsm_targets` lookup is the read-only half of
this conversation: it lists the id set the SDK admits without opening a
write path; this verb is the write half.

The driver's method itself is not yet plumbed on `G1Driver` today (refs
#358 for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a `set_fsm` accessor
with a message naming the verb, the `driver` parameter and the accessor.
Once the driver method lands the same call returns the driver's envelope
verbatim — this is the same shape `g1_start_task` (whose driver method
refuses with a registry-not-wired string today) already ships.

`strands_robots.tools.g1.g1_set_fsm.g1_set_fsm` is the agent-facing side
of that write: one duck-typed call on `driver.set_fsm`, the envelope the
driver produced returned verbatim, and the same live-handle refusals every
write-side verb in this package owes (`driver` is `None`, a robot *name*,
or any object without a callable `set_fsm`). The verb adds four
data-parameter refusals on top — a `None` `fsm_id`, a non-int shape, a
`bool` payload (which would coerce to `1` Damp silently), and a
non-positive-finite `wait` (validated through the shared
`positive_finite_number_error`) — so the driver is not asked to sleep on
`nan` / `inf` / negative or transition to a state the caller did not name.
Importing the module pulls no `unitree_sdk2py` submodule (the package's
SDK-load-hygiene contract, refs #358), and the module docstring names the
five things this verb does not do (decide which ids the SDK admits, decode
`rc=7302`, warn about safety, restate the driver's refusal wording,
schedule transitions) so a caller reading it does not misread the surface.

The test suite (`tests/drivers/test_g1_set_fsm_writes_the_driver_envelope.py`)
grades sixteen shapes: the SDK-load-hygiene pin, three driver-side
envelopes as pass-through (a rc=7302 refusal, a gate refusal, a future
success envelope carrying the fsm-round-trip), the two live-handle
refusals through the shared `live_handle_refusal` guard, seven data-parameter
refusals (missing `fsm_id`, wrong-shape `fsm_id`, `bool` `fsm_id`, negative
`wait`, `nan` `wait`, zero `wait`), one exactly-once write pin, one
arguments-pass-through pin, and one signature-default parity pin. The
universal auto-discovery test at
`tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py` grades
the verb the moment it lands (its first parameter is `driver: Any`), so
the two live-handle refusal rules are held by that suite too.
