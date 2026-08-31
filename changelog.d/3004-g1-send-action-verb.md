### Added: `g1_send_action` verb for the driver's one-frame `rt/lowcmd` write

`G1Driver.send_action` publishes one `LowCmd_` frame on `rt/lowcmd` for a
joint-name-keyed action dict.  It is the driver's *one-frame* write — a caller
who wants a schedule (500 Hz, 200 Hz) reaches
`G1Driver.run_policy` which owns the control loop today, or calls this verb on
their own timer.  The write path already re-gates through
`G1Driver._check_motion_gates` with scope `"arm"`, so a caller whose FSM is
outside the arm-SDK admission set `{500, 501, 801}` or whose battery is under
the driver's floor gets the driver's own refusal string (refs #2916).

`strands_robots.tools.g1.g1_send_action.g1_send_action` is the agent-facing
side of that single write: one duck-typed call on `driver.send_action`, the
envelope the driver produced returned verbatim, and the same live-handle
refusals every write-side verb in this package owes (`driver` is `None`, is a
robot *name*, or is any object without a callable `send_action`).  The verb
adds three `action`-parameter refusals on top — a `None` action, a non-dict
shape, and an empty dict — so the driver is not asked to admit a wire frame
that names no joint through the arm-SDK gate.  Importing the module pulls no
`unitree_sdk2py` submodule (the package's SDK-load-hygiene contract, refs
#358), and the module docstring names the four things this verb does not do
(build the `LowCmd_` payload, re-run the gate, schedule frames, decode
`fsm_id` / `mode_machine`) so a caller reading it does not misread the surface.

The test suite (`tests/drivers/test_g1_send_action_writes_the_driver_envelope.py`)
grades the four shapes the driver produces (success write, gate refusal,
publisher-not-initialised refusal, publish-error refusal) as pass-through, the
three `action` refusals as verb-side envelopes, and the two live-handle
refusals through the shared `live_handle_refusal` guard that every write-side
verb in this package already routes through.  The universal auto-discovery
test at `tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py`
grades the verb the moment it lands (its first parameter is `driver: Any`),
so the two live-handle refusal rules are held by that suite too.
