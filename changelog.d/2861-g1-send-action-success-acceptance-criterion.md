### Tests: the G1 acceptance criterion is pinned as a marker a wiring commit must turn over

`harness#361` asks for one thing the tree did not grade: that `send_action`
returns `status="success"` on a connected driver with a decoded `LowState_` and
a healthy pack. Every box on that checklist is satisfied by a body existing or
by a mocked test passing, so it has read as complete thirteen times. The
existing pin grades the *un*-reachability (the refusal must name the FSM, not
the battery); nothing graded the positive outcome.

`tests/drivers/test_g1_send_action_succeeds_on_a_healthy_wired_driver.py`
adds it as `pytest.mark.xfail(strict=True)`: today it xfails against the shipped
`FSM id unknown` refusal, and the day a motion-switcher decoder gives `_fsm_id`
a producer it XPASSes, which `strict=True` turns into a failure the wiring
commit clears by deleting the marker.

A strict xfail is only a checkpoint for the shapes that can reach XPASS, so the
reachable set is measured rather than assumed. The fixture drives
`_on_lowstate` and `_on_bms` instead of assigning `_mode_machine` and
`_battery`, because a fixture that assigns bypasses the callbacks where those
producers live -- with a producer planted beside `_mode_machine`, an assigning
fixture fires nothing in this file. Driving the decoders turns the marker over
for a producer added in either decoder, in `__init__`, in the gate, or in
`send_action`. A producer added inside `connect_eagerly` stays out of reach,
since that method opens a DDS participant; it is caught by the assignment-count
cell in the un-reachability pin, and the module docstring records the split.
