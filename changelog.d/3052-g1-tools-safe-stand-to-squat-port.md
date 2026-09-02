### Added: `g1_safe_stand_to_squat` verb for the driver's compound STAND->SQUAT write

`G1Driver.safe_stand_to_squat` is the driver-side compound-posture entry
point for the STAND->SQUAT transition: a caller passes a Damp-preamble
duration in seconds and the driver publishes `LocoClient.Damp`, sleeps for
`preamble_s`, then issues `LocoClient.SetFsmId(2)` over the same DDS
singleton `ensure_dds` opens. The bare `SetFsmId(2)` route sidesteps the
SDK's own `LocoClient.StandUp2Squat`, which mis-calls `SetFsmId(706)`
(the Squat2StandUp FSM) and therefore does the opposite of what its name
promises; the neon bundle documented the workaround in
`cagataycali/neon-the-g1/tools/g1_safe_posture.py` and the driver's own
path (once landed) follows the same route.

The Damp preamble is the SDK's controller-to-controller handoff smoother —
firing it against an unheld robot leaves it slumping toward the floor, so
the driver's own path is where the FSM-set precondition gate
(`{500, 501, 801}` — the read-only envelope
`strands_robots.tools.g1.g1_safe_posture_fsm_gates` names that set, refs
#358) fires. The neon variant kept `pose_check=False` because the
upright-FSM precondition already implies a controller-managed
weight-bearing pose, so a knee-angle re-read against the driver's own
LowState cache would be redundant.

The neon bundle's `g1_safe_stand_to_squat` verb
(`cagataycali/neon-the-g1/tools/g1_safe_posture.py`) wrapped the call with
an `_assert_safe_for_damp` guard that refused outside
`{500, 501, 801}`; the read-only half of that envelope already landed as
`strands_robots.tools.g1.g1_safe_posture_fsm_gates` and
`strands_robots.tools.g1.g1_damp_transition_envelope`, and this module is
the write-side companion that hands the target to the driver.

The driver's method itself is not yet plumbed on `G1Driver` today (refs
#358 for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a
`safe_stand_to_squat` accessor with a message naming the verb, the
`driver` parameter and the accessor. Once the driver method lands the
same call returns the driver's envelope verbatim — this is the same shape
`g1_set_fsm` (refs #3025), `g1_set_stand_height` (refs #3031),
`g1_set_swing_height` (refs #3032), `g1_balance_stand` (refs #3033),
`g1_safe_squat_to_stand` (refs #3044) and `g1_safe_lie_to_stand`
(refs #3048) already ship, completing the three safe-posture-verb
row `g1_safe_posture_fsm_gates` names.

The `preamble_s` domain-refusal uses the shared
`positive_finite_number_error` validator (`nan`/`inf`/non-numeric/negative/
zero/bool subclass are all refused with a named envelope). The test suite
grades sixteen shapes: SDK-load-hygiene, a driver-side refusal round-trip,
a future success envelope round-trip, `None` / wrong-shape driver refusal,
call-count invariance, argument pass-through, the neon-bundle default,
missing/wrong-shape/zero/negative/boolean/non-finite `preamble_s` refusal,
and a large value reaching the driver unchanged.
