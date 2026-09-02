### Added: `g1_safe_squat_to_stand` verb for the driver's compound SQUAT->STAND write

`G1Driver.safe_squat_to_stand` is the driver-side compound-posture entry
point for the SQUAT->STAND transition: a caller passes a Damp-preamble
duration in seconds and the driver publishes `LocoClient.Damp`, sleeps for
`preamble_s`, then issues `LocoClient.Squat2StandUp` over the same DDS
singleton `ensure_dds` opens. The Damp preamble is the SDK's
controller-to-controller handoff smoother — firing it against an unheld
robot leaves it slumping toward the floor, so the driver's own path is
where the FSM-set precondition gate (`{3, 4, 706}` — the read-only envelope
`strands_robots.tools.g1.g1_safe_posture_fsm_gates` names that set, refs
#358) and the `avg_knee <= 1.4` rad pose gate fire.

The neon bundle's `g1_safe_squat_to_stand` verb
(`cagataycali/neon-the-g1/tools/g1_safe_posture.py`) wrapped the call with
an `_assert_safe_for_damp` FSM+pose guard that refused outside `{3, 4,
706}` or when the average knee angle exceeded `1.4` rad; the read-only
half of that envelope already landed as
`strands_robots.tools.g1.g1_safe_posture_fsm_gates` and
`strands_robots.tools.g1.g1_damp_transition_envelope`, and this module is
the write-side companion that hands the target to the driver.

The driver's method itself is not yet plumbed on `G1Driver` today (refs
#358 for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a
`safe_squat_to_stand` accessor with a message naming the verb, the
`driver` parameter and the accessor. Once the driver method lands the
same call returns the driver's envelope verbatim — this is the same shape
`g1_set_fsm` (refs #3025), `g1_set_stand_height` (refs #3031),
`g1_set_swing_height` (refs #3032) and `g1_balance_stand` (refs #3033)
already ship.

`strands_robots.tools.g1.g1_safe_squat_to_stand.g1_safe_squat_to_stand`
is the agent-facing side of that write: one duck-typed call on
`driver.safe_squat_to_stand`, the envelope the driver produced returned
verbatim, and the same live-handle refusals every write-side verb in this
package owes (`driver` is `None`, a robot *name*, or any object without a
callable `safe_squat_to_stand`). The verb adds five data-parameter shape
refusals on top through the shared
`strands_robots.utils.positive_finite_number_error` validator — a `None`
`preamble_s`, a non-numeric shape, a `bool` payload (which would coerce
to a silent `1.0` or `0.0`), a `nan` / `inf` payload (which would silently
no-op or block indefinitely), and a non-positive value (`0.0` collapses
to a bare `Squat2StandUp` write the neon bundle documented as a distinct
`use_unitree` verb; a negative value raises `ValueError` from `time.sleep`).
The neon-bundle-observed usable-range admission itself is not enforced by
this verb — the module docstring names "does not refuse a `preamble_s`
outside the neon-bundle-observed usable range" as one of the things this
verb does not do, and refusing the domain here would fork the neon
bundle's admission set into a second source of truth this module would
then have to keep in sync with the envelope lookup.

The FSM and pose gates the neon bundle's own `_assert_safe_for_damp`
implements are also not consulted here — the driver's own
`_check_motion_gates` (refs #2916) is the one FSM gate for actuation, and
the LowState pose-check belongs on the same side because the driver
already caches `LowState` for its own subscribers. A second gate call
here would double the FSM read against the driver's cache, would refuse
a preamble the driver's own path admits, and would fork the FSM-set
precondition table into a second source of truth.

`import strands_robots.tools.g1.g1_safe_squat_to_stand` pulls no
`unitree_sdk2py` submodule (the package's SDK-load-hygiene contract, refs
#358) and the sibling test at
`tests/drivers/test_g1_safe_squat_to_stand_writes_the_driver_envelope.py`
holds every one of the shape-refusal, pass-through, single-call and
default-preamble cells the driver's own path will surface once the
method lands.

Refs #358, #2916, #3025, #3031, #3032, #3033.
