### Added: `g1_wave_hand_loco` verb for the driver's `LocoClient.WaveHand` write

`G1Driver.wave_hand_loco` is the driver-side `LocoClient.WaveHand` entry
point: a caller passes a boolean `turn_flag` and the driver publishes
`LocoClient.WaveHand` (which internally composes one of two `SetTaskId`
payloads) over the same DDS singleton `ensure_dds` opens. The SDK exposes
`WaveHand` as a public `LocoClient` method that composes `turn_flag=False`
into the wave-in-place task and `turn_flag=True` into the wave-and-turn-around
task (the two variants the neon bundle observed in-hand). The neon bundle's
`g1_wave_hand_loco` verb (`cagataycali/neon-the-g1/tools/g1_locomotion.py`)
wrapped the call under a single-writer lock and coerced the argument through
`bool(turn)` before dispatch; the read-only half of that envelope already
landed as `strands_robots.tools.g1.g1_wave_hand_turn_flag_envelope` (refs
#358), and this module is the write-side companion that hands the target to
the driver.

The driver's method itself is not yet plumbed on `G1Driver` today (refs #358
for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a `wave_hand_loco`
accessor with a message naming the verb, the `driver` parameter and the
accessor. Once the driver method lands the same call returns the driver's
envelope verbatim — this is the same shape `g1_set_fsm` (refs #3025),
`g1_set_stand_height` (refs #3031), `g1_set_swing_height` (refs #3032) and
`g1_balance_stand` (refs #3033) already ship.

`strands_robots.tools.g1.g1_wave_hand_loco.g1_wave_hand_loco` is the
agent-facing side of that write: one duck-typed call on
`driver.wave_hand_loco`, the envelope the driver produced returned verbatim,
and the same live-handle refusals every write-side verb in this package owes
(`driver` is `None`, a robot *name*, or any object without a callable
`wave_hand_loco`). The verb adds two data-parameter shape refusals on top —
a `None` `turn_flag` (no default is defensible, the two admitted variants
`False` and `True` are the two data points the read-only envelope surfaces
and a caller who did not pass one has not decided the write) and a non-`bool`
`turn_flag` (an `int`, `float` or `str` payload which the neon wrapper's own
`bool(turn)` coercion would silently transform into an admitted task id the
caller had not named on purpose) — using inline `isinstance` shape checks
that mirror the read-only envelope module's own `turn_flag must be bool`
refusal so both paths render the same shape a caller can grep for. The
in-set admission (both `False` and `True` today) itself is not enforced by
this verb — the module docstring names "does not refuse a `turn_flag`
through Python's `bool()` coercion" as one of the things this verb does not
do beyond the shape refusal, because the two admitted variants are the two
data points the envelope module names as source of truth, and a firmware
release that narrowed the set would land on the envelope module and this
verb would pass the narrower refusal through the driver. Importing the
module pulls no `unitree_sdk2py` submodule (the package's SDK-load-hygiene
contract, refs #358), and the module docstring names the five things this
verb does not do (encode the `LocoClient.WaveHand` dispatch, decode the
SDK's `rc`, restate the driver's refusal wording, compose the `SetTaskId`
payload id, chain a companion FSM transition) so a caller reading it does
not misread the surface. Unlike the arm-SDK write verbs `WaveHand`
dispatches through `SetTaskId` rather than the arm-SDK path, so this verb
does not read the driver's `_check_motion_gates` admission set — the neon
bundle's docstring named the verb as "does not require FSM 500+ because it
uses SetTaskId" (the *observed* behaviour of the wave varies with the
current FSM the same way the sibling `g1_shake_hand_loco` primitive does,
but the SDK admits the call from every FSM).

The test suite
(`tests/drivers/test_g1_wave_hand_loco_writes_the_driver_envelope.py`)
grades fourteen shapes: the SDK-load-hygiene pin, two driver-side envelopes
as pass-through (a driver-side refusal, a future success envelope), the two
live-handle refusals through the shared `live_handle_refusal` guard, one
exactly-once write pin, two admitted-value pins (`False` wave-in-place and
`True` wave-and-turn-around, both reaching the driver verbatim), one
wrong-shape-driver-not-called ordering pin, and four data-parameter shape
refusals (missing `turn_flag`, `int` `turn_flag` which `bool()` would
coerce to a silent task id, `str` `turn_flag` which `bool()` would coerce
because any non-empty string is truthy, and `float` `turn_flag`). The
universal auto-discovery test at
`tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py` grades
the verb the moment it lands (its first parameter is `driver: Any`), so
the two live-handle refusal rules are held by that suite too.
