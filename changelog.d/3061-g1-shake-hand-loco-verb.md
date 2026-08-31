### Added: `g1_shake_hand_loco` verb for the driver's `LocoClient.ShakeHand` write

`G1Driver.shake_hand_loco` is the driver-side `LocoClient.ShakeHand` entry
point: a caller passes an integer `stage` and the driver publishes
`LocoClient.ShakeHand` (which internally composes a `SetTaskId` payload
against a three-stage table) over the same DDS singleton `ensure_dds` opens.
The SDK exposes `ShakeHand` as a public `LocoClient` method that admits
`0` (reach the arm out), `1` (shake the extended hand) and `-1` (toggle the
SDK's internal stage counter; the SDK's own default reads through the
sentinel). The neon bundle's `g1_shake_hand_loco` verb
(`cagataycali/neon-the-g1/tools/g1_locomotion.py`) wrapped the call under a
single-writer lock and coerced the argument through `int(stage)` before
dispatch; the read-only half of that envelope already landed as
`strands_robots.tools.g1.g1_shake_hand_stage_envelope` (refs #358, #2976),
and this module is the write-side companion that hands the target to the
driver.

The driver's method itself is not yet plumbed on `G1Driver` today (refs #358
for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a `shake_hand_loco`
accessor with a message naming the verb, the `driver` parameter and the
accessor it read for. Once the driver lands, the same call returns the
envelope the driver wrote verbatim. This is the same shape `g1_set_fsm`
(refs #3025), `g1_set_stand_height` (refs #3031), `g1_set_swing_height`
(refs #3032), `g1_balance_stand` (refs #3033) and `g1_wave_hand_loco`
(refs #3041) already ship.

The verb refuses shape errors on `stage` at the tool surface:

- `None` — no defensible default. The three admitted stages are the data
  points the read-only envelope surfaces, and a caller who did not pass
  one has not decided the write. (The neon bundle defaulted to `-1`,
  which routes the SDK to toggle its internal counter — a semantic that
  requires the caller to have named the sentinel on purpose.)
- `bool` — `bool` is an `int` subclass, so `True` would coerce to `1`
  (shake) and `False` to `0` (reach out); both inside the admitted
  set, so a caller writing the boolean would dispatch a stage they did
  not name.
- non-`int` (`float`, `str`) — the neon wrapper's `int(...)` coercion
  silently transformed cross-type shapes rather than declined them.
  Matches the `g1_balance_stand` verb's own cross-type refusal shape a
  caller can grep for.

Out-of-set integers (e.g. `7`) reach the driver's own refusal or the SDK's
`rc=7303` ("Invalid task id (loco)") handler through the verb's
pass-through; the in-set admission belongs on the driver's write path and
the read-only envelope module, not on this verb (refs #2976).

`ShakeHand` dispatches through `SetTaskId` rather than the arm-SDK path, so
the driver's `_check_motion_gates` admission set (refs #2916) is not
consulted. The neon bundle documented the verb as call-twice-with-3s-between
to complete the motion; this verb does not chain the two calls or sleep
between them (that is orchestration the caller owns).

The verb passes the driver's envelope through unchanged (SDK-load-hygiene
holds: no `unitree_sdk2py` submodule pulls at import; refs #358).

Ports `cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_shake_hand_loco`.

Refs #358, #2916, #2976, #3025, #3033, #3041.
