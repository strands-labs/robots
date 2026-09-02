### Added: `g1_move_velocity` verb for the driver's velocity-triple loco write

`G1Driver.move_velocity` is the driver-side velocity-command entry point:
a caller passes a `(vx, vy, vyaw, duration)` quadruple and the driver
publishes `LocoClient.SetVelocity(vx, vy, vyaw, duration)` over the same
DDS singleton `ensure_dds` opens. The Python SDK exposes `SetVelocity`
as a public `LocoClient` method that walks the robot at the argument
triple for `duration` seconds; the neon bundle's `g1_move_velocity` verb
(`cagataycali/neon-the-g1/tools/g1_locomotion.py`) fronted the call
under a single-writer lock, clamped the arguments to the observed
envelope `strands_robots.tools.g1.g1_velocity_envelope` already
surfaces to an agent, and returned an rc envelope. This module is the
write-side companion of that read-only envelope (refs #358, #2965 for
the envelope, #2972 for the duration envelope, #3035 for the sibling
`g1_stop_move`), and it is the third locomotion write to reach the
driver's own path after `g1_stop_move` and `g1_wave_hand_loco`.

The driver's method itself is not yet plumbed on `G1Driver` today
(refs #358 for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a `move_velocity`
accessor with a message naming the verb, the `driver` parameter and
the accessor. Once the driver method lands the same call returns the
driver's envelope verbatim — this is the same shape `g1_stop_move`
(refs #3035), `g1_wave_hand_loco` (refs #3041), `g1_release_arm`
(refs #3034), `g1_balance_stand` (refs #3033), `g1_set_stand_height`
(refs #3031) and `g1_set_swing_height` (refs #3032) already ship.

The four data parameters carry the domains the sibling loco verbs
already use: `vx` / `vy` / `vyaw` are signed finite floats (a reverse
walk is a negative `vx`, a clockwise turn a negative `vyaw`) and go
through the shared `finite_number_error` validator; `duration` is a
positive finite float through `positive_finite_number_error`. Both
validators refuse `None`, non-numeric, `nan`, `inf` and the `bool`
subclass that would coerce to `0.0` / `1.0` silently — refusing the
numeric domain before dispatch keeps the driver's rc-decoded refusal
reserved for the SDK's own return codes (a caller reading the
`ERR_CODES` table surfaced by `g1_error_codes` sees numbers only the
SDK wrote).

The FSM gate is not consulted here. `G1Driver._check_motion_gates` is
the one gate for arm-SDK writes (`send_action` / `run_policy` /
`start_task`); `SetVelocity` sits on the loco side of the same DDS
singleton and belongs on the locomotion admission set `WALK_FSMS`
the driver's own path will check when the write lands. The envelope
clamps (`vx` / `vy` / `vyaw` magnitudes, `duration` upper bound) are
not restated here either — the neon bundle's verb clamped its
arguments before dispatch, and that decision belongs on the driver's
own write path so a caller who reaches `SetVelocity` through a
different entry point (a future `use_unitree` dispatcher, refs
#3037) is held to the same envelope.

The neon verb's `continuous=True` branch (which reached the SDK's
`LocoClient.Move(..., continous_move=True)` overload instead of
`SetVelocity`) is deliberately NOT included in this port. That
overload is a distinct SDK call with a distinct rc surface and
belongs on a distinct driver method the driver's own write path
will define when it lands; bundling the two SDK paths into one verb
here would trap the verb to whichever branch the driver plumbed
first.

The test suite grades sixteen shapes: SDK-load-hygiene, a
driver-side refusal round-trip, a future success envelope
round-trip, `None` / string / int driver refusal, call-count
invariance (one call to the verb produces exactly one call to
`driver.move_velocity` — no retry inside the wrapper because the
SDK's handler is not re-entrant), argument-quadruple pass-through
(no clamp/round/reshape), `None` / `nan` / `inf` / `bool` velocity
component refusal, zero / negative `duration` refusal, and the
SDK's `rc=3104` RPC-timeout refusal round-trip.
