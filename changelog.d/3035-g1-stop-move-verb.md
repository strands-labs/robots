### Added: `g1_stop_move` verb for the driver's StopMove write

`G1Driver.stop_move` is the driver-side stop-move entry point: a caller
invokes the verb and the driver publishes `LocoClient.StopMove` over the
same DDS singleton `ensure_dds` opens. The Python SDK exposes `StopMove`
as a public `LocoClient` method that zeroes the last-commanded
`(vx, vy, vyaw)` velocity triple without changing the FSM the robot is
in; the neon bundle's `g1_stop_move` verb
(`cagataycali/neon-the-g1/tools/g1_locomotion.py`) wrapped the call under
a single-writer lock and returned an rc envelope with the same one-shot
contract (`SAFE to call anytime - doesn't change FSM, just kills
movement`). This module is the write-side companion of the read-only
velocity envelope `strands_robots.tools.g1.g1_velocity_envelope` that
already landed (refs #358, #2965).

The driver's method itself is not yet plumbed on `G1Driver` today (refs
#358 for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a `stop_move`
accessor with a message naming the verb, the `driver` parameter and the
accessor. Once the driver method lands the same call returns the
driver's envelope verbatim — this is the same shape `g1_release_arm`
(refs #3034), `g1_balance_stand` (refs #3033), `g1_set_stand_height`
(refs #3031) and `g1_set_swing_height` (refs #3032) already ship.

`strands_robots.tools.g1.g1_stop_move.g1_stop_move` is the agent-facing
side of that write: one duck-typed call on `driver.stop_move`, the
envelope the driver produced returned verbatim, and the same live-handle
refusals every write-side verb in this package owes (`driver` is `None`,
a robot *name*, or any object without a callable `stop_move`). The verb
adds no data-parameter refusals — the halt request has no caller-facing
shape beyond the driver handle, which is the one thing the neon bundle's
verb also asked for.

The FSM gate is not consulted here. The driver's `_check_motion_gates`
is the arm-write gate (`send_action` / `run_policy` / `start_task`); a
stop is the *end* of a walking window a prior `g1_move_velocity` /
`g1_walk_forward` / `g1_turn` opened, not a new locomotion frame, so
refusing it under a narrower gate here would leave the robot walking
when a caller asked it to stop. The neon bundle called `StopMove` an
emergency-stop path exactly because the SDK admits it on any FSM (refs
#358, #2916).

The SDK-load-hygiene contract every file under `strands_robots.tools.g1`
carries holds here too: `import strands_robots.tools.g1.g1_stop_move`
pulls no `unitree_sdk2py` submodule; the SDK loads only inside function
bodies (through the driver's own `stop_move` write path once it lands).

Contract-graded off the driver, no real `unitree_sdk2py` or DDS bus.
Eight tests: SDK-load hygiene, driver-side refusal round-trip, future
success envelope round-trip, three live-handle refusals (`None`, `str`,
`int`), single-call ordering, and the SDK's `rc=3104` RPC-timeout
refusal round-trip. Each cell names the one shape a caller can rely on;
none quote the driver's refusal wording verbatim (refs #2874).
