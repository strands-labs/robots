### Feature

- Added `strands_robots.tools.g1.g1_list_wave_hand_turn_flags` and
  `strands_robots.tools.g1.g1_wave_hand_turn_flag_admits`: pure-reference
  agent-facing lookups over the two `turn_flag` values
  `unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.WaveHand(turn_flag=bool)`
  admits as dispatches (`False` wave in place, `True` wave while
  turning 180 degrees), so a caller can decide the SDK's `rc=7303`
  refusal decidably before a future driver-side wrapper for
  `WaveHand` fires. Snapshotted from the neon bundle's
  `g1_wave_hand_loco` wrapper
  (`cagataycali/neon-the-g1/tools/g1_locomotion.py`) which drove the
  same two values against the real robot with the boolean surface.
  Each descriptor carries `composed_task_id` (the `SetTaskId` value
  the variant routes through: `0` for `False`, `1` for `True` -
  both members of `g1_loco_task_ids._LOCO_TASK_MAP`, so the two
  envelopes share the SDK's task-dispatch handler), `sdk_method`
  (always `"WaveHand"`, the caller-facing entry), and
  `admits_loco_writes` (always `True` here, surfaced for shape
  parity with `g1_loco_task_ids` and `g1_shake_hand_stage_envelope`).
  Both verbs surface `WALK_FSMS` so a caller comparing an intended
  dispatch against both conditions (envelope + gate) has the FSM
  set on hand; the `refusals` list carries the `rc=7303` invalid-
  task code and the `rc=7404` gate-refused code the driver's
  `_check_motion_gates` would quote on the follow-up write.
  `g1_wave_hand_turn_flag_admits` refuses non-`bool` inputs
  (including `int` values Python's `bool()` would coerce) as shape
  errors at the tool surface rather than resolving them through
  the coercion the neon wrapper's `bool(turn)` applies at dispatch
  time. No DDS is touched, no `unitree_sdk2py` submodule loads at
  import (the same hygiene rule every other file in the package
  carries). Refs #358.
