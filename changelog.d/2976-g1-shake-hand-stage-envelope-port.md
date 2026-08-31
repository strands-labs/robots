### Feature

- Added `strands_robots.tools.g1.g1_list_shake_hand_stages` and
  `strands_robots.tools.g1.g1_shake_hand_stage_admits`: pure-reference
  agent-facing lookups over the three `stage` values
  `unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.ShakeHand(stage=int)`
  admits as dispatches (`-1` toggle, `0` reach out, `1` shake), so a
  caller can decide the SDK's `rc=7303` refusal decidably before a
  future driver-side wrapper for `ShakeHand` fires. Snapshotted from
  the neon bundle's `g1_shake_hand_loco` wrapper
  (`cagataycali/neon-the-g1/tools/g1_locomotion.py`) which drove the
  same three values against the real robot. Each descriptor carries
  `sequenced` (`True` on the ordered reach-and-shake pair, `False`
  on the toggle sentinel), `toggle` (`True` on `-1`, the SDK's own
  advance-the-counter default), and `admits_loco_writes` (always
  `True` here, surfaced for shape parity with `g1_loco_task_ids`
  and `g1_fsm_targets`). Both verbs surface `WALK_FSMS` so a caller
  comparing an intended dispatch against both conditions (envelope
  + gate) has the FSM set on hand; the `refusals` list carries the
  `rc=7303` invalid-stage code and the `rc=7404` gate-refused code
  the driver's `_check_motion_gates` would quote on the follow-up
  write. No DDS is touched, no `unitree_sdk2py` submodule loads at
  import (the same hygiene rule every other file in the package
  carries). Refs #358.
