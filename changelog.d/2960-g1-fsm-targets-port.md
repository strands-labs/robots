### Feature

- Added `strands_robots.tools.g1.g1_list_fsm_targets` and
  `strands_robots.tools.g1.g1_fsm_target_admits`: pure-reference
  agent-facing lookups over the FSM ids
  `unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.SetFsmId` admits
  as transition targets, so a caller can decide the SDK's `rc=7302`
  refusal decidably before a future driver-side wrapper for
  `SetFsmId` is called. Snapshotted from the neon bundle's
  `FSM_NAMES` table (10 targets: `0` ZeroTorque, `1` Damp, `2`
  Squat, `3` Sit, `4` StandUp, `500` Start, `501` Walk, `702`
  Lie2StandUp, `706` Squat2StandUp, `801` BalanceExpert). Each
  descriptor also carries a `dangerous` flag naming the off-gantry
  targets (`0`, `1`) and `admits_arm_writes` / `admits_loco_writes`
  flags naming the two write-gate sets (`HANDSHAKE_FSMS`,
  `WALK_FSMS`) so a caller planning a transition sees the follow-up
  write path's admission on the same call. No DDS is touched, no
  `unitree_sdk2py` submodule loads at import (the same hygiene
  rule `g1_arm_actions`, `g1_motion_gates`, and `g1_joints`
  carry). Refs #358.
