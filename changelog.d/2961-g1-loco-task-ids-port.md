### Feature

- Added `strands_robots.tools.g1.g1_list_loco_tasks` and
  `strands_robots.tools.g1.g1_loco_task_admits`: pure-reference
  agent-facing lookups over the task ids
  `unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.SetTaskId` admits
  as dispatch targets, so a caller can decide the SDK's `rc=7303`
  refusal decidably before a future driver-side wrapper for
  `SetTaskId` is called. Snapshotted from the neon bundle's
  `g1_set_task_id` docstring (4 tasks observed against the real
  robot: `0` WaveHand-no-turn, `1` WaveHand-with-turn,
  `2` ShakeHand-reach, `3` ShakeHand-shake). Each descriptor also
  carries a `sequenced` flag naming the two stage-only ids
  (`2`, `3`) and an `admits_loco_writes` flag (always `True` here,
  surfaced for shape parity with `g1_fsm_targets`) so a caller
  planning a dispatch sees the follow-up write path's admission on
  the same call. No DDS is touched, no `unitree_sdk2py` submodule
  loads at import (the same hygiene rule `g1_arm_actions`,
  `g1_motion_gates`, `g1_joints`, and `g1_fsm_targets` carry).
  Refs #358.
