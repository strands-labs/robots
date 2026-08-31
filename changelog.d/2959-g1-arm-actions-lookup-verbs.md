### Feature

- Added `strands_robots.tools.g1.g1_list_arm_actions` and
  `strands_robots.tools.g1.g1_arm_action_admits`: pure-reference
  agent-facing lookups over
  `unitree_sdk2py.g1.arm.g1_arm_action_client.action_map`, so a caller
  can decide the SDK's `rc=7402` refusal decidably before a future
  driver-side wrapper for `ExecuteAction` is called. Snapshotted from
  the SDK today (16 gestures including `release arm=99`); no DDS is
  touched, no `unitree_sdk2py` submodule loads at import (the same
  hygiene rule `g1_motion_gates` and `g1_joints` carry). Refs #358.
