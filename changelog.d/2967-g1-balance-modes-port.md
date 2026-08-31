### Feature

- Added `strands_robots.tools.g1.g1_list_balance_modes` and
  `strands_robots.tools.g1.g1_balance_mode_admits`: pure-reference
  agent-facing lookups over the balance-mode ids
  `unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.BalanceStand`
  admits, so a caller can decide the refusal decidably before a
  future driver-side wrapper for `BalanceStand` is called.
  Snapshotted from the neon bundle's `g1_balance_stand` verb
  (`cagataycali/neon-the-g1/tools/g1_posture.py`): two modes
  observed against the real robot (`0` Static — the SDK default,
  `3` Dynamic — the higher-headroom option). Each descriptor also
  carries an `admits_loco_writes` flag (always `True`; every
  admitted balance mode is a locomotion-shaped write by
  definition) so the payload shape matches the `g1_fsm_targets`
  and `g1_arm_actions` verbs verbatim. The verbs also surface
  `walk_ready_fsm_ids` (quoting `WALK_FSMS`) and the `7404`
  gate-refusal code so a caller comparing an intended write
  against both conditions has the FSM set on hand. No DDS is
  touched, no `unitree_sdk2py` submodule loads at import (the same
  hygiene rule `g1_arm_actions`, `g1_fsm_targets`, and
  `g1_motion_gates` carry). Refs #358.
