### Feature

- Added `strands_robots.tools.g1.g1_list_damp_transitions` and
  `strands_robots.tools.g1.g1_damp_transition_admits`: pure-reference
  agent-facing lookups over the safe damp-preamble transition
  preconditions the neon bundle's `g1_safe_posture.py` verbs
  (`cagataycali/neon-the-g1/tools/g1_safe_posture.py`) each front, so a
  caller can decide the refusal decidably before a future driver-side
  wrapper for the damp-preamble path is called. Snapshotted from the
  neon bundle's `_assert_safe_for_damp` refusal boundary observed
  against the real robot: three transitions (`squat_to_stand`,
  `lie_to_stand`, `stand_to_squat`) each carrying the FSM set the
  transition is safe from, a `pose_check` flag naming whether the
  `avg_knee > 1.4 rad` refusal is enforced, and the `avg_knee_max_rad`
  threshold when enabled. Each descriptor also carries a `description`
  field naming the neon bundle's one-line semantic, so an agent-facing
  planner has the transition intent on hand without re-reading the
  neon docs. The verbs also surface `walk_ready_fsm_ids` (quoting
  `WALK_FSMS`) and a module-local gate-refusal text built from that
  same set, so a caller comparing an intended write against the
  general locomotion gate has the FSM set on hand too. No refusal
  quotes an SDK return code: the driver's `_check_motion_gates`
  returns free-text refusals and names none, and the two refusals this
  module makes have different remedies (an FSM outside the locomotion
  gate wants a transition; an unknown transition label wants a
  different string), so each carries its own text. No DDS is touched, no `unitree_sdk2py` submodule
  loads at import (the same hygiene rule `g1_balance_modes`,
  `g1_arm_actions`, `g1_fsm_targets`, and `g1_motion_gates` carry).
  Refs #358.
