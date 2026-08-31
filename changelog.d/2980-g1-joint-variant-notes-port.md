### Feature

- Added `strands_robots.tools.g1.g1_list_joint_variant_notes` and
  `strands_robots.tools.g1.g1_joint_variant_note`: pure-reference
  agent-facing lookups over the per-slot variant caveats the neon
  bundle observed against the physical Unitree G1 builds
  (`cagataycali/neon-the-g1/tools/g1_joints.py::INVALID_NOTES`).
  Six of the driver's 29 motor-map slots are physically absent on at
  least one narrower G1 variant (waist roll/pitch on the `23dof` and
  the `29dof-with-waist-locked` builds; wrist pitch/yaw on the
  `23dof` build); the driver's `send_action` admits every name in
  the 29-slot map regardless of the physical build, so a caller
  pointing at a missing DoF sees a firmware refusal at wire time
  rather than a name-error at admission. This port surfaces the
  six-row observation table as an agent-facing snapshot so a caller
  planning a rollout can decide the wire refusal decidably before
  `send_action` is attempted. The list verb returns each caveat's
  slot index, `has_note=True` flag, note text (verbatim from the
  neon bundle), and the covered/uncovered slot counts (`29` /
  `23`); the admits verb answers on one slot at a time and refuses
  a `bool`, non-`int`, or out-of-range slot at admission with
  refusals that cite both `#358` (the SDK-facing gate work these
  lookups sit under) and `#2765` (the still-open build-detection
  question the driver does not yet answer). No DDS is touched, no
  `unitree_sdk2py` submodule loads at import (the same hygiene rule
  `g1_arm_actions`, `g1_fsm_targets`, `g1_motion_gates`, and
  `g1_joints` carry). 17 tests, ruff and mypy clean. Refs #358,
  refs #2765.
