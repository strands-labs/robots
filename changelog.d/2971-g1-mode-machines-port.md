### Feature

- Added `strands_robots.tools.g1.g1_list_arm_ready_mode_machines` and
  `strands_robots.tools.g1.g1_mode_machine_admits_arm`: pure-reference
  agent-facing lookups over the `mode_machine` ids the neon bundle
  observed as arm-ready on the real robot. Snapshotted from the neon
  bundle's `ARM_READY_MODE_MACHINES` observation
  (`cagataycali/neon-the-g1/tools/_g1_common.py`, set `{5, 6}` — the
  two hardware-layout ids the firmware publishes on `rt/lowstate`
  when the balance controller admits an arm write). This
  repository's driver does **not** consult that membership: its
  `_check_motion_gates` decides admission on `_fsm_id` (from the
  motion-switcher API) and reads `mode_machine` only for the
  `is None` liveness refusal, so `admitted` here answers the
  membership question about the neon-observed contract and is not a
  prediction of the driver's admission decision — an arm-ready
  `mode_machine` is necessary-by-observation, not sufficient. A
  future driver-side `mode_machine` fallback would read the same
  set. Each descriptor carries a `mode_machine` id and an
  `admits_arm_writes` flag (always `True`; every listed id is
  arm-ready by construction) so the payload shape matches the
  `g1_fsm_targets` / `g1_arm_actions` / `g1_balance_modes` verbs
  verbatim. Unlike those verbs the refusal path does not carry an
  SDK `rc=` code, and a refused query lands on one of two distinct
  channels so the remedy the text implies is always one that can
  work: a `mode_machine=None` query surfaces the driver's local
  liveness string (`"mode_machine unknown - lowstate has not
  delivered yet"`, remedy: wait for `rt/lowstate`), while a
  delivered-but-non-arm-ready query surfaces a membership refusal
  naming the queried value and the set it must reach (e.g.
  `"mode_machine 0 is not arm-ready; needs one of [5, 6]"`, remedy:
  reach one of those ids). No DDS is touched, no `unitree_sdk2py` submodule loads
  at import (the same SDK-load-hygiene rule every other file under
  `strands_robots.tools.g1` carries, refs strands-labs/robots#358);
  the verbs answer the arm-ready membership question that
  complements the SDK-side FSM lookup already shipped in
  `g1_fsm_targets`, so a caller reading the driver's
  `get_status` envelope can resolve its live `mode_machine` against
  the neon-observed set before dispatching a `send_action`.
  Contract-graded off the module's own snapshot (14 tests: import
  hygiene, snapshot value and typing, refusal string, list-verb
  envelope shape, fresh-container guarantee, admits-a-ready-value on
  both `5` and `6`, refuses-a-non-ready-value, liveness-vs-membership
  refusal channels stay distinct, `None` liveness query, default
  query, `bool` refusal on both truth values, non-`int` refusal). No wire touches (no live `unitree_sdk2py`, no
  DDS bus, no driver instance); pins the neon-observed contract as a
  module-level snapshot, refs strands-labs/robots#358.
