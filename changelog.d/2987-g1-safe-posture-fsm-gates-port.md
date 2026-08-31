### Added: agent-facing lookup for the neon safe-posture FSM whitelists

The neon bundle's ``g1_safe_posture.py``
(``cagataycali/neon-the-g1/tools/g1_safe_posture.py``) refuses the
``LocoClient.Damp()`` preamble every one of its three compound-posture
verbs issues (``g1_safe_squat_to_stand`` / ``g1_safe_lie_to_stand`` /
``g1_safe_stand_to_squat``) unless the driver's live FSM sits inside a
verb-specific whitelist -- ``{3, 4, 706}``, ``{1, 702}``, and
``{500, 501, 801}`` respectively. Damping a robot outside those sets
leaves the joints without an active controller and collapses the pose,
so the gate is a hard safety precondition rather than a stylistic
check.

``strands_robots.tools.g1.g1_safe_posture_fsm_gates`` snapshots the
three whitelists as a module-level ``dict[str, frozenset[int]]`` and
surfaces them as two agent-facing verbs -- ``g1_list_safe_posture_fsm_gates``
returns the whole envelope, ``g1_safe_posture_fsm_admits`` decides one
membership query -- so a caller planning a compound-posture rollout can
name the precondition decidably before any Damp-preamble verb lands on
the driver side. The verb pair pulls no ``unitree_sdk2py`` submodule at
import time (the SDK-load-hygiene contract every other file in the
package carries, refs strands-labs/robots#358), and every FSM id in
every whitelist is cross-checked against
``strands_robots.tools.g1.g1_fsm_targets._FSM_NAME_MAP`` -- the
SDK-admitted transition-target set -- so a neon-side widen that named
an FSM id the SDK does not admit would surface at CI rather than at
wire time.

Mirrors the pattern already merged for ``g1_motion_gates``,
``g1_fsm_targets``, and ``g1_dds_topics``: one snapshot per
neon/SDK-facing table, one verb pair per snapshot. Refs
strands-labs/robots#358, follows the merged strands-labs/robots#2916
which wired the driver-side FSM gate every safe-posture actuation
path must eventually route through.
