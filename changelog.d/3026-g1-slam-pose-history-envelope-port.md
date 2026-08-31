### Added: g1_slam_pose_history_envelope port names the neon SLAM runner's trail-capacity ceiling

Ports the neon SLAM runner's ``_process_frame`` pose-history
truncate ceiling
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._process_frame``)
into a read-only agent-facing lookup pair in
``strands_robots.tools.g1.g1_slam_pose_history_envelope``.

- ``g1_list_slam_pose_history_envelope()`` returns the envelope
  descriptor (``pose_history_max = 2000``) plus the module-local
  refusal text a future driver-side wrapper would surface.
- ``g1_slam_pose_history_admits(session_length)`` grades the
  session-length dimension against the ceiling and reports the
  bound the argument violates; a caller planning a long-running
  SLAM session sees the ceiling and can shorten the session,
  buffer to disk on its own timer, or accept the neon runner's
  soft-truncate-to-tail semantics.

Import hygiene: the module pulls no ``unitree_sdk2py``, ``numpy``,
``open3d`` or ``kiss_icp`` submodule at load time -- a caller
authoring a SLAM plan before any SLAM extra is installed still
gets the ceiling back verbatim.  Twin of
``strands_robots.tools.g1.g1_slam_relocalize_envelope`` and
``strands_robots.tools.g1.g1_slam_map_liveness_envelope``: those
name ``_try_relocalize``'s match-quality and precondition halves
against a candidate map, this one names ``_process_frame``'s
bookkeeping capacity against the runner's own trail.  Refs
strands-labs/robots#358.
