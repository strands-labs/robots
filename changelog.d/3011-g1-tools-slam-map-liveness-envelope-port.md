### Added: g1_slam_map_liveness_envelope port names the neon SLAM relocalise precondition

Ports the neon SLAM runner's ``_try_relocalize`` precondition
(``cagataycali/neon-the-g1/tools/g1_slam.py`` reads
``if _o3d is None or map_pts is None or len(map_pts) < 100: return None``)
into a read-only agent-facing lookup pair in
``strands_robots.tools.g1.g1_slam_map_liveness_envelope``. Twin of the
merged ``g1_slam_relocalize_envelope`` (strands-labs/robots#3006), which
names the match-quality dimensions on the same gate; this module names
the point-count precondition the runner refuses before any ICP dispatch.

- ``g1_list_slam_map_liveness_envelope()`` returns the envelope descriptor
  (``point_count_min = 100``) plus the module-local refusal text a future
  driver-side wrapper would surface on a below-floor map.
- ``g1_slam_map_liveness_admits(point_count)`` reuses the shared
  ``positive_count_error`` validator to refuse bool / non-int / value-below-one
  shape mistakes decidably before the runner-observed floor is asked, then
  grades the map-cardinality floor on top; refusals cite the module-local text
  rather than borrowing the motion-FSM ``7404`` rc (which would name a
  locomotion-FSM remedy for a map-cardinality argument).

Import hygiene: the module pulls no ``unitree_sdk2py``, ``open3d`` or
``kiss_icp`` submodule at load time -- a caller authoring a relocalise plan
before any SLAM extra is installed still gets the floor back verbatim.
Refs strands-labs/robots#358.
