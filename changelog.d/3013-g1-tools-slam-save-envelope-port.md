### Added: g1_slam_save_envelope port names the neon SLAM save gate

Ports the neon SLAM runner's ``save_map`` gate
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner.save_map``) into
a read-only agent-facing lookup pair in
``strands_robots.tools.g1.g1_slam_save_envelope``.

- ``g1_list_slam_save_envelope()`` returns the envelope descriptor
  (chunks accumulated floor ``chunks_count_min = 1`` -- the neon
  runner's ``if not chunks: return {"ok": False, ...}`` check) plus the
  module-local refusal text a future driver-side wrapper would surface.
- ``g1_slam_save_admits(chunks_count, name)`` grades both dimensions
  (accumulated chunks count against the floor; name against the
  shape rules the neon runner's ``_safe_map_path`` refuses on -- non-
  string, empty, path-separator, ``..`` traversal, hidden-file leading
  dot) and reports every rule the pair violates; a caller planning a
  save sees the whole shape of the refusal at once rather than the neon
  runner's short-circuiting first-refusal-only reply.

Twin of ``g1_slam_map_liveness_envelope`` (strands-labs/robots#3011) --
that envelope names the *load*-side floor (the neon runner refuses ICP
relocalise on a map with fewer than 100 points), and this envelope
names the *save*-side floor on the same accumulated chunks list.

Import hygiene: the module pulls no ``unitree_sdk2py``, ``numpy``,
``open3d`` or ``kiss_icp`` submodule at load time -- a caller
authoring a save plan before any SLAM extra is installed still gets
the envelope back verbatim.  Refs strands-labs/robots#358.
