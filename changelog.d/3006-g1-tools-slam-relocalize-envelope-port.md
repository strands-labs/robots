### Added: g1_slam_relocalize_envelope port names the neon ICP-relocalise gate

Ports the neon SLAM runner's ICP-relocalise gate
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._try_relocalize``) into a
read-only agent-facing lookup pair in ``strands_robots.tools.g1.g1_slam_relocalize_envelope``.

- ``g1_list_slam_relocalize_envelope()`` returns the envelope descriptor
  (fitness floor 0.3, fitness ceiling 1.0, translation magnitude ceiling 50 m,
  rotation trace floor 0.0, rotation trace ceiling 3.0) plus the module-local
  refusal text a future driver-side wrapper would surface.
- ``g1_slam_relocalize_admits(fitness, translation_m, rotation_trace)`` grades
  all three dimensions and reports every bound the triple violates; a caller
  planning a relocalise sees the whole shape of the refusal at once rather than
  the neon runner's short-circuiting first-refusal-only reply.

Import hygiene: the module pulls no ``unitree_sdk2py``, ``numpy``, ``open3d`` or
``kiss_icp`` submodule at load time -- a caller authoring a relocalise plan
before any SLAM extra is installed still gets the envelope back verbatim.
Refs strands-labs/robots#358.
