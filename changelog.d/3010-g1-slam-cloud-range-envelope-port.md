### Added

- `strands_robots.tools.g1.g1_slam_cloud_range_envelope` ports the read-only
  half of neon-the-g1's SLAM kiss-icp frame preprocessor
  (`cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._make_odometry`):
  two agent-facing verbs (`g1_list_slam_cloud_range_envelope`,
  `g1_slam_cloud_range_admits`) surface the neon runner's ``KISSConfig``
  per-point radial band — ``data.min_range = 1.0`` m (near-clip) and
  ``data.max_range = 40.0`` m (far-clip). A point whose Euclidean
  distance from the sensor origin sits outside ``[1.0, 40.0]`` is
  dropped by kiss-icp's frame preprocessor before the ICP registration
  sees the cloud. The single refusal reason
  ``range_outside_kiss_icp_preprocessor`` names both the below-min
  and above-max cases; the comparison descriptor (``"value < bound"``,
  ``"value > bound"``, ``"non-finite"``) distinguishes which bound the
  value violated. Non-finite input (``math.inf``, ``math.nan``,
  ``-math.inf``) refuses with ``comparison="non-finite"``. Twin of
  `g1_slam_icp_fitness_envelope` (`cagataycali/robots#3008`, the
  *registration-quality* dimension on the same SLAM surface — the
  fitness scalar the ICP returns after the frame has already been
  range-filtered) and `g1_slam_relocalize_envelope`
  (`cagataycali/robots#3006`, the ICP *voxel-size* dimension on the
  map-side relocalise path). Read-only. No driver instance, no DDS,
  no SDK, no ``kiss_icp`` submodule import at load time (refs
  strands-labs/robots#358).
