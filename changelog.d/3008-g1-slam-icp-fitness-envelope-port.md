### Added

- `strands_robots.tools.g1.g1_slam_icp_fitness_envelope` ports the read-only
  half of neon-the-g1's `_SlamRunner._do_relocalize` ICP fitness gate
  (`cagataycali/neon-the-g1/tools/g1_slam.py`): two agent-facing verbs
  (`g1_list_slam_icp_fitness_envelope`, `g1_slam_icp_fitness_admits`) surface
  the neon runner's ``0.3`` minimum-admitted ICP fitness plus the ``[0.0,
  1.0]`` shape bounds Open3D's ``result.fitness`` takes on. The neon runner's
  own comparison is ``result.fitness < _ICP_FITNESS_THRESHOLD`` (strict less
  than), so equality at ``0.3`` is admitted — refusing at the boundary would
  drop the neon bundle's own admitted edge. Two refusal reasons are decided:
  ``fitness_below_threshold`` (value inside Open3D's own bounds but below
  the neon admission threshold, the *quality* refusal) and
  ``fitness_outside_open3d_range`` (value the ICP itself does not produce,
  the *shape* refusal). Non-finite input (``math.inf``, ``math.nan``) refuses
  as a shape violation with ``comparison="non-finite"``. The two neon-runner
  transform-space refusals (translation magnitude ``> 50.0`` m and negative
  rotation-trace) are not decided here because they need the ICP output on
  hand; this envelope answers the fitness half only. Twin of
  `g1_slam_relocalize_envelope` (`cagataycali/robots#3006`, the ICP voxel-size
  dimension on the same relocalise surface) and `g1_lidar_max_points_envelope`
  (the downsample cap on the frame the ICP consumes). Read-only. No driver
  instance, no DDS, no SDK, no ``open3d`` submodule import at load time
  (refs strands-labs/robots#358).
