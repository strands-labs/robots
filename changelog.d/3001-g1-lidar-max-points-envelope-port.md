### Added

- `strands_robots.tools.g1.g1_lidar_max_points_envelope` ports the read-only
  half of neon-the-g1's `g1_lidar_snapshot(max_points=...)` downsample envelope
  (`cagataycali/neon-the-g1/tools/g1_lidar.py`): two agent-facing verbs
  (`g1_list_lidar_max_points_envelope`, `g1_max_points_admits`) surface the
  neon-parser's stride-divisor clamp (``[1, 50000]``), plus the two
  neon-observed values on the same argument tuple — the agent-facing default
  (``4000`` "downsample target (stride-based). Default 4000.") and the
  SLAM-feeder internal value (``50000`` in `g1_slam.py` that keeps the ICP
  registration cost bounded on a full Livox MID-360 frame). Module-local
  refusal text stays on-surface — the lidar frame parser runs on the caller's
  Python thread in-process and never touches ``rt/lowcmd``, so no motion-FSM
  ``7404`` code is re-borrowed for a downsample bounds violation. Twin of
  `g1_capture_rate_candidates` (the *rate* dimension on the same lidar
  surface; capture rate is a driver-side subscribe cadence while downsample is
  a caller-side parser argument, so the modules stay separate). Read-only. No
  driver instance, no DDS, no SDK, no ``numpy`` submodule import at load time
  (refs #358).
