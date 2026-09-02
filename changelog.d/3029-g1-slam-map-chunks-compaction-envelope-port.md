### Added

- `strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope` --
  agent-facing lookup pair for the chunk-list compaction trigger the
  neon SLAM runner fires at.  The neon bundle
  (`cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._process_frame`)
  reads `if len(self._map_chunks) > 100:` and steals the chunk list
  under its own lock on strict-greater, stitches every chunk into a
  single `(N, 4)` XYZI array, calls the module-level `_voxel_dedup`
  (the 5 cm grid the merged strands-labs/robots#3014 names), and
  installs the deduped result back as the sole chunk.  The value is
  authored inline in the neon runner rather than routed as a caller
  argument, so every neon SLAM host on the same runner build fires
  the compaction at exactly 100 chunks.

  The lookup pair carries `g1_list_slam_map_chunks_compaction_envelope`
  (returns the neon-observed value and a module-local refusal string)
  and `g1_slam_map_chunks_compaction_admits` (grades a caller-supplied
  batch depth against the ceiling and against the shared
  `positive_count_error` domain).  A caller planning a long-running
  SLAM accumulation session reads the envelope before a future
  driver-side wrapper fires, decides the batch-depth question
  decidably, and sees the module-local refusal text a future write
  verb would surface on an over-ceiling batch.  Refs
  strands-labs/robots#358.

  Twin of `g1_slam_pose_history_envelope` (the merged
  strands-labs/robots#3026) and `g1_slam_frame_queue_envelope`
  (strands-labs/robots#3027, in flight) -- all three port a distinct
  in-memory ceiling the same neon `_SlamRunner` reads on the same
  `_process_frame` code path, and each names a different remedy on a
  different surface.  Also a twin of the same-surface dedup-cell-size
  envelope `g1_slam_voxel_dedup_envelope` (the merged
  strands-labs/robots#3014): that names the 5 cm *cell size* the
  compaction reads at each fire; this names the 100-chunk *trigger
  count* that decides when the compaction fires at all.  A widen to
  one would not co-widen the other, so keeping them on separate
  surfaces keeps each widen local.

  Import hygiene: `import strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope`
  pulls no `unitree_sdk2py`, `numpy`, `open3d`, `kiss_icp`, or stdlib
  `queue` submodule at load time -- the contract every other file in
  `strands_robots/tools/g1/` carries.  Asserted in the test module's
  `TestTheImportPullsNoOptionalSlamModule` cell (three separate
  assertions).
