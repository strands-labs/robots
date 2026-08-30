### Added: `g1_slam_voxel_dedup_envelope` lookup pair reads the neon runner's map compaction grid

The neon SLAM bundle's `_SlamRunner._process_frame` stitches every
100 accumulated LiDAR chunks into a single XYZI array and passes
that array through the module-level `_voxel_dedup` helper, which
reads a single authored constant (`_VOXEL_DEDUP_SIZE = 0.05`) and
collapses every point landing in the same 5 cm grid cell into
one.  The neon runner reads this constant inline; a caller who
wants a different edge today must patch the neon module.  A
future driver-side SLAM accumulation wrapper that parameterises
the edge will land on this envelope's shape grader.

Two agent-facing verbs snapshot the observation and grade a
caller-supplied proposal.  `g1_list_slam_voxel_dedup_envelope`
returns the neon-observed value verbatim
(`voxel_dedup_neon_default_m = 0.05`) and carries the
module-local refusal text a driver-side wrapper would surface on
a shape violation.  `g1_slam_voxel_dedup_admits(voxel_edge_m)`
routes the caller's argument through the shared
`positive_finite_number_error` domain and reports a shape refusal
(a `bool`, a non-`numbers.Real` type, `nan`/`inf`, a value
`<= 0`, or a value past the float64 range) with the module-local
text alongside the shared-domain error message.

The envelope is captured as module-level constants so the import
pulls no `unitree_sdk2py` submodules and no `numpy`, `open3d`,
or `kiss_icp` submodules; a caller authoring a dedup plan
before any SLAM extra is installed on their host still gets the
envelope back verbatim.  The refusal text names the SLAM
voxel-dedup surface rather than borrowing the motion-FSM `7404`
rc so an agent planner reading the refusal reaches a remedy on
the write path the constant belongs to.  Refs
strands-labs/robots#358.
