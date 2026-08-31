### Added: g1_slam_frame_queue_envelope port names the neon SLAM producer-consumer capacity

Ports the neon SLAM runner's ``_frame_q`` single-slot policy
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner`` constructs
``queue.Queue(maxsize=1)`` at ``__init__`` and the producer-side
``_on_cloud`` swallows ``queue.Full`` on ``put_nowait``) into a
read-only agent-facing lookup pair in
``strands_robots.tools.g1.g1_slam_frame_queue_envelope``. Twin of the
merged ``g1_slam_pose_history_envelope`` (strands-labs/robots#3026)
on the same ``_SlamRunner`` surface; that envelope names the
``_process_frame`` bookkeeping ceiling on the pose trail, this envelope
names the ``_on_cloud`` producer-side queue capacity on the runner's
handoff to the ICP worker. Two distinct code paths, two distinct
remedies: pose-history points a planner at "shorten the session or
buffer to disk", this one points at "accept the freshness-preferring
drop policy or throttle upstream of the runner".

- ``g1_list_slam_frame_queue_envelope()`` returns the envelope
  descriptor (``frame_queue_max = 1``) plus the module-local refusal
  text a future driver-side wrapper would surface on an above-ceiling
  capacity argument.
- ``g1_slam_frame_queue_admits(queue_capacity)`` reuses the shared
  ``positive_count_error`` validator to refuse bool / non-int /
  value-below-one shape mistakes decidably before the runner-observed
  ceiling is asked, then grades the single-slot ceiling on top;
  refusals cite the module-local text rather than borrowing the
  motion-FSM ``7404`` rc (which would name a locomotion-FSM remedy for
  a producer-consumer capacity argument).

Import hygiene: the module pulls no ``unitree_sdk2py``, ``numpy``,
``open3d``, ``kiss_icp``, or stdlib ``queue`` submodule at load time --
a caller authoring a SLAM plan before any SLAM extra is installed
still gets the ceiling back verbatim. Refs strands-labs/robots#358.
