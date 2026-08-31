### Feature

- Added `strands_robots.tools.g1.g1_list_dds_topic_idl_types` and
  `strands_robots.tools.g1.g1_topic_idl_type`, two read-only ``@tool``
  verbs that snapshot the ``(idl_module, idl_class)`` string pair
  `G1Driver` hands ``ChannelSubscriber`` (or ``ChannelPublisher`` on
  the write side) at ``connect_eagerly`` time for each of its seven
  topics. The six read entries mirror the driver's own
  ``_subscription_plan`` return values byte-for-byte
  (``rt/lowstate`` → ``LowState_``, ``rt/lf/bmsstate`` → ``BmsState_``,
  ``rt/utlidar/lidar_state`` → ``LidarState_``,
  ``rt/utlidar/cloud_livox_mid360`` → ``PointCloud2_``,
  ``rt/mainboardstate`` → ``MainBoardState_``,
  ``rt/pressuresensorstate`` → ``PressSensorState_``); the write entry
  names the ``LowCmd_`` type the driver's own ``rt/lowcmd``
  ``ChannelPublisher`` is constructed with. A caller planning a
  ``g1_dds_snapshot``-shaped subscribe (the neon bundle's
  ``cagataycali/neon-the-g1/tools/g1_dds.py`` verb) against a driver
  topic uses this pair to name the same IDL type the driver already
  holds so the two subscribes decode the wire bytes identically;
  a caller planning a mesh publish on the write topic uses it to
  see which IDL type the driver's own publisher wants. The IDL
  modules are named as strings rather than imported so
  ``import strands_robots.tools.g1.g1_dds_topic_idl_types`` pulls no
  ``unitree_sdk2py`` submodule and the verbs stay loadable on a
  headless CI runner or on Thor before an office bring-up (a caller
  who wants the actual class object calls
  ``importlib.import_module(idl_module)`` on the returned string). A
  new contract test walks ``G1Driver._subscription_plan()`` and
  asserts every ``(idl_module, idl_class)`` pair matches the tool's
  snapshot, so a driver-side widen or narrow surfaces here as a diff
  rather than as a silent decode-shape disagreement. Non-string,
  ``bool``, empty-string, and off-set arguments to
  ``g1_topic_idl_type`` refuse decidably with reason strings that
  name the shape error rather than falling through to a confusing
  "unknown topic" refusal, and the off-set refusal carries the
  driver's known-topic list so the caller can retry without a
  second lookup. Refs strands-labs/robots#358.
