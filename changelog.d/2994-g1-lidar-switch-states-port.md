### Feature

- Added `strands_robots.tools.g1.g1_list_lidar_switch_states` and
  `strands_robots.tools.g1.g1_lidar_switch_state_admits`, two
  read-only ``@tool`` verbs that snapshot the two ASCII wire literals
  the ``rt/utlidar/switch`` DDS topic admits as writes: ``"ON"``
  (powers the head-mounted Livox Mid-360 on; the LiDAR then begins
  publishing ``rt/utlidar/cloud_livox_mid360`` point-cloud frames
  within one to two seconds and reports the active operating mode on
  ``rt/utlidar/lidar_state``) and ``"OFF"`` (powers the LiDAR off;
  the point-cloud publisher stops within one publish tick). The
  tuple order matches the neon bundle's ``g1_lidar_switch``
  (``cagataycali/neon-the-g1/tools/g1_lidar.py``) truth table
  byte-for-byte: the neon verb defaults ``on=True`` which selects
  ``"ON"`` on the argumentless-default branch, so preserving the
  ordering pins which literal a future driver-side wrapper's
  argumentless call would select. Each admitted literal carries a
  ``role`` label (``power_on`` / ``power_off``), a ``description``
  naming the observable effect the write has, and an ``is_default``
  flag naming the ``"ON"`` entry the neon verb selects with no
  arguments. The two verbs let a caller decide the ``rc=7404``
  gate-refused refusal decidably before a future driver-side
  ``g1_lidar_switch`` wrapper reaches the wire, so a caller reading
  the admitted set can enumerate the two literals without pulling
  the DDS binding itself. No ``unitree_sdk2py`` submodule and no
  ``cyclonedds`` binding load on import - the switch-state snapshot
  is an immutable string tuple, matching the SDK-load hygiene the
  rest of this package carries and letting a machine without
  ``cyclonedds`` installed read the admitted set. Non-string
  (including ``bool``), empty-string, case-variant (``"on"``,
  ``"On"``, ``" ON"``), and missing arguments to
  ``g1_lidar_switch_state_admits`` refuse decidably with reason
  strings that name the shape error rather than coercing the case
  and letting the LiDAR firmware silently drop the write at the
  wire (the firmware admits ASCII uppercase only and drops any
  other ``data`` string without a refusal). Refs
  strands-labs/robots#358.
