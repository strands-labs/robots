### Feature

- Added `strands_robots.tools.g1.g1_list_dds_topics` and
  `strands_robots.tools.g1.g1_topic_role`, two read-only ``@tool`` verbs
  that snapshot the seven CycloneDDS topics
  :class:`~strands_robots.drivers.g1.G1Driver` opens on the bus: the six
  ``"read"``-side subscribes it makes at ``connect_eagerly`` time
  (``rt/lowstate``, ``rt/lf/bmsstate``, ``rt/utlidar/lidar_state``,
  ``rt/utlidar/cloud_livox_mid360``, ``rt/mainboardstate``,
  ``rt/pressuresensorstate``) and the single ``"write"``-side publish
  (``rt/lowcmd``) it fronts through ``send_action`` / ``run_policy``.
  Each snapshot entry carries the topic string, the wire ``direction``
  and a ``role`` matching the driver's cache attribute (``lowstate`` ->
  ``_imu`` + ``_mode_machine``, ``battery`` -> ``_battery``,
  ``lidar_state`` -> ``_lidar_state``, ``lidar_cloud`` ->
  ``_lidar_summary``, ``mainboard`` -> ``_mainboard``, ``pressure`` ->
  ``_pressure``, ``lowcmd`` -> the driver's ``_pubs`` write path).
  ``g1_list_dds_topics`` returns the full seven-entry table or, when
  ``direction`` is ``"read"`` / ``"write"``, only the matching side;
  ``g1_topic_role`` decides one topic-string membership by returning
  the driver's own decode label on admit or refusing an unknown topic
  string with the known set quoted verbatim (DDS topic names are exact
  strings on the wire, so a mis-cased ``rt/LowState`` refuses rather
  than resolving to ``rt/lowstate``). A test in
  ``tests/drivers/test_g1_dds_topics_reads_the_driver_subscription_set.py``
  reads the driver's own ``_TOPIC_*`` module-level constants and
  asserts identity with the tool module's snapshot, so a driver-side
  widen or narrow of the subscription set surfaces as a diff rather
  than as a diverging table the author would have to keep in sync by
  hand. No ``unitree_sdk2py`` submodule loads on import - the topic
  strings are a module-level snapshot, matching the SDK-load hygiene
  the rest of this package carries. Sourced from the neon bundle's
  ``g1_dds.py`` (``cagataycali/neon-the-g1/tools/g1_dds.py``), where a
  ``g1_dds_snapshot``-shaped verb wrapped a second CycloneDDS reader
  against arbitrary topics; this lookup ports the read-only lookup
  half - which topics the driver's own reader path already carries -
  without also introducing a second reader path the driver does not
  yet front. Refs strands-labs/robots#358.
