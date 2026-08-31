### Feature

- Added `strands_robots.tools.g1.g1_list_bms_state_topics` and
  `strands_robots.tools.g1.g1_bms_state_topic_admits`: pure-reference
  agent-facing lookups over the three DDS topic names the neon
  bundle's `_BMS_TOPICS` tuple
  (`cagataycali/neon-the-g1/tools/g1_battery.py`) sweeps when
  subscribing the battery-management state
  (`rt/lf/bmsstate` first, then two historical spellings
  `rt/bmsstate` and `rt/bms_state`). Useful before a future
  driver-side wrapper for the neon fallback sweep is dispatched, so
  a caller planning a firmware-portable BMS subscribe can name the
  candidate set and neon sweep-order rank decidably. Each descriptor
  carries the topic string, a priority rank matching the neon
  tuple's own index order, a plain-text description of the topic's
  wire lineage, and the shared BMS payload IDL type string
  (`unitree_sdk2py.idl.unitree_hg.msg.dds_.BmsState_`). The
  `rt/lf/bmsstate` entry is the same string
  `strands_robots.tools.g1.g1_dds_topics` names on the driver's own
  battery subscription; the invariant a future firmware rename must
  preserve is byte-for-byte identity between the two files' topic
  strings, so a rename lands on both surfaces or the neon sweep and
  the driver's own subscribe silently diverge. No DDS is touched,
  no `unitree_sdk2py` submodule loads at import (the same hygiene
  rule every other file in the package carries). Refs #358.
