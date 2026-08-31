### Feature

- Added `strands_robots.tools.g1.g1_list_dds_topic_categories` and
  `strands_robots.tools.g1.g1_dds_topic_category_admits`:
  pure-reference agent-facing lookups over the seven category labels
  the neon bundle's `_dds_engine.TOPIC_CATALOG` partitions its 22
  catalog topics into (`state`, `lidar`, `joystick`, `control`,
  `hand`, `slam`, `config`), so a caller planning a category-scoped
  catalog read can name the neon `g1_dds_list_topics` filter argument
  decidably before a future driver-side wrapper for that verb lands.
  Snapshotted from the neon bundle's `_dds_engine.py`
  (`cagataycali/neon-the-g1/tools/_dds_engine.py`) whose
  `g1_dds_list_topics` verb takes a `category` argument and filters
  its returned per-topic descriptors against the same label set.
  Each descriptor carries a plain-text `description` naming the wire
  intent the category label partitions the neon catalog by (read
  side vs write side, joint state vs SLAM channel, etc.) plus a
  `topic_count` field surfacing how many neon-catalog topics carry
  the label today (9 `state`, 4 `control`, 3 `slam`, 2 `lidar`, 2
  `hand`, 1 `joystick`, 1 `config`; total 22 matches the neon
  `TOPIC_CATALOG` size). The `admits` verb refuses shape errors
  (`bool`, non-str, empty string) decidably before the membership
  test runs, and its off-partition refusal lists every known label
  so a caller can resolve the drift without a follow-up call. The
  `control` category count is pinned at 4 to catch the drift with
  the sibling `strands_robots.tools.g1.g1_dangerous_publish_topics`
  snapshot (5 dangerous topics, of which 4 are `control` and 1 is
  `hand` - the Inspire hand cmd). No DDS is touched, no
  `unitree_sdk2py` submodule loads at import (the same hygiene rule
  every other file in the package carries). Refs #358.
