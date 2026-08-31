### Feature

- Added `strands_robots.tools.g1.g1_list_dangerous_publish_topics` and
  `strands_robots.tools.g1.g1_dangerous_publish_topic_admits`:
  pure-reference agent-facing lookups over the five DDS write topics
  the neon bundle's `_dds_engine.DANGEROUS_PUB_TOPICS` set marks as
  known-dangerous publish paths (`rt/lowcmd`, `rt/armsdk`,
  `rt/user_lowcmd`, `rt/inspire/cmd`, `rt/bmscmd`), so a caller
  planning a generic DDS publish can decide the caller-side refusal
  decidably before a future driver-side publish wrapper actually
  fires. Snapshotted from the neon bundle's `_dds_engine.py`
  (`cagataycali/neon-the-g1/tools/_dds_engine.py`) whose
  `g1_dds_publish` verb refuses membership on the set unless the
  caller passes an explicit `unsafe=True` override. Each descriptor
  carries a `description` label (the neon bundle's own topic-catalog
  description with the `U+1F6A8` marker stripped so the string domain is
  plain text) plus a `dangerous_to_publish` flag (always `True` on
  membership - the set itself is the danger contract, surfaced for
  shape parity with the neon bundle's own `list_topics` payload).
  Both verbs surface a `refusal_advice` string that names the
  `unsafe=True` override argument verbatim so a caller reading the
  refusal sees the same knob the caller-side verb exposes. The
  `rt/lowcmd` entry is byte-for-byte identical to the driver's own
  write topic in `strands_robots.tools.g1.g1_dds_topics`; a widen on
  one side must update both together or the refusal on one surface
  and the same-named topic on the other silently diverge. No DDS is
  touched, no `unitree_sdk2py` submodule loads at import (the same
  hygiene rule every other file in the package carries). Refs #358.
