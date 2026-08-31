### Added

- `strands_robots.tools.g1.g1_dds_max_buffer_envelope` ports the read-only
  half of neon-the-g1's `g1_dds_subscribe(max_buffer=...)` deque-maxlen
  envelope (`cagataycali/neon-the-g1/tools/g1_dds.py`,
  `cagataycali/neon-the-g1/tools/_dds_engine.py::_SubHandle.buffer`): two
  agent-facing verbs (`g1_list_dds_max_buffer_envelope`,
  `g1_max_buffer_admits`) surface the neon-observed subscribe-window clamp
  (``[1, 10000]``), plus the neon-tuned default (``20`` for the agent-facing
  read verb's token budget on one `g1_dds_read` payload). Module-local
  refusal text stays on-surface — the DDS subscription handle sits on the
  SDK-owned reader thread in-process and never touches ``rt/lowcmd``, so no
  motion-FSM ``7404`` code is re-borrowed for a buffer bounds violation.
  Twin of `g1_dds_topic_categories` and `g1_dds_topic_idl_types` (the
  *topic* and *IDL type* dimensions on the same subscribe surface;
  disjoint refusal shapes so the modules stay separate). Read-only. No
  driver instance, no DDS, no SDK, no `collections` submodule import at
  load time (refs #358).
