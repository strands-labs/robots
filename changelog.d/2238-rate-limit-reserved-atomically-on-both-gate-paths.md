### Fixed: `robot_mesh` reserves a rate-limit slot atomically on both gate paths

The per-action sliding window bounds LLM-driven actuation, and the early
`_rate_limit_check` deliberately does not consume a slot so a declined
human-in-the-loop approval cannot lock an operator out of a genuine emergency.
The approved path closed the resulting window with an atomic check+record, but
the ungated path - an action outside `STRANDS_MESH_HITL_ACTIONS` - appended a
slot unconditionally. Two concurrent `emergency_stop` calls with one slot left
therefore both passed the check and both recorded: two fleet-wide broadcasts
reached the mesh, the window held four entries for a configured maximum of
three, and both calls reported `status="success"`. That is the configuration
where the cap is the only bound left, since narrowing the gate removes the
operator from the path. Both paths now reserve through the same atomic
check+record and report the refusal through the tool envelope with an audit
record; the record-only primitive that made the racy pattern reachable is
gone, and the race message no longer names an operator approval that the
ungated path never asks for.
