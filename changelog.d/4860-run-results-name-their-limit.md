### Fixed: a run limited by `seconds=` reports `stopped_by="seconds"`; a splice session keeps `policy.calls`

`robot.run(policy, seconds=2)` and `sim.run({...}, seconds=2)` now end with `stopped_by="seconds"`, the limit the caller
gave; `steps=` still reports `"steps"`. Before, both reported `"steps"`, because seconds were converted to steps.
A remote policy under `execution="splice"` now fills `policy.calls` like a sequential session does (`wall_s`, `rows`,
`server`, `instruction` per chunk), so the two executions' turnaround can be compared from strands. Before, the list
stayed empty under splice.
