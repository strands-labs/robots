### Fixed: the safety surface names things the way a person reads them

The `RunPlan` an operator approves names the policy (`policy='sway'`, or the handle an agent used, `policy='wave'`)
instead of `<function sway at 0x...>`. An e-stop refusal says who pressed it (`e-stop is engaged by alice
(estop())`). A `Robot` from `Mesh.connect()` reprs as `remote on <tenant>`, not `hardware`. `from_agent` is a name on
`strands_robots.robot`.
