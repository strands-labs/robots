### Added: `robot.tool(policies=[...])` — the developer's policies, run by an agent by name

`robot.tool` is the tool as before; `robot.tool(policies=[ttt])` is the same tool with policy **handles** registered.
The model runs one with `run(policy="ttt", instruction=..., seconds=...)`; with exactly one handle `policy` may be left
out. A handle is named by its `.name`, else a remote policy's deployment, else the keys of a `{name: policy}` mapping;
the tool spec lists the names. Handles are objects the developer built — a `Policy("zenoh://…")` included — so an agent
can drive a remote model without ever typing an address: a model-typed `policy="zenoh://..."` stays refused (and the
refusal now names the registered handles). On a real arm the run still asks `approve=` once and passes the safety chain
on every step. Before, the only way to let an agent run a remote policy was a hand-written tool wrapping `robot.run`.
