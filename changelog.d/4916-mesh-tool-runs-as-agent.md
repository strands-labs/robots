### Fixed: the `use_mesh` agent tool runs as an agent, so a model cannot point a run at a network address

`use_mesh`'s `run` action skipped the agent mark that every other agent tool sets. A model could therefore start a
run with `policy="zenoh://<host>:<port>/<deployment>"`, which streamed the robot's observations to an address the
model chose, and the operator was never asked.

- Every `use_mesh` call now runs in `from_agent()`.
- Its `run` action refuses any `://` policy itself.
- A run on a mesh device (always treated as a real robot) needs the operator's answer:
  `use_mesh(mesh, approve=<callable(RunPlan) -> bool>)`. Without it the run is refused when it is asked for,
  rather than from the run's thread after it reported "started".
- `mesh.tool` has no `approve`, so it refuses agent runs.
