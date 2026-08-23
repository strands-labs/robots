### Added: `gr00t_inference(..., deterministic=True)` - curated mount of the GR00T determinism wrapper

The `gr00t_inference` container lifecycle grows an operator-facing
`deterministic: bool = False` flag that mounts the in-repo determinism
wrapper (read-only, fixed container path, library-resolved source) and runs
the N1.7 server through it instead of the bare `run_gr00t_server`
entrypoint. Server-side per-episode reseeding
(`cudnn.deterministic=True`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, a
`Gr00tPolicy.reset` patch that applies the client-forwarded seed) now works
without hand-written `docker -v` plumbing, so CI can pin a GR00T
`success_rate` through the one-call lifecycle orchestration.

The tool's volume lockdown holds: the flag toggles between two
operator-blessed container configurations - it does not open a volume
parameter to the agent, and the guard rejects everything it rejected
before. `deterministic=False` (default) is byte-identical to the previous
behavior. The wrapper itself was promoted from `examples/libero/` into the
package (`strands_robots/policies/groot/server_wrapper.py`, shipped in the
wheel); the example path keeps a thin re-export. `run.py mujoco --policy
groot` grows a `--deterministic` passthrough, and the
`STRANDS_GR00T_SERVER_SEED` / `STRANDS_GR00T_STRICT_DETERMINISTIC` env vars
are forwarded into the container and documented in the README env-var
table.
