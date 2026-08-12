### Added: `run_multi_policy` is a base-class contract, with its validation shared across backends

`run_multi_policy` - the synchronized multi-robot driver (per-robot policies
and instructions, one lockstep physics loop, one merged recording frame per
timestep) - was a MuJoCo-only method, so Isaac/Newton callers got a bare
`AttributeError` instead of a contract. It is now documented on
`strands_robots.simulation.base.SimEngine` next to `run_policy`, with a default
implementation that returns the structured not-supported error naming the
backend class (never a silent per-robot fallback, which would interleave
recording frames). The backend-independent first phase of the MuJoCo loop -
empty-`policies` rejection, `instructions` normalization, the
distinct-instructions one-task-per-frame warning, and per-robot
`action_horizon` normalization - moved into shared base helpers
(`_validate_multi_policies`, `_normalize_multi_policy_instructions`,
`_normalize_multi_policy_horizons`) so the upcoming Isaac implementation
(#2122) cannot drift from MuJoCo's refusal texts. MuJoCo behavior is unchanged:
its `run_multi_policy` now calls the shared helpers and keeps the
physics/render/record loop, and every pre-existing test passes unmodified.
