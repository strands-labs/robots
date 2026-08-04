### Added: notebook 6 orchestrates a heterogeneous fleet from a single goal

`examples/notebooks/06_fleet_orchestration.ipynb` takes the notebook series from
one robot to several. It brings up three arms and a quadruped in one
`Simulation`, reads the capability tags each registry entry exposes, decomposes a
plain-language goal into per-robot tasks, dispatches them together through
`run_multi_policy`, and re-plans when a robot goes offline so its work is left
`UNASSIGNED` rather than silently dropped or handed to a robot that cannot do
it.

Decomposition runs rule-based by default so the notebook needs no credentials;
an optional cell swaps in a Strands agent planner and falls back to the
rule-based plan when no model is reachable. Every robot runs a `MockPolicy`, so
the whole notebook executes on a laptop with no model weights and no hardware -
on hardware the dispatch code is unchanged.

`MUJOCO_GL` is defaulted the way the rest of the series defaults it,
`"cgl" if sys.platform == "darwin" else "egl"`. The first draft hard-coded
`"cgl"`, which is macOS-only and would have broken the notebook on headless
Linux; `tests/test_examples_mujoco_gl.py` covers notebooks as well as tracked
`.py` files and fails on the unguarded form.

Two limits the notebook states rather than hides. `run_multi_policy` records one
task per frame, so it warns and stores a single instruction when the robots are
given different ones - the dispatch is still per-robot, only the recorded task
column is shared. And it takes no `stop_when`, which `run_policy` does have, so
the rollout runs a fixed `n_steps` instead of returning on a semantic condition.
