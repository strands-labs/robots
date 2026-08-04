### Added: notebook 6 orchestrates a heterogeneous fleet from a single goal

`examples/notebooks/06_fleet_orchestration.ipynb` takes the notebook series from
one robot to several. It brings up two arms and a quadruped in one `Simulation`,
classifies each by capability, decomposes a plain-language goal into per-robot
tasks, dispatches them together through `run_multi_policy`, and re-plans when a
robot goes offline so its work is left `UNASSIGNED` rather than silently dropped
or handed to a robot that cannot do it.

Capabilities are derived from the registry's `category` via `get_robot`, not
listed per robot name, so a robot the registry already knows is classified
without editing the notebook - `go2` resolves through its alias to
`unitree_go2`/`mobile`. The registry carries no capability field of its own, so
the category-to-capability mapping is orchestration policy and the notebook says
so rather than implying the registry supplies it. An unmapped category raises
with the canonical name and category in the message instead of defaulting to a
capability set the robot may not have.

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
