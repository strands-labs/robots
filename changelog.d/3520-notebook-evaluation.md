### Added: the notebook series can score the policy it just trained

`examples/notebooks/` taught the whole data loop and never once evaluated it.
Notebooks 1 to 6 build a robot, record a dataset, train an ACT policy, export it,
load it back, and orchestrate a heterogeneous fleet.
Measured across all six, references to `evaluate_benchmark`, `eval_policy`,
`success_rate` and `list_benchmarks`: **zero**.

The gap is sharpest in notebook 3, whose own cell headings read *record a small
dataset -> train -> load the trained checkpoint back -> where to go from here*. It
produces a checkpoint and stops, so a reader finishes the series holding a trained
policy and no way to tell whether it is better than the untrained one. The library
has had that answer since `evaluate_benchmark` shipped, and
`examples/10_evaluate_benchmark.py` demonstrates it as a script; the notebook
series just never reached for it.

`07_evaluate_a_policy.ipynb` closes it, CPU-only like the rest of the series. It
scores the `mock` policy first and says why - mock emits sinusoids and does not
walk, so its score is the **floor**, and without a floor a task that is trivially
satisfiable is indistinguishable from a policy that works. It then reads the
per-attempt rows rather than the average, because `success_rate` says something is
wrong and never says what, and each row carries the seed it ran under so a single
attempt is replayable on its own. `success` and `failure` are printed as separate
fields, not as opposites: the row where both are false is the third outcome, an
attempt that ran out of steps, and that is the row a reader most needs to tell
apart from a fall. Then it authors a new task as a spec dict and shows the two
refusals the closed predicate registry makes at `from_dict` time, before a scene
exists - an unknown condition name and an unexpected keyword.

That last cell is the one worth having in a notebook rather than in prose. A spec
that compiled and then evaluated to `False` on every step would report a
completely plausible 0% success rate, and the reader would go and look at the
policy.

Two test modules, split by what they can establish without a GL context.
`tests/test_notebook_evaluation_spec.py` parses the notebook's `SPEC` with
`ast.literal_eval` rather than executing it, which is the assertion rather than a
convenience: the notebook tells the reader a spec is safe to load from an
agent-produced file because there is nothing executable in it, and a spec that
grew a call or an f-string fails there. It then compiles the spec against the live
`PREDICATE_REGISTRY`, names each predicate individually so a rename reports the
whole list in one run, asserts the spec exercises all three clause kinds (success,
failure and reward terms - a success-only spec would compile and leave the two
parts a reader is most likely to copy ungraded), and executes the notebook's own
refusal cell, whose `else` branch is the assertion.

`tests_integ/simulation/test_notebook_evaluation_runs.py` runs every code cell in
order and asserts the result carries the fields the cells index -
`episodes_completed`, `episodes`, and per row `success` / `failure` / `steps` /
`cumulative_reward` / `seed`. A renamed key raises `KeyError` in the reader's
browser and in no test, because nothing imports a notebook. It also asserts the
replay claim by running the same evaluation twice: the same master seed produced
the same attempt seeds and a different one did not, so the sentence telling a
reader they can replay an attempt is checked rather than asserted.

Both modules read the notebook's cells rather than a transcription of them,
selecting each by content instead of index, so inserting a cell does not silently
point an assertion at the wrong code and a copy cannot keep passing after the
notebook regresses - the same discipline `tests/test_notebook_capability_model.py`
already applies to notebook 6.
