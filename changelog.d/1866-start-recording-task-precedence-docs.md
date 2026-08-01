### Docs: `start_recording(task=)` says which frame task actually wins, and names the `"untitled"` it falls back to

`add_frame` owns the whole rule - `frame["task"] = task or self.default_task or
"untitled"` - and `start_recording(task=...)` feeds only the middle term, through
`create(task=...)` / `resume(task=...)`. The term that wins is `add_frame`'s own,
and every rollout hook passes the rollout instruction there:
`simulation/{isaac,newton}/recording.py` and `simulation/mujoco/simulation.py` all
call `add_frame(..., task=instruction)`. So the effective precedence is three
levels deep,

```
frame["task"] = run_policy(instruction=...)  or  start_recording(task=...)  or  "untitled"
```

and which level supplies the value depends on an argument to a *different* method
- `run_policy(instruction: str = "")` defaults to empty.

Two of the three docstrings described the parameter as the value "recorded with
every frame", which is the one thing it is not, and the third documented it not at
all:

| surface | `task` | `push_to_hub` |
| --- | --- | --- |
| `simulation/isaac/recording.py` | "Default task description recorded with every frame." | documented |
| `simulation/newton/recording.py` | same wording | documented |
| `simulation/mujoco/recording.py` | *no entry at all* | *no entry at all* |

The wrong claim is quieter than the omission. A caller who reads it, sets
`task="pick up the red cube"`, then drives the rollout with
`run_policy(..., instruction="place the cube in the bin")` gets every frame
annotated with the second string; both are plausible, so the dataset reads as
correctly annotated. And the terminal level was named nowhere: set neither and
every frame carries the literal `"untitled"`, which is the conditioning signal for
every language-conditioned policy this repo targets (pi0/pi0.5, SmolVLA, GR00T,
MolmoAct2) - a dataset recorded that way trains against a constant instruction.
`start_recording`'s success text already hinted at the coupling with
`Task: {task or '(set per policy)'}`, but only in the returned string, and it does
not say where an unset instruction lands.

All three entries now state the precedence and cite
`DatasetRecorder.add_frame` for it, rather than restating it a fourth time - the
same single-prose-owner arrangement #1865 established for `resolve_dataset_dir`,
with one addition: that resolver already documented its own rules, while
`add_frame`'s `task` entry read "uses default if None" and named neither the
session default it meant nor the fallback. The owner is now written out, so the
cross-references resolve. MuJoCo's `Args` block gains `task` and `push_to_hub`,
completing it after #1865 added `repo_id`.

`TestStartRecordingDocumentsTheTaskThatLandsOnAFrame` guards all of it in
`tests/simulation/test_recording_dataset_home_across_backends.py`, beside the
directory-resolution checks that already read these docstrings. Eleven of its
sixteen checks fail on the previous wording - including the two backends that were
already *complete*, since completeness is not accuracy.

Docstrings and tests only - no behaviour change.
