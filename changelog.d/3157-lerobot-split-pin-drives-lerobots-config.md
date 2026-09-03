### Fixed: the lerobot split pin drives lerobot's own `DatasetConfig`

`LerobotTrainer.validate` refuses `streaming` together with `val_episodes`
because lerobot holds out a validation split only on a **map-style** dataset:
`make_train_eval_datasets` rebuilds both halves as `LeRobotDataset` objects,
which is what makes the split addressable by episode. The cell that establishes
that property drove `make_train_eval_datasets` with a hand-built
`SimpleNamespace` for `cfg.dataset`, which has to mirror every field lerobot's
factory reads and skips `DatasetConfig.__post_init__`.

Both cost something inside the declared `lerobot>=0.6.1,<0.7.0` range. lerobot
0.6.2 has the split path read `repo_type` and `depth_output_unit` (7 fields ->
9), so the stand-in raises `AttributeError` from inside lerobot and two cells
fail; and 0.6.2's `DatasetConfig` now refuses the pair when it is constructed
(`eval_split requires map-style datasets and is not supported with
dataset.streaming=true.`), which a namespace stand-in cannot see. So the
consequence the refusal quoted -- that the whole dataset is materialized and the
stream silently dropped -- is the pre-0.6.2 outcome only; on 0.6.2 a launched
run fails at config construction instead.

The pin now builds lerobot's own `DatasetConfig`, defaults intact, overriding
only what the cell drives: the field set tracks lerobot by construction and the
invariants run. It grades the constraint in whichever form the installed lerobot
expresses it, and still fails with "lerobot may now honor the stream and the
preflight refusal should be lifted" if neither holds -- the reason the original
cell existed. The `make_dataset` double honours `dataset.streaming` the way
lerobot's own does, so the recorded build order reflects the kind the factory
chose rather than the kind the double happens to be.

The refusal text and the training guide state the constraint lerobot itself
states and name both outcomes rather than only the older one. The refusal and
its two remedies are unchanged.
