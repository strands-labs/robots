### Fixed: `val_episodes` now produces a validation loss instead of only a smaller training set

`val_episodes=N` reserved the last N episodes by emitting
`--dataset.episodes=[0..total-N-1]`, which restricts the TRAINING set only.
lerobot builds its evaluation dataloader from `dataset.eval_split`, so the
reserved episodes were used by neither half and no validation loss was ever
computed - and adding `eval_steps` alone is refused by lerobot
(`eval_steps > 0 requires dataset.eval_split > 0.0`). Every surface taking
`val_episodes` (the `lerobot_train` tool, `train_policy`, and `TrainSpec` on both
the argv and in-process paths) now emits lerobot's coupled pair: the
`dataset.eval_split` fraction that holds out exactly N episodes, plus an
`eval_steps` cadence taken from `save_freq` so each saved checkpoint has a
validation loss beside it. The fraction is the midpoint of the interval that
ceils to N rather than its `N / total` boundary, which is not float-safe
(`25 * (7 / 25) == 7.000000000000001` reserves 8). A multi-task dataset, where a
per-task fraction cannot express a global count, is refused with the fraction to
pass instead. `dataset.eval_split` or `eval_steps` supplied through the
passthrough still win.
