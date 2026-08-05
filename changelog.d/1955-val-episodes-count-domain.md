### Fixed: `val_episodes` is one shared positive-count domain across both writers of lerobot's `eval_split`

The held-out validation episode count was compared rather than validated, by each
of the two surfaces that build lerobot's `dataset.eval_split` flag, and the two
comparisons disagreed. The count is converted into a real-valued split fraction
whose ceiling lerobot takes, so a comparison is wrong at both ends: `0` and a
negative produced no split and no `eval_steps` at all - the run trained on the
whole dataset and logged no validation loss, with `LerobotTrainer.validate()`
reporting no problem while the `lerobot_train` tool refused the same value -
while `True` reserved 1 episode, `2.7` reserved 3, and `0.5` emitted an
evaluation cadence over a held-out set of zero episodes. A non-numeric count
raised `TypeError` out of the comparison, from a `validate` documented to
return problems.

Both writers now apply the shared `positive_count_error` domain through a new
`Trainer._validation_episodes_problems` gate, so an unusable count is reported
(never raised) with a message naming the backend, and the tool refuses it before
it reads the dataset. The dataset-dependent bounds are unchanged.
