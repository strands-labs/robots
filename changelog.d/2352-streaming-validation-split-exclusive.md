### Fixed

- `LerobotTrainer.validate()` now refuses `streaming=True` together with `val_episodes`
  instead of reporting a launchable spec that delivers neither. A non-zero
  `dataset.eval_split` routes lerobot into `make_train_eval_datasets`, which rebuilds both
  splits as map-style `LeRobotDataset` objects without consulting `dataset.streaming`, so
  the whole dataset was materialized - the disk/RAM blowup `streaming` exists to avoid -
  with nothing reported. The refusal names both remedies.
