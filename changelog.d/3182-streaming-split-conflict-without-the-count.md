### Fixed

- A streamed training run that also asks for a validation split is refused on
  the source `streaming` exists for. `LerobotTrainer.validate` asked whether
  `streaming` and `val_episodes` can both be delivered from inside the branch
  that had already read the dataset's `total_episodes`. That count is only
  readable from a local `meta/info.json`, and streaming a Hub dataset is the
  case with no local copy -- the download `streaming` exists to avoid -- so on
  that source the pair was never refused. What was reported instead was the
  unreadable-count refusal, whose remedy is
  `extra={'dataset.eval_split': <fraction>, 'eval_steps': <steps>}`: applying it
  with `streaming` still set left `validate()` reporting no problems at all and
  emitted `--dataset.streaming=true` beside `--dataset.eval_split=0.1`, where
  lerobot's split path rebuilds both halves map-style and materializes the whole
  dataset with the stream annulled and nothing reporting it. Whether the two
  fields can both be honored is a property of the two fields, so it is now
  decided from them alone, ahead of every count-derived check; a local root with
  an unreadable `meta/info.json` was the second source with the same gap and is
  covered by the same change. Where the count is unreadable `streaming=False`
  alone still cannot deliver the split, so the refusal names the local copy that
  configuration also needs and both remedies it offers are honored. The refusal
  also attributed the construction-time failure to "lerobot 0.6.2", which is not
  a release -- 0.6.1 is the latest, and its same-worded guard
  (`eval_split requires map-style datasets`) is keyed on `repo_type='bucket'`,
  which this backend never sets, so on 0.6.1 the pair constructs and the silent
  materialization is what happens. `lerobot.__version__` is
  `importlib.metadata.version("lerobot")`, so it reports the installed
  distribution rather than a release the claim could be checked against; the
  refusal now names the predicate instead.
