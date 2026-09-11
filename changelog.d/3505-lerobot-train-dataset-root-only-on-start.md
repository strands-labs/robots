### Fixed: `lerobot_train` no longer demands a dataset to report on a run

`dataset_root` was the tool's only required parameter, yet `status`, `stop` and
`list` never read it - they look a session up by name. An agent asked to check
on a training run therefore had to invent a dataset path before the call was
accepted, and `lerobot_train(action="list")` from Python raised `TypeError`
instead of listing anything. `dataset_root` is now optional; `start` still
refuses to launch without it, naming the parameter, and the three session verbs
answer without it.
