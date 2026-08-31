### Tests: the lerobot stand-in config tracks its factory's field reads

`make_train_eval_datasets` reads `cfg.dataset.<field>` straight off the config it
is handed, so a stand-in missing one raises `AttributeError` from inside lerobot
rather than failing an assertion in the suite that owns it. lerobot's depth-map
support added `depth_output_unit`, and the training suite's stand-in did not
carry it, which surfaced as that suite's own train/eval split assertion appearing
to break.

The field is added, mirroring lerobot's own default, and the expectation is now
derived from the function the split test actually calls, so the next field
lerobot starts reading fails with a name and a remedy instead of an opaque
`AttributeError`. The scope is the function rather than the module on purpose:
`make_dataset` also reads `episodes`, `exclude_episodes` and `repo_type` on paths
that test never reaches, so a module-wide rule would demand three fields the
stand-in has no reason to carry.
