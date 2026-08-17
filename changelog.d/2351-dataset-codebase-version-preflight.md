### Fixed

`LerobotTrainer.validate()` now refuses a local dataset root whose declared
`codebase_version` has an older major than the installed lerobot's
`CODEBASE_VERSION`, instead of reporting the spec as launchable and letting the
dataset load fail. The refusal names the root, both versions, and a remedy that
exists: lerobot's `convert_dataset_v21_to_v30` for a v2.1 root, and an explicit
"no converter" for an older one. Previously a v2.1 root reached
`BackwardCompatibilityError` whose command named lerobot's `local` sentinel
rather than a convertible repo, and every other older major reached a bare
`NotImplementedError: Contact the maintainer on [Discord](...)` that named
neither the dataset nor the version. The version is read from the same
`meta/info.json` the trainer already reads, so the check stays offline: a Hub
dataset with no local cache is left unflagged, and an unparseable version fails
open.
