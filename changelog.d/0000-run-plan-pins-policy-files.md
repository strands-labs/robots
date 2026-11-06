### Fixed: an agent's approved run reads the policy files the operator approved

An agent's `run()` on a real robot asks `approve(RunPlan)` before the policy is loaded, and the plan named a local
checkpoint or dataset only by its path. If the files at that path changed while the operator was deciding, the run
used the new files. `RunPlan` now has `files`, the SHA-256 of the names and bytes of every file under a policy path on
this machine (symlinked folders included). After loading, the run checks the files again and refuses with
`RequestRefused` before anything moves if they differ. Presets, hub repo ids and callables have `files=None`, as before.
