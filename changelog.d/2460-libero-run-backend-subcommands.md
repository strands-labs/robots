### Changed: `examples/libero` merges `run_mujoco.py` and `run_isaac.py` into `run.py` with backend subcommands

The two LIBERO drivers promised "switch backend" but the switch was a
different script: five helpers were copy-pasted between them (already
drifting at the docstring level), and the GR00T server orchestration
existed only on the Isaac side. `examples/libero/run.py` now carries one
shared core - parser base, the shared helpers, task resolution, GR00T
orchestration (the MuJoCo path gains auto-orchestration's extracted
`_orchestrate_groot_server` plus `HF_TOKEN` env-var support), the
eval-and-report loop, and one result-line formatter - with two thin
per-backend setup shims behind `run.py mujoco` / `run.py isaac`
subcommands. Isaac-only flags (`--robot-usd`, `--robot-urdf`,
`--eef-body-name`) exist only on the `isaac` subcommand;
`--deterministic` is now available on both. The chosen backend is
imported only after parsing. Both subcommands now emit the same
grep-stable result line, including `resolved_task=` and `backend=`
(previously Isaac-only fields; the matrix parser already tolerated
them). `libero_backend_matrix.py` invokes `run.py <backend>`, and the
output-line contract is pinned against the driver's own formatter in
`tests/test_examples_libero_drivers.py`. The `--container` default is
now derived per subcommand (`gr00t-libero-<backend>`), preserving the
two old drivers' hardcoded names so side-by-side runs keep separate
containers; that derivation - and an explicit `--container` surviving
it verbatim - is pinned in the same test file.
