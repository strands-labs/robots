### Removed: the vendored LIBERO benchmark adapter, suite and drivers

`strands_robots/benchmarks/` is gone in full. LIBERO was its only citizen, so the
namespace goes with it, along with the `[benchmark-libero]` extra, its
`[tool.uv] conflicts` pair with `[vera-sim]`, the `libero.*` mypy override,
`examples/libero/`, the three `examples/mujoco_gs` LIBERO drivers, 30 test
modules and the `libero` dependency subtree in `uv.lock` (`wandb`,
`tensorboard`, `sentry-sdk`, `thop` and the `robosuite` 1.4.0 fork among them).
`robosuite` stays at 1.4.1 through `[vera-sim]`, which still needs it. Net
change: 96 files, +336/-33,213 lines.

LIBERO is an external benchmark. Vendoring an adapter for it made this package
the owner of a second simulator stack - robosuite plus bddl plus an OSC
controller, a BDDL parser and a suite loader - whose only consumer was that
benchmark.

One removed path is worth naming for anyone who mounted it by hand:
`examples/libero/gr00t_server_deterministic_wrapper.py` was a shim that
re-exported `main()` from the wheel-shipped determinism wrapper at
`strands_robots/policies/groot/server_wrapper.py`. Mount that packaged file
instead, or pass `deterministic=True` to the `gr00t_inference` lifecycle, which
resolves and mounts it for you. The wrapper itself is unchanged and still
pinned: it ships inside the package, imports on the host without torch or
gr00t, and stays free of `strands_robots` imports so it can run alone inside
the container.

The declarative benchmark surface is untouched:
`strands_robots.simulation.benchmark` (the `BenchmarkProtocol` and its
registry), `strands_robots.simulation.benchmark_spec` and
`strands_robots.simulation.builtin_benchmarks` still work exactly as before, so
`list_benchmarks`, `register_benchmark_from_file` and `evaluate_benchmark` are
unchanged. Authoring a benchmark is a spec file; it is no longer a vendored
adapter.

Three guards were re-anchored rather than deleted, because each pins a fact that
outlives the adapter. `tests/test_dependency_audit.py` imported its
numba/coverage clash floor from the adapter; the clash is a
numba/coverage/robosuite property, so the floor is now declared where the audit
reads it. `tests/test_optional_dependency_skips_bind_their_names.py` used
`libero` as its worked example and cited two adapter test modules for the
resolve-into-a-local idiom; both are now shape descriptions.
`tests/test_lockfile_parity_gate.py` and `scripts/check_lockfile_parity.py`
justified "every locked version below the floor, not any" with `robosuite`
locked at both 1.4.0 and 1.4.1 - the 1.4.0 fork arrived through the LIBERO
extra, so those docstrings now name the three distributions that are still
multi-version (`gymnasium`, `torchcodec`, `transformers`) and record that
nothing carries a below-floor version today. The rule itself is unchanged.

Comments describing upstream LIBERO's own scene conventions are kept verbatim in
`strands_robots.simulation.predicates` and `strands_robots.simulation.base` - the
`<name>_main` root-body fallback, robosuite geom naming and the combined
contact-plus-geometry `body_on` check exist because real MJCF scenes are named
that way, which this removal does not change.

`[all]` is now 19 of 30 extras. `README.md`, `docs/architecture.md`,
`docs/getting-started/installation.md`, `docs/api-reference.md`,
`docs/simulation/isaac.md`, `docs/simulation/world-building.md` and
`examples/README.md` follow.
