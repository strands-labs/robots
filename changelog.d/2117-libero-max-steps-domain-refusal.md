### Quality: drive the LIBERO `max_steps` refusal on every surface that carries it

`LiberoAdapter.__init__` bounds `max_steps` with the shared strict-count domain,
and its own comment records what the bound is for: the value becomes the
benchmark's per-episode `range(max_steps)` bound, so a zero or negative horizon
runs episodes of zero length that still report a 0% success rate. The rejecting
half of that domain had never executed. Every existing test passed a usable
horizon, so the constructor, both `from_*` classmethods and `load_libero_suite`
all forwarded the parameter with the refusal unverified -- while the sibling
`init_jitter` refusal one branch above was exercised.

Adds `tests/benchmarks/libero/test_libero_max_steps_domain.py`, which drives the
refusal on all four surfaces, asserts each message equals the shared helper's
verdict rather than a local copy, pins that the suite loader's `except ValueError`
registers no task and reports the value once per task file, keeps a usable
horizon and the `None` "not supplied" spelling as controls, and adds a structural
sweep so a fifth surface cannot read the horizon without the domain. No library
behaviour changes.
