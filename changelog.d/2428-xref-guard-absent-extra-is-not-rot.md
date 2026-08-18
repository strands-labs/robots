### Fixed

- **tests**: the docstring cross-reference guard no longer reads a module whose optional dependency is absent as a dead pointer. Such a target is undecidable in that environment rather than wrong, so it is skipped - the rule the guard's short-form half already applied - while a path naming no `strands_robots` module stays reported. A lazy module-level `__getattr__` raising `ModuleNotFoundError` through `hasattr` no longer ends the sweep with nothing graded.
