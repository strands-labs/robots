### Fixed

- **tests**: the docstring cross-reference guard now grades `tests/` and `tests_integ/` alongside the shipped package, and resolves a qualified target's members with the same permissive rule the short-form half already used. Two test-module docstrings cited `strands_robots.mesh_session.get_session`, a module that was folded into the `mesh` package; they now name the seam their fixtures patch.
