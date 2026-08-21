### Fixed

- **mesh**: `strands_robots.mesh` no longer re-exports the private in-process
  robot registry. `_LOCAL_ROBOTS` and `_LOCAL_ROBOTS_LOCK` were listed in
  `__all__`, which is the sole reason `from strands_robots.mesh import *` bound a
  mutable registry dict and its lock into the importer's namespace - a
  star-import skips underscore names unless `__all__` overrides that. Reach a
  snapshot through the already-public `get_local_robots()`, or
  `strands_robots.mesh.core` for code that needs to mutate the registry itself.
