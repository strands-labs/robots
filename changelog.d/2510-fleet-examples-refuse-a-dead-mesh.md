### Fixed

- **examples/fleet**: refuse a mesh that did not start instead of proceeding to the
  presence wait. `init_mesh` returns `None` only when the mesh is switched off on
  purpose; when it is enabled but no session opens it returns a `Mesh` whose `alive`
  is `False`, which passed the four live-fleet builders' `is None` guard. Examples 02,
  03 and 04 then spent their 15 s presence-discovery timeout on a message naming no
  dependency, and 05 - which has no presence wait - ran its whole queue, attributed
  every order to a per-robot `dispatch_failed`, and exited 0. Each builder now checks
  `mesh.alive` (the observable `dashboard.py` already refuses on) and names the peer,
  the remedy and the `--dry-run` alternative. The deliberate `STRANDS_MESH=0` opt-out
  keeps its own distinct advice.
