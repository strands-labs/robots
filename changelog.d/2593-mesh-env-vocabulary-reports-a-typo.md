### Fixed

- **mesh**: `STRANDS_MESH` now reports a value it does not recognize instead of
  ignoring it. The variable's vocabulary was split between the `Robot` factory,
  which owned the affirmative spellings, and `mesh.core.mesh_disabled_by_env`,
  which owned the kill-switch ones - so each half correctly treated the other's
  words as none of its business and neither had the standing to call a third
  value a mistake. `STRANDS_MESH=off` fell through both in silence and
  constructed the mesh it was meant to kill, advertising a `gateway-*` peer to
  the fleet against an explicit `mesh=True`; `off` is a spelling
  `STRANDS_ROBOT_MESH_DC` accepts, so an operator reaches it without guessing.
  `strands_robots._mesh_switch` now owns both halves and returns a tristate,
  reporting an unrecognized value once per distinct offending value and naming
  the spellings that work. Recognized spellings, unset and empty are unchanged:
  whether `off` should *mean* `false` is a behaviour change on a safety switch
  and is left to its owner.
