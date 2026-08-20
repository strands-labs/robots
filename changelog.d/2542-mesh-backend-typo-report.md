### Fixed

- **mesh**: an unrecognized `STRANDS_MESH_BACKEND` is now reported by the resolver that
  actually sees it. Two readers resolved the variable and only the transport factory
  reported an unknown value, but the session gate in front of it resolves first and a typo
  never reaches the factory - so `STRANDS_MESH_BACKEND=iott` produced a plain Zenoh session
  indistinguishable from an explicit `zenoh`, with nothing naming the variable. The accepted
  values, the fallback and the report now live in one owner both readers ask, and the report
  is emitted once per distinct offending value because that gate runs per published message.
  Every value resolves to the same backend and transport as before.
