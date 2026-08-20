### Fixed

- **mesh**: report an absent `eclipse-zenoh` at `WARNING` instead of `DEBUG`, so a
  default-configured consumer learns why the mesh never came up. Both session-open
  paths and `Mesh.start` left the degradation below the level anything prints, so the
  first observable symptom was whichever downstream wait expired first. The
  dependency is named once per process with the extra that supplies it; the mesh
  still stays off rather than raising.
