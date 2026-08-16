### Fixed

- **Packaging**: `strands-agents` is now floored at `>=1.7.0` (was `>=1.0.0`) in both
  the core dependency and the `[ollama]` extra. The package imports
  `strands.types.tools.ToolContext` (first shipped in strands-agents 1.5.0) and
  `strands.types._events.ToolResultEvent` (first shipped in 1.7.0) at module
  scope, so every release in `1.0.0-1.6.x` satisfied the declared range while
  leaving the MuJoCo backend and the real-hardware `Robot` unimportable. Because
  both are lazy exports, that install did not refuse cleanly: `import
  strands_robots` succeeded, `strands_robots.Simulation` raised a bare
  `AttributeError`, and `Robot(..., mode="sim")` blamed the MuJoCo backend rather
  than the dependency. No source change - the declared range now describes the
  configuration that was always required.
