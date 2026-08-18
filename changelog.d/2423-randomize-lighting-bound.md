### Fixed

- **simulation/mujoco**: `randomize(randomize_lighting=True)` now offsets each light from its authored position rather than from the position the previous call left it at, so a per-episode randomization loop keeps every light inside the documented +/-0.5 m bound instead of walking it out of the scene, and replaying a `seed` reproduces the same lighting. A world whose authored light poses cannot be read is refused with nothing applied.
