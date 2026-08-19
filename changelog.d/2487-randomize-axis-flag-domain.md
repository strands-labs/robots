### Fixed

- **simulation**: `randomize()` checks each axis flag on the shared boolean-flag
  domain instead of reading it for truthiness, on both backends that implement
  it. `randomize_colors` / `randomize_lighting` / `randomize_physics` /
  `randomize_positions` select a posture, so `"false"`, `"no"`, `"off"` and
  `"0"` - every spelling an operator reaches for to turn an axis off - are
  non-empty strings that turned that axis **on** while the call reported it
  applied. Randomization is destructive and the physics and position axes
  default to `False` for that reason: the misread rescaled mass, inertia and
  friction and displaced every object by up to 44 mm, and because the position
  axis writes `model.qpos0` the displacement survived `reset()`. On Newton the
  flags are stored through `bool()`, so a misread persisted to every later
  rebuild, and the MuJoCo-parity refusal - which branches on
  `randomize_positions` - answered `randomize_positions="false"` with
  "not supported by the Newton backend", an unsupported-axis verdict for a
  caller who had asked *not* to randomize positions. A misspelled axis *name*
  was already refused with the valid list; this is the same guarantee for the
  value that name carries.
