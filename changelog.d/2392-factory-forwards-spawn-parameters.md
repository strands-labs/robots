### Fixed

- **robot**: `Robot(...)` now forwards `orientation` and `keyframe` to the backend's
  `add_robot`. Both are `SimEngine.add_robot` parameters, but the factory did not pass them
  on, so they fell into `**kwargs` and were absorbed by the backend constructor: a requested
  base rotation or `<keyframe>` spawn pose was silently dropped and the robot came up
  unrotated in the zero configuration while the factory reported success. The refusals
  `add_robot` documents for a malformed quaternion and for an unknown keyframe now reach the
  caller too, instead of the unknown keyframe falling back to zeros. Both are keyword-only
  and omitting them is unchanged.
