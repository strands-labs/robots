### Fixed

- **simulation/mujoco**: `actuate_robot` now checks `gravity_compensation` and
  `disable_self_collision` on the shared boolean domain instead of reading them
  by truthiness. Both are published as `"type": "boolean"` on the agent tool
  surface, and both were laundered through `bool()` before reaching the spec, so
  `disable_self_collision="false"` (also `"no"`, `"off"`, `"0"`, `nan`) zeroed
  `contype`/`conaffinity` on every one of the robot's geoms - the branch the
  caller was opting out of - while `None` and `0.0` took the other branch
  without being a declared spelling of it. Their four numeric siblings in the
  same signature have been on a shared domain all along.
