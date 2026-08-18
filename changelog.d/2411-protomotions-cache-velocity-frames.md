### Fixed

- **policies/protomotions**: name the frame the reference-motion cache's velocity
  channels are in. `_quat_finite_diff_ang_vel` documented its return as a
  local-frame angular velocity while left-multiplying the quaternion delta, which
  is the world-frame quantity that every other surface here already expects -
  `compute_root_local_ang_vel` takes a world-frame `rigid_body_ang_vel` and the
  policy reads a `body_ang_vel_world` observation key. Neither
  `qpos_to_motion_data`'s `Returns:` section nor the `MotionPlayer` cache-format
  contract named a frame at all, so a caller building a cache by hand had nothing
  to follow; local-frame rows stored there are rotated a second time rather than
  used as-is. On a bridged G1 walk clip the two frames differ by up to 6 rad/s.
