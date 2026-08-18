### Fixed

- **policies/protomotions**: `qpos_to_motion_data` no longer refuses a G1 MJCF that
  declares its own ground. Whether the model already has a floor is now read from
  the geom list MuJoCo parses out of the file rather than from the first
  `<worldbody>`'s direct `<geom>` children, so a ground declared in a second
  `<worldbody>` section, nested inside a body, or pulled in via `<include>` is
  honoured instead of being duplicated into a `repeated name 'floor' in geom`
  compile failure.
