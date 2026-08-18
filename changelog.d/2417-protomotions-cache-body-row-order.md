### Fixed

- **policies/protomotions**: `qpos_to_motion_data` now fills the reference-motion
  cache's body rows by name against `GTP_G1_BODY_NAMES` - the order the tracker
  indexes them by - and refuses an MJCF that cannot supply a body it reads or whose
  `qpos` layout is not the tracker's. Read positionally, a fingerless 30-body G1
  shifted every row after the missing `head`, so the tracker was handed
  `left_shoulder_pitch_link` where it asked for its `torso_link` anchor.
