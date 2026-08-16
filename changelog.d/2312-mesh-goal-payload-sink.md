### Fixed

- **mesh**: a `tell()` goal payload now reaches the call that reads it. The
  sim-peer dispatcher forwarded the well-known goal (`target_pose` /
  `target_joints` / `world_update`) in `policy_config`, which is expanded into
  the Policy constructor; no provider names a goal key there, so the goal was
  discarded and the planner then refused the request that had carried it. It now
  travels in `policy_kwargs`, which the runner forwards verbatim to every
  `get_actions` call, while constructor extras (`model_path`, `server_address`,
  `policy_type`, `pretrained_name_or_path`) keep travelling in `policy_config`.
