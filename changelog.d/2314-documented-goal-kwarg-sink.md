### Fixed

- Documented `run_policy` examples for the `curobo`, `moveit2` and `wbc_gait`
  providers wrote a per-call goal (`target_pose`, `target_velocity`,
  `gait_frequency`) directly as a `run_policy` keyword. `run_policy` has no such
  parameter and no `**kwargs`, so the example raised `TypeError` before the first
  tick; the goal belongs in `policy_kwargs`, which the runner hands to every
  `get_actions()` call. The `gr00t_inference` lifecycle example passed `tag=` and
  `model_id=`, neither of which exists - the image name is operator config
  (`STRANDS_GR00T_IMAGE`, allowlist-checked) and the checkpoint keyword is
  `hf_repo`. Three prose passages claimed `Robot.start_task` forwards a goal
  through a `**policy_kwargs` it does not have. A new guard grades every `python`
  fence in `docs/` and `README.md` against the real signatures.
