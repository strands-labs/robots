### Fixed

- **simulation/mujoco**: `tool_spec.json`'s `policy_config` description
  attributed three keys to the `lerobot_local` provider that
  `LerobotLocalPolicy.__init__` does not declare - `trust_remote_code`,
  `observation_mapping` and `action_mapping`. `policy_config` is splatted into
  `create_policy`, so each was swallowed by the policy's `**kwargs` absorber:
  the build reported success and the value was dropped without a warning, which
  reads as the capability being broken rather than as the key being wrong. The
  list now names declared parameters, including the `camera_key_map` and
  `obs_rename_override` keys the runtime's own camera-routing refusals already
  tell callers to pass. A new test grades the per-provider key list against the
  live constructor signatures so the schema cannot drift from them again.
