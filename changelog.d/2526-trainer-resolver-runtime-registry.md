### Fixed

- **training**: `import_trainer_class` now resolves a provider registered with
  `register_trainer`, so it serves every name `list_trainers()` advertises.
  It consulted only the `policies.json` `"trainer"` block and auto-discovery on
  `strands_robots.training.<provider>`, which refused the runtime-registered
  providers - including `ppo` and `fast_sac`, whose modules live in the
  `training.rl` subpackage - while `create_trainer` resolved them; the refusal
  it raised also listed those providers as available. The runtime lookup now
  lives once in `import_trainer_class`, which `create_trainer` delegates to,
  and keeps its precedence over a shipped `trainer` block.
