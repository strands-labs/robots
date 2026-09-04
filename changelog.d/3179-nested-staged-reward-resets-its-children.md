### Fixed

- **`staged_reward`: a nested phase machine is reset at the episode boundary too.**
  `staged_reward` is a registered predicate and a stage's `reward` is compiled by
  calling back into `make_predicate`, so a stage may hold another phase machine and
  a curriculum can be authored as a machine of machines. `_StagedReward.reset`
  cleared its own `_phase` and nothing else, so a nested machine kept whatever stage
  the previous episode left it in: the next episode opened inside a sub-curriculum
  it had not earned, never emitted that sub-curriculum's earlier shaping term again,
  and paid its one-time `bonus` once per process rather than once per episode. The
  outer phase reset correctly either way, so nothing about the composite's own state
  reported the leak. Measured on a real MuJoCo scene through a benchmark compiled
  from a spec dict, a two-episode run scored `[1, 1, 1, 101, 7, 7]` then
  `[7, 7, 7, 7, 7, 7]`. Sub-terms are now cleared by the same duck-typed rule
  `SimEnv.reset` and `DeclarativeBenchmark.on_episode_start` apply to the terms they
  hold - anything exposing a zero-arg `reset()` - so a composite applies to what it
  holds exactly the rule its consumers apply to it, and a nested machine clears its
  children in turn.
