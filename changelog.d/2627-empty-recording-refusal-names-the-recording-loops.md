### Fixed

- **simulation/recording**: the empty-recording refusal now classifies each loop by
  whether it takes an `on_frame` hook. `stop_recording` refuses a session that
  captured nothing, and that refusal is the only thing a caller whose dataset came out
  empty reads, but it was wrong in two opposite directions. It said frames "are written
  only by `run_policy(...)`" and that "eval_policy / evaluate / replay_episode and bare
  step loops do NOT feed the recorder" - denying a documented, working route, since
  `eval_policy` and `evaluate_benchmark` both take an `on_frame` hook and
  `eval_policy`'s own docstring recommends it for exactly this ("Use it to record
  frames"). Measured, a hook calling `add_frame` over a 20-step eval writes 20 frames
  and `stop_recording` then saves the episode. Meanwhile `teleoperate` - the apply loop
  a caller recording a teleoperated demonstration reaches for, and the one that
  genuinely has no such hook - was absent from the list entirely, so a caller who drove
  30 leader frames into an open recording read a "does not feed" list that never
  mentioned the loop they had just used. The old text also named `evaluate`, which is
  not a method on the simulation surface. The refusal now names `run_policy` as the
  loop that feeds the recorder on its own, `eval_policy` / `evaluate_benchmark` as the
  ones that record when the caller passes a hook, and `replay_episode` / `teleoperate`
  / bare `step` loops as the ones that cannot. No behaviour changes: the refusal fires
  on the same condition and still carries the phrases its existing assertions pin. The
  classification is derived from the signatures in the new guard rather than restated,
  so an apply loop that gains an `on_frame` parameter fails until the refusal moves it.
