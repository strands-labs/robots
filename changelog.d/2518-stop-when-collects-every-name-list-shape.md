### Fixed

- **simulation**: collect every name in a `stop_when` clause's `particles` /
  `containers` kwarg whatever sequence shape it is spelled as. `run_policy` probes a
  clause's entity names against the live scene before arming it, because a typo'd name
  compiles clean, degrades to a constant `False`, and lets the rollout burn its whole
  step budget reporting `stopped_reason="budget"` - what an honest miss reports too.
  The walker collected the list-valued kwargs only when the value was a `list`, while
  the pour predicates take their shape contract from `name_list_error`, which accepts
  any `Sequence` that is not a `str`/`bytes` - the same domain under which
  `render_all(cameras=("default",))` and `set_robot_state_keys` already accept a tuple.
  A clause spelling its beads as a tuple therefore compiled, armed, and never probed a
  single bead name: with one bead misspelled it ran all 200 steps and reported success.
  The walker now accepts the shapes the factories accept, so both spellings of the same
  clause reach the same refusal at step 0.
