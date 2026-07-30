### Fixed: notebook 5 recorded zero frames while reporting success

Cell 3 recorded at `fps=30` and called `run_policy` without a
`control_frequency`, which defaults to 50 Hz. The recorder writes one frame per
control step with no decimation, so the rate guard refused the rollout rather
than writing frames whose timestamps implied a 1.667x slower episode. The
refusal arrived as a status dict that the cell discarded, so it printed
`recorded -> /tmp/nb5_dataset` over a dataset containing nothing but
`meta/info.json`, and every later cell in the notebook ran against it.

The rollout now runs at `control_frequency=30.0` to match the recording, and
both `run_policy` and `stop_recording` have their status checked so a refusal
stops the notebook where it happens.
