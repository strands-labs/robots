### Fixed: the WBC gait clock integrates at the rate the runtime states

`GaitClock` advances the bipedal phase by `dt * freq` per tick, and
`WBCGaitPolicy.get_actions` called it without a `dt` -- so the phase always
advanced by the upstream reference period (0.02 s) however fast the executing
loop queried the policy. A commanded `gait_frequency` was therefore only in
steps per second at 50 Hz; at any other control rate the realised cadence came
out scaled by `control_frequency / 50` (6.0 steps/s for a commanded 1.5 at
200 Hz, 0.5 at 12.5 Hz), and the warm-up window `0.5 / freq` was mistimed by
the same factor.

The rate was already available. `PolicyRunner` calls
`Policy.set_control_frequency` before the rollout loop, and the
`Policy.control_frequency` contract says a provider that needs the rate must
warn loudly and fall back rather than silently assume one. `get_actions` now
integrates at `1 / control_frequency`; an unstated rate warns once per policy
and falls back to the upstream period, so a caller driving `get_actions`
directly is unaffected.
