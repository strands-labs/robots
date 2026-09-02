### Fixed: a rollout `duration` that resolves to no control step is refused instead of reporting an empty success

`duration` is one factor of the rollout horizon - with no `n_steps` the loop runs
`int(duration * control_frequency)` control steps - and the guard judged the factor rather
than the product, so every span shorter than one control period was positive, finite, and
resolved to zero steps. Across 32 MuJoCo rollouts (4 rates x 8 spans, `video=` requested on
each), all 16 sub-period rows reported `status="success"` with `0 steps`, `sim_t=0.000s` and
no MP4 on disk, including `duration=1.9` at 0.5 Hz.

`SimEngine._validate_duration` now takes the rate and refuses a horizon of no steps, naming
both factors and the minimum span. One guard covers all four surfaces that resolve a duration
horizon: `run_policy`, MuJoCo `start_policy`, and `run_multi_policy` on the MuJoCo and Isaac
backends. A fractional duration that does produce steps is unaffected - the boundary is only
at zero - and the value-domain message is unchanged.
