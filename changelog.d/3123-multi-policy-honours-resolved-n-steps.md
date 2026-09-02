### Fixed: `run_multi_policy` honours the step count it resolved

Both synchronized multi-robot rollout loops (MuJoCo, Isaac) discarded the
`n_steps` that `_resolve_horizon` normalized and recomputed the loop bound as
`int(duration * control_frequency)` from the `n_steps / control_frequency`
duration. That float round trip truncates at any rate the count does not divide
evenly: `n_steps=29` at 50 Hz ran 28 steps, and `n_steps=1` at 49 Hz ran zero
and still reported a completed rollout. One merged frame is recorded per
timestep, so a truncated horizon also shortened the recorded dataset episode.
The single-robot loop (`PolicyRunner.run`) already forwarded the count verbatim;
both multi-robot loops now do the same.
