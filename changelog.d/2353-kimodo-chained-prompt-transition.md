### Fixed

`KimodoPolicy` now eases a newly sampled motion off the pose it last commanded
when the prompt changes mid-rollout, instead of emitting the fresh sample from
its own canonical start pose. Chaining prompts is the documented way to drive a
long-horizon sequence, but every segment boundary stepped all 29 joints at once:
across the 600 ordered pairs of a 25-motion corpus the median seam moved a joint
1.6 rad in a single tick, and 84% of transitions exceeded the largest per-tick
step the motions themselves ever take, all reported as a successful rollout. The
transition length is `KimodoConfig.transition_frames` (default 5, matching the
`num_transition_frames` Kimodo's own sampler applies to a multi-prompt
sequence). A single-prompt rollout is byte-for-byte unchanged.
