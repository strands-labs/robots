### Fixed

- **policies/kimodo**: a per-episode seed now reaches the sampler. `KimodoPolicy`
  re-entered the sampler only when the prompt changed, so `reset(seed=...)` - the
  call `PolicyRunner.evaluate` makes once per episode - recorded the seed without
  ever applying it, and every episode after the first replayed episode 0's motion.
  The `diffusion_steps` and `guidance_scale` per-call overrides were dropped the
  same way. The buffered motion is now identified by every input that produced it,
  so a changed prompt, knob or seed re-samples while a repeated seed still replays
  the buffer and `reset()` without a seed still only rewinds.
