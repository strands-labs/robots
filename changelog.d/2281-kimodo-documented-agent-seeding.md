### Fixed

- **docs/policies/kimodo**: the documented native-runtime `motion_agent=` adapter
  now applies the `seed` it is handed. NVIDIA's runtime draws its initial noise
  from the global torch generator and accepts no generator or seed argument, so
  the example seeded nothing and every request sampled fresh noise. An adapter
  that accepts `seed` and ignores it still satisfies `KimodoMotionAgent`, so
  nothing raised - but it silently defeats the per-episode seed `eval_policy`
  derives, giving every episode a motion no seed can reproduce. The example is
  executed by the test suite against a stub runtime that draws from a global
  stream the way the real one does, so the same seed is pinned to one motion.
