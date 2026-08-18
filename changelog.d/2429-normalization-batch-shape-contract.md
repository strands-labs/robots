### Fixed

- **training/rl**: `EmpiricalNormalization` now refuses a batch it cannot fold instead of
  silently broadcasting it. An unbatched `(num_obs,)` observation was read as `num_obs`
  samples of a one-feature stream - pooling the per-feature statistics into a single number
  and returning a `(1, num_obs)` tensor the caller never passed in - and a mis-shaped batch
  that did raise had already advanced the persisted sample count. Both entry points now
  validate the batch shape against the per-feature shape they were built for and name the
  remedy, and `count` is committed only after the fold succeeds, so a rejected batch leaves
  every buffer untouched.
