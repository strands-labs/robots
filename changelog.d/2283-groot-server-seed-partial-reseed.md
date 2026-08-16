### Fixed

- **policies/groot**: the determinism wrapper's per-episode reseed now applies a
  seed to every RNG or to none of them. It seeds Python `random`, then NumPy's
  legacy global RNG, then torch, and only the second bounded the value - so any
  seed NumPy refuses (negative, non-integral, or above `2**32 - 1`) left the
  server part-way reseeded, with Python `random` moved, NumPy and torch
  untouched, and the upstream `reset` never reached. The client swallows a
  failed reset at `INFO`, so an episode drew part of its randomness from a fresh
  stream and part from the previous episode's while reporting that no reseed had
  happened. The seed is now checked before the first applier runs, against the
  domain `randomization_seed_error` already enforces for a rollout seed. A
  refused per-episode request falls back to the already-validated configured
  seed with the reason printed, so the episode still gets one whole reseed and
  still reaches the upstream `reset`; `bool`, numeric strings and floats are
  refused rather than read or coerced into a seed the caller never named; and an
  unusable `STRANDS_GR00T_SERVER_SEED` is refused at startup naming the variable
  and the domain instead of raising out of NumPy with the interpreter's RNG
  already moved.
