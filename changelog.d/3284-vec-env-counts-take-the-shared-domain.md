### Fixed: the counts `VecSimEnv` is built from are held to the shared count domain

`VecSimEnv` is the class the RL trainers hand `RLTrainSpec.num_envs` to and the
one that acts on it, calling `env_factory` once per environment and sizing one
thread pool from `max_workers`. Every other owner of a parallel-environment
count grades it through `positive_count_error` -- `RLTrainSpec` via the PPO,
FastTD3 and FastSAC backends, and the Isaac backend's config and `replicate` --
but this one hand-rolled `if num_envs < 1` and then coerced with `int()`, and
`max_workers` had no domain at all.

A bare `< 1` test followed by a coercion admits values no number of environments
can honour: `num_envs=2.5` built two live engines, `4.0` built four and `True`
built one, each a count the caller never asked for rather than a refusal.
`float("inf")` and `"4"` left the constructor as `OverflowError` and
`TypeError`, outside the documented `Raises: ValueError` and naming neither the
class nor the parameter, and `max_workers=float("inf")` reached
`ThreadPoolExecutor` as its pool size.

Both counts now take the shared domain, so a count refused for a training spec
cannot be accepted by the class that spends it, and a refusal is reported before
any environment is built. The `int()` coercion is gone: the domain admits only a
true `int`, so it could only restate the value, and while it was there it was
what turned a `2.5` into two engines.
