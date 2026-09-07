### Fixed: the Isaac parallel-environment count is a count at both of its owners

`IsaacConfig.num_envs` is the configured default; `IsaacSimulation.replicate(num_envs=...)`
is the per-call request honoured instead of it. Neither was held to the shared count
domain `strands_robots.utils.positive_count_error`, which the sibling `camera_width` /
`camera_height` fields in the same `__post_init__` already take.

The field's hand-rolled `< 1` test read `True` as a count of 1 while refusing `False`,
and accepted `4.0`, `2.7`, `nan` and `inf`; a `str`, `None` or a list raised `TypeError`
from the comparison itself, naming neither the field nor a remedy. `IsaacSimulation`
logs this field with `%d`, so a stored `2.7` was announced as `num_envs=2` and a stored
`nan` made that logging call raise.

The argument had no domain at all and resolved by truthiness, so a supplied `0` or
`False` was read as "not supplied" and replicated to the configured count instead,
announcing that count under `status: "success"` with the caller's request discarded.
Every truthy value was stored and reported three times over -- by `replicate`, by
`get_state`, and by `destroy` as `num_envs_released` -- and latched `_replicated`, which
`add_robot` refuses on, so an unusable count locked the scene until `destroy`.
`num_envs="4"` rendered as `"Replicated to 4 environments"`, byte-identical to what the
int `4` produces, while the payload carried the `str`.

Both owners now reach one verdict on the shared domain: the field raises `ValueError`
like every other field in its `__post_init__`, the request reports `{"status": "error"}`
through the channel its no-world and no-robot refusals already use, and `num_envs=None`
still resolves to the configured count.
