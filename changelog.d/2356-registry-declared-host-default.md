### Fixed

`build_policy_kwargs` no longer overrides a policy provider's declared `host`
default. The generic `policy_host` parameter defaulted to the literal
`"localhost"`, which the merge could not distinguish from a caller-supplied
value, so `host` was injected on every call and no declared default was
reachable -- contradicting the function's documented rule that "a default only
ever fills a key the caller left unset". `moveit2` and `vera` (which declare
`127.0.0.1` in `policies.json`) and `lerobot_async` and `remote` (which declare
it as their constructor default) were all handed `localhost`. `policy_host` now
defaults to `None` like the other generic parameters; an explicit `host=` or
`policy_host=` still wins as before.
