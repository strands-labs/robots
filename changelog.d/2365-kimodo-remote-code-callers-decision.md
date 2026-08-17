### Fixed: `KimodoPolicy` lets the caller decide whether a checkpoint runs its own code

`KimodoConfig.trust_remote_code` defaulted to `True` and was reachable by no
keyword at all: it was neither a `KimodoPolicy.__init__` parameter nor a
`config_keys` entry, so `KimodoPolicy(trust_remote_code=False)` and
`create_policy("kimodo", trust_remote_code=False)` raised `TypeError` while
`build_policy_kwargs` - the route a JSON `policy_config` travels - dropped the
key without a word and loaded with `True` anyway. Only hand-building the frozen
dataclass could reach the flag. `STRANDS_TRUST_REMOTE_CODE` did not cover the
gap: it decides whether the provider may be constructed and never sets this
field, so opting in to the provider also opted in to executing repository code
for every `model_id` the process loaded.

The flag now defaults to `False` and is an explicit constructor parameter and
`config_keys` entry, alongside `cache_dir`, which was unreachable for the same
reason. The shipped default `model_id` is refused at load time either way, so a
default rollout is unchanged.
