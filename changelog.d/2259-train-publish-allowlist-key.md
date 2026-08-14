### Fixed: `lerobot_train`'s `push_to_hub` description named an allowlist key that does not clear it

`push_to_hub` is the one blocked flag `_BLOCKED_EXTRA_FLAGS` names twice, bare
and `policy.`-prefixed, and the two entries are not interchangeable: the named
parameter is gated under `policy.push_to_hub` (the only spelling LeRobot accepts,
since `push_to_hub` is a field of its policy config rather than of the train
config), while the bare key covers the raw
`extra_flags={"push_to_hub": True}` passthrough. The parameter's agent-facing
description said the approval could be pre-approved "exactly as the
`extra_flags={'push_to_hub': True}` spelling already does", which reads as one
shared opt-out -- so a headless run that set
`STRANDS_TRAIN_EXTRA_FLAGS_ALLOW=push_to_hub` still hit the approval prompt with
nobody there to answer it, and the description was the only place that env var is
documented at all. The description now names `policy.push_to_hub` and says which
spelling each entry clears, and the blocklist carries a comment recording why the
pair is not a duplicate.
