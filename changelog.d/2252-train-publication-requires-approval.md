### Fixed: a train run that publishes to the HF Hub now requires operator approval whichever spelling asks for it

`lerobot_train` blocks a set of LeRobot flags -- output paths, telemetry,
publication, code loading -- behind an operator approval prompt. Three of those
blocked flags are also named parameters of the tool, and the gate only inspected
`extra_flags`, so `push_to_hub=True` reached the argv unasked while
`extra_flags={"push_to_hub": True}` was refused. A caller could launch a
training run configured to publish the trained checkpoint -- to a destination
supplied through the ungated `policy.repo_id` flag -- with nobody approving it.

The named `push_to_hub` now routes through the same gate as `pretrained_path`
already did, so both spellings reach one verdict. The default `False` is not
gated, and the documented escape hatches (an approving `tool_context`, the
`STRANDS_TRAIN_EXTRA_FLAGS_ALLOW` allowlist, `BYPASS_TOOL_CONSENT`) are
unchanged.
