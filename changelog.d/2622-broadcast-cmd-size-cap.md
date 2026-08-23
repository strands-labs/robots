### Fixed: broadcast reports the cmd-topic size cap it shares with send

The transport's `low_pass_filter` caps `**/cmd` and `**/broadcast` with one rule
(`strands_cmd_size_cap`, 16 KiB, ingress and egress) while the command validator admits a
`world_update` up to `MAX_WORLD_UPDATE_BYTES` (64 KiB). `Mesh.send` pre-checked that gap and
answered with a structured error naming the cap and the env var that raises it.
`Mesh.broadcast` publishes on the other key expression of the same rule and did not, so a
validator-valid command four times the wire limit was handed to the filter and returned the
empty list a broadcast nobody answered also returns.

`Mesh._cmd_topic_size_problem` is now the one owner of that cap and both publishers ask it, so
the two key expressions of one filter rule cannot come to disagree about the limit they share.
`broadcast` reports the reason the way it already reports a client-side rejection: log it and
return no responses. `send`'s error wording is unchanged.
