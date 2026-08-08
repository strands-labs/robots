### Fixed: a non-finite mesh env knob no longer disables the safety bound it sizes

Both float-valued env resolvers in the mesh package read their value with
`float()`, which accepts `nan`, `inf`, `Infinity` and `1e999` (that last one
overflowing to `inf`), and neither range test beside them implies finiteness:
`_parse_positive_float_env` compares `value < minimum`, which is `False` for
`nan`, and the bridge deduplicator's `_resolve_dedup_ttl` compares `v > 0`,
which is `True` for `inf`.

Every knob the first resolver serves is one side of a comparison on the safety
path, so a `nan` did not widen the bound but removed it. Measured:
`STRANDS_MESH_RESUME_FRESHNESS_S=nan` made the presence stale/future test
(`age > window or age < -skew`) `False` for every envelope, so a year-old
presence was accepted; the replay-cache TTL purge kept all 5 of 5 stale
entries; and `STRANDS_MESH_RESUME_BACKOFF_S=nan` armed the resume brute-force
cooldown - the throttle over the E-stop override code - to
`monotonic() + nan`, which no later `now < locked_until` test can satisfy, so
it never engaged. `inf` failed open on the first two and closed on the third:
the cooldown never expired, so a resume could never be granted again. Nothing
was logged in any of these cases.

Both resolvers now fall back to their documented default for a non-finite
value, which is the rule the package's three other env-float resolvers already
applied - `mesh.security._env_pos_float` both documents it and names
`mesh.core._parse_positive_float_env` as its analogue. A structural test
asserts every env-float resolver in the package tests finiteness, so a sixth
cannot ship without it. The `0` floor is unchanged and pinned: the two
resolvers legitimately disagree there, and what a zero means differs per knob.
