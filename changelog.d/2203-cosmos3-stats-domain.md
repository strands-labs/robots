### Fixed: Cosmos 3 explicit de-normalization stats must declare their domain

`decode_cosmos_chunk_to_targets` accepted an explicit `stats=` override that need
not describe the embodiment being decoded. Only `droid_lerobot` and
`bridge_orig_lerobot` ship bundled quantiles, while `umi` and `av` -- the two
domains `nvidia/Cosmos3-Edge` documents for its forward- and inverse-dynamics
examples -- do not, and `load_action_stats` refused them while advising the caller
to pass stats explicitly. Because `umi`, `droid_lerobot` and
`bridge_orig_lerobot` are all 10 action columns, the only stats a caller could
load were accepted for `umi` by the width check and silently rescaled every
commanded pose delta: the two bundled domains decode the same normalized action
into translations differing by up to 2.77x. Explicit `stats=` now requires
`stats_domain=`, a domain other than the embodiment's is refused by name, and the
missing-domain message names the safe way to supply them.
