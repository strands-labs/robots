### Fixed

`create_policy` now enforces the trust-remote-code gate for every spelling a
provider declares, not only its canonical name. `_resolve_policy_class` returned
the caller's spelling where it documents `canonical_provider_name`, and the gate
membership-tests a set of canonical names, so `create_policy("lerobot")`,
`create_policy("kimodo_g1")` and `create_policy("text2motion")` constructed
providers that load HuggingFace models with `trust_remote_code=True` without the
`STRANDS_TRUST_REMOTE_CODE` opt-in. Resolution now reports the canonical name
through a single canonicalisation helper that `get_policy_provider` and
`import_policy_class` had each restated inline.
