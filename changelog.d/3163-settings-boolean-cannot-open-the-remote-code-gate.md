### Fixed: a dashboard setting that is not a boolean can no longer open the remote-code gate

`runtime.trust_remote_code` is published to `STRANDS_TRUST_REMOTE_CODE` by
`dashboard.settings.apply_mesh_env`, which publishes a key by *truthiness*. The
lenient coercion path (the settings file, the environment, the CLI) returned the
value that had failed to be a boolean, so a non-boolean sat in a boolean slot
and was published as the literal `"1"` - the exact spelling
`policies.factory._check_trust_remote_code` accepts. A `settings.json` holding
`"maybe"`, `"yes-please"` or `{"a": 1}` therefore opened the
arbitrary-code-execution gate, while the same spelling handed to that gate
directly is refused and `update_strict` refuses it by name. The file path was
more permissive than the UI/API path for the one setting where that matters.

The module already stated the rule this broke: a lenient degrade must land on
the key's own SHAPE, "a list key that fell back to a scalar poisons every
comma-split consumer, which is worse than the empty default". That rule covered
the list keys and the four numeric keys and omitted the one boolean key.
`_BOOL_KEYS` now owns that family for both the strict and the lenient path - the
strict check spelled its own inline copy, so the two could disagree about which
keys are booleans - and the degrade is fail-closed: a gate that cannot read its
own setting stays shut. The spellings that do mean something (`true`, `1`,
`false`, `off`, `""`, `0`) are unchanged in both directions.

`docs/security.md` documented the environment-variable opt-in and never
mentioned that the dashboard setting writes that same variable, which is how
this was reachable without contradicting any documented behaviour; it now names
the setting and its fail-closed reading.

`tests/test_dashboard_settings_value_domain.py` pins both paths' answers across
the whole value domain as one table, so a schema key added without a domain is
visible. That domain was untested, which is why this reached `main`:
`dashboard/settings.py` statement coverage rises from 63% to 95%.
