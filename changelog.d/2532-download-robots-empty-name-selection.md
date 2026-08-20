### Fixed: an empty robot-name selection is refused, not read as "every robot"

`names` on `download_robots` selects a subset of the sim robots the registry
lists, and `None` is the documented "all of them". It was read by truthiness, so
`names=[]` -- a selection that asks for no robot -- fell through to the branches
that never read it. Measured on the shipped registry it downloaded 56 robots on
its own, and 13 (the whole `humanoid` category) when a `category` was also
passed, reporting either count as the caller's own request.

Nothing had to write `[]` to reach that. The `download_assets` tool builds the
list from a comma-separated string through its own blank-field filter, so the
non-empty `robots=","`, `" "`, `",,,"` and `"  ,  "` each parsed to zero names --
and an agent that emitted one of them cloned the entire registry and was told
`status: success`.

The selector is now read `is None`, and an empty selection is refused ahead of
`get_user_assets_dir()` -- the call that creates the asset cache directory -- so
a refused selection leaves nothing behind. The tool refuses in its own
vocabulary, naming `robots=` rather than the `names=` its caller never passed; an
absent or empty `robots=` still means "all", because for a single string argument
unset and empty genuinely coincide.

Only the emptiness verdict is taken locally. The shape is deliberately not routed
through the shared `name_list_error` domain: this surface resolves each name by
membership into a dict, so a repeat resolves to its first occurrence and a mapping
and a one-shot iterator are each read exactly once. All three are honored as
written today, and the tests pin them so that stays deliberate.
