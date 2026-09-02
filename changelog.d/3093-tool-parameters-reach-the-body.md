### Fixed: an agent tool no longer advertises a parameter its body never reads

`lerobot_calibrate` exposed `format_output: str = "rich"` to the model,
described as `"Output format (rich, simple, json)"`, and never referenced it
again. The parameter reached the tool schema, so a model could choose a value,
report having set it, and change nothing. It was also not simply unimplemented:
an agent tool returns a fixed `content` list of typed blocks and every
`lerobot_calibrate` success path already emits both a `text` rendering and a
`json` one, so there was no format axis for the parameter to select. It is
removed rather than implemented.

`tests/tools/test_agent_tool_parameters_reach_the_body.py` closes the class for
all 87 `@tool` functions in the package, and joins the
`scripts/check_whole_tree_graders.py` roster. It grades a third direction its
sibling `tests/tools/test_agent_tool_parameter_descriptions.py` does not: that
guard requires every exposed parameter to carry a real description and every
`Args:` entry to name a real parameter, and a described-but-unread parameter
passes both. Because the new scan reads the tree with `ast` instead of
importing, it also covers the `tools/g1`, `tools/reachy` and `mesh` tool
families, which the sibling's non-recursive `pkgutil` walk does not reach.
