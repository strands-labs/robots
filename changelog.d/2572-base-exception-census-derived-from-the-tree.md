### Changed

- **AGENTS.md**: the `py/catch-base-exception` census is derived from the tree instead of
  transcribed into the passage. Its handler table names each site as `path::function` and lists
  `strands_robots/` completely, and the handler count is gone in favour of the property the
  section's disposition rests on - every `except BaseException` handler re-raises lexically but
  one, the cross-thread marshal box. `tests/test_codeql_query_filters.py` now measures that
  property against the tree, so a second non-re-raising handler fails loudly rather than arriving
  as an unrecorded alert under a passage that reads as if it had a disposition for it.
