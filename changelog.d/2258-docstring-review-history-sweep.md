### Docs: describe the code rather than the review that produced it

Docstrings and comments across `strands_robots/` carried review-history
markers -- `reviewer caught X`, `pre-#164 behaviour`, `(post-PR #101)`,
`requested by @handle during review`, `variant-B` -- that name a conversation
a downstream reader cannot open and a sequence they were never part of. 48
edits replace each one with the fact it was standing in for: the fallback
`pre-#164` described is now named as "when RoboSuite is absent", the dataclass
field documented as `(post-PR #101)` now says when it is populated, and a
"well-known kwargs" contract now points at the documentation instead of an
issue number.

Executable content is unchanged: every touched module's docstring-stripped
AST digest is byte-identical before and after. A new guard keeps the markers
out, and admits a pull-request reference only where a reader can follow it --
an upstream project this package declares as a dependency, or an `AGENTS.md`
section whose heading is verified to still exist.
