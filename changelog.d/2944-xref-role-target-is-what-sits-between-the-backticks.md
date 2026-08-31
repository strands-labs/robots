### Fixed: a cross-reference role naming a file is reported rather than exempted

`strands_robots.policies.moveit2.policy`'s module docstring pointed a reader at
the sidecar deployment with a `:mod:` role whose target was
`strands_robots.policies.moveit2.server.docker-compose.yml` - a fully-qualified
target naming a YAML file. No module answers to that dotted path, so Sphinx
rendered a plain-text token and an IDE could not follow it: the dead pointer
`tests/test_docstring_xref_roles_resolve.py` exists to catch. The compose file
is real, so the citation's intent was sound; it is now named the way the sibling
`server` package's own docstring names it, as a plain code span.

That guard graded the role one line above it and never saw this one. Its target
pattern admits whitespace so a role wrapped over a line break is reported rather
than failing to match and exempting itself - the module docstring states that
reasoning - and the same pattern excluded every other character a wrong target
carries, so a hyphen stopped the match before the closing backtick. The pattern
now admits any non-backtick character after the leading identifier: the
backticks delimit the target, so whatever sits between them is the target as
written. The leading identifier is still required, which keeps a relative role
(`:mod:`.protocol``) out of scope, since that resolves against Sphinx's
current-module context rather than here.
