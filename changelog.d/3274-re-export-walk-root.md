### Fixed: a grader deriving its walk root from a re-exported symbol is refused rather than silently unrostered

`AGENTS.md` asks a whole-tree grader to derive its walk root from an imported
symbol, and states the one spelling that does not resolve: a symbol imported
from a module that only re-exports it, where the import does not say which file
the symbol came from. Nothing refused a site that broke the rule, and the
failure is silent in the reassuring direction -- the root resolves to nothing,
so the module drops out of the preflight roster entirely and
`scripts/check_whole_tree_graders.py` reports a clean run over the graders it
can see while collecting none of that one.

`tests/policies/test_service_port_domain.py` broke it, deriving the scan root
for `TestNoProviderShipsAnUnguardedPort` from `create_policy` imported out of
`strands_robots.policies` rather than out of
`strands_robots.policies.factory`, which defines it. It now imports from the
defining module; `inspect.getfile` answers the same object either way, so the
files the grader scans do not change.

`tests/test_whole_tree_graders_roster_is_complete.py` gains a cell that screens
every module under `tests/` for the shape and names the module to import from,
so the rule is enforced rather than only written down. The resolver's half was
already pinned on planted source; a correct resolver cannot keep the shape out
of the tree, which is what this cell adds.

`check_whole_tree_graders.module_file` and
`check_whole_tree_graders.MODULE_FILE_FUNCS` are public, so the pin screens with
the derivation's own resolver instead of restating what "the module that defines
it" means.
