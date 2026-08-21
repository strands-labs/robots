### Added: a blocked pull request's report names the rule it left unsatisfied

`scripts/check_merge_blockers.py` reads the branch ruleset and names every rule a
pull request leaves unsatisfied, together with the party who can clear it -- a
conflict, an unresolved review thread or a failing check (the author), a missing
approval (any reviewer), an approval only its own pusher supplied (a different
reviewer), a required check held at `action_required` (a maintainer), a check
still running (nobody), or no unsatisfied rule at all, which means the state is
stale and the merge is worth attempting.

`mergeStateStatus: BLOCKED` is one word for all of these and names none of them,
so two approved and green pull requests sat idle after approval on obligations
their own authors could have cleared. Takes `--pr` for one pull request or
`--all-open` to sweep the queue; composes `check_last_push_approval.py` so what
counts as a current approval keeps one owner, and gates nothing.
