### Added: a check for a pull request whose recorded head is not its branch's tip

`scripts/check_pr_head_is_current.py` compares `pullRequest { headRefOid }`
against the tip of the branch in the head repository, per pull request or across
the whole open set with `--all-open`.

The two are meant to be the same commit and can disagree for hours. #2508 sat
approved, green and unmergeable for over five hours recording `21ea097e` -
pushed 07:56:47, eight green check suites - while its branch tip was `271ec912`,
pushed 13:58:16 and carrying no check suite at all. Every gate resolves the head
through the pull request's own view of it, so `reviewDecision` read `APPROVED`,
`call-test-lint / Test and Lint` read `SUCCESS`, no thread was unresolved, and
both existing `--all-open` sweeps reported no finding - each of them correctly,
about a commit that was no longer the tip.

Both merge APIs refuse such a pull request with `Head branch is out of date`,
which names a gate this repository does not have: the `default` ruleset sets
`strict_required_status_checks_policy: false` and `main` carries no classic
protection. The one field that moves, `mergeable` at `UNKNOWN`, is also the
weakest available signal, since a first `UNKNOWN` is the documented and benign
result of lazy computation - so "read it again" is indistinguishable from the
ordinary case for as long as one keeps reading.

The check reads the head repository's ref rather than the pull request's own
answer, because that answer is the value under suspicion. It reports and does
not gate: the remedy is a close/reopen, which reconciles the record without a
commit and therefore without spending the approval, and is not something a
branch author clears by pushing.
