### Fixed: the merge-blocker report separates a held fork run from a head with no run

`scripts/check_merge_blockers.py` reported one outcome, `required-check-absent`,
for two situations that need opposite actions, and the remedy it printed was
correct for only one of them. Both are identical in every field the check read:
`mergeStateStatus` BLOCKED, a null-ish `statusCheckRollup`, `reviewDecision`
REVIEW_REQUIRED, and the required context absent from `check_conclusions`. One
field separates them and it was not read.

A fork run held at `action_required` is cleared by approving each run. A head
carrying no check suite at all is not: that is the shape a head commit written
through the API under the Actions `GITHUB_TOKEN` produces, whose events are
suppressed so a workflow cannot re-trigger itself, and it has no held run to
approve and no suite to re-run. The outcome's own text named "authorising or
re-running the workflow" for both, so on the second a reader following the
printed advice had nothing to click - and the detail said so out loud, stating
the ambiguity accurately and then leaving it unresolved at exactly the moment
one field would settle it.

The wrong guess is not symmetric, which is why these are now two outcomes rather
than one with a longer sentence. The cheap reflex on a null-ish rollup is the
close/reopen, and applied to a held fork run it is a no-op that looks like a
completed remedy: the runs already exist and stay held, so the flip re-queues
nothing while costing a closed/reopened pair on a contributor's pull request. The
reverse error is harmless by comparison, because approving runs that do not exist
is simply unavailable.

`resolve_check_suites` reads the head's suite census and `evaluate` splits on it:
suites present with at least one at `conclusion: action_required` keeps
`required-check-absent` and prints the approval; zero suites is the new
`check-suite-absent`, which says that neither authorising nor re-running is
available rather than merely unhelpful; and suites present with none held keeps
the older description, which is the one shape for which "never started" is
honest. An unread census keeps the older ambiguous wording and says the
observation was not made, so naming a remedy always follows from having looked -
and every existing caller classifies exactly as before. The census is
`tuple[str | None, ...] | None` rather than a count defaulting to zero for that
reason: `()` is a positive observation and "not read" is a third answer, and a
default would have silently picked one of the two remedies.

The approval remedy also names which field to read, because the obvious
confirmation is misleading. `action_required` is a *conclusion* on this surface,
never a status: a held run reports `status: "completed"`, so a client-side scan
comparing the `status` field matches none of them. Measured on a live pull
request carrying two held suites, a scan of the field matched zero while the
conclusion matched two. The query parameter is not the problem - `GET
/actions/runs?status=action_required` does accept a conclusion there and returned
every held run - so the remedy tells the reader to find the runs by `head_sha`
and recognise them by conclusion, which is the part a status comparison gets
wrong.

Both branches are pinned in `tests/test_merge_blockers.py`, which drives the
outcome table from fixtures, so the split needs no live pull request to grade.
One of those pins asserts that neither remedy appears in the detail for the state
it cannot clear, which is the property whose absence was the defect. Alongside
them, a derived guard now requires every outcome the report can emit to have an
owner in `_OWED_BY`, taken by dataflow from what `Blocker` is constructed with
rather than by spelling: a spelling rule misses the single-word `draft` and,
loosened to admit it, matches the single-word owner `nobody`.

Closes #2912.
