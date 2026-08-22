### Fixed: an uncomputed mergeability is no longer reported as a clean merge

`scripts/check_merge_blockers.py` modelled `mergeable` as `bool | None` but
classified only two of those three states. GitHub computes mergeability on
demand and returns null while it works, and a merge into the base invalidates
the cached value for every open pull request -- so a sweep run just after a
merge is exactly when the null appears. Falling through the conflict branch
reported it as if the branch merged cleanly, and the rules underneath it then
owned the next action: #1035 was measured reporting `pusher-only-approval`,
owed by a reviewer, while it was in fact `CONFLICTING`/`DIRTY`. An otherwise
satisfied pull request in the same state reported `no-unsatisfied-rule`, whose
printed remedy is to attempt the merge.

The third state is now named: a `merge-state-unknown` outcome, owed by nobody,
carrying the re-read that resolves it. It is reported as *gating*, so an
approval underneath it reads as necessary but not sufficient rather than as the
next action, and it is deliberately not a finding, so the exit status keeps its
meaning. A null is still never reported as a conflict.
