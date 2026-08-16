### Fixed: a commit pushed to `main` runs its own suite to completion

`pr-and-push.yml` keyed its concurrency group on
`github.event.pull_request.number || github.ref` under `cancel-in-progress: true`.
Only the first operand is ever set, so the expression was two behaviours in one
line: per pull request, where cancelling a superseded run is the point, and - on a
push, which carries no pull request number - `refs/heads/main` for *every* commit.
One group for the whole branch, so each merge cancelled the previous merge's run
and the commit it was testing kept a permanently unfinished `call-test-lint`.

A cancelled context is not `SUCCESS` and one non-`SUCCESS` context drags
`statusCheckRollup.state` to `FAILURE`, so that commit read red with no failing
check anywhere in it. #1800 measured this on #1788/#1794/#1796 and concluded it
was tolerable to read around. Counting the branch is what changes the verdict: of
the last 25 commits on `main`, 24 had a settled rollup, 11 of those read `FAILURE`
and **9 of the 11 had no failing check at all**. The false reds arrive in bursts
because that is what the mechanism requires - three landed inside 55 seconds, two
more inside 7.

What makes it a wrong answer rather than a wrong colour is that reading rollups
along the branch is the only way to ask *when did `main` break*, and a burst
destroys the evidence for which commit in it broke `main` in the same act as
creating the fault: #2303 dated a breakage that way while two of the commits it
read were red only from cancellation. A branch's stale red clears on its next
push; a commit on `main` is immutable and already merged, so the wrong answer is
permanent.

The push side now keys on `github.sha`, leaving the pull-request operand
byte-identical. The two spellings the issue offered are not interchangeable and
the pin is written against the rendered group rather than the text for that
reason: keeping `github.ref` and gating `cancel-in-progress` on the event name
still leaves one group per branch, and GitHub holds at most one *pending* run per
group, so a third merge cancels the second while it is queued and reproduces the
context being removed.

A burst of N merges now runs N suites instead of 1, which is what buys each commit
its own answer, and it removes the older cost recorded in `AGENTS.md` step 8:
merging faster than the suite runs no longer leaves only the tip verified, so a
red tip no longer costs a manual bisect. `docs.yml` holds a per-commit `build` and
a shared-resource `deploy` under one ref-keyed group and so keeps a narrower form
of the defect; it is pinned as an explicit exemption with the decision tracked
separately.
