### Docs: a `null` reviewDecision is at least two states, and resolving the thread clears only one

`AGENTS.md` > PR Workflow step 8 documented `reviewDecision: null` as a third
reading meaning "one resolve from merging", recorded from #1974 where resolving
the sole unresolved thread moved `mergeStateStatus` to `CLEAN` and the decision
to `APPROVED`. That measurement is real, but the passage did not name the input
that made the resolve *sufficient*, and it explicitly told the reader the case
"needs no review at all".

#2328 presented the same signature - `MERGEABLE`, `BLOCKED`, `null`, one
unresolved `github-advanced-security` thread, `call-test-lint / Test and Lint`
`SUCCESS` - and settled the other way. `null` is also what a pull request with
no approving review at all reads, because a `COMMENTED` review contributes no
approval:

| pull request | approving review present | after `resolveReviewThread` |
|---|---|---|
| #1974 | one `APPROVED`, post-dating the head | `APPROVED` / `CLEAN` - merges |
| #2328 | none - every review `COMMENTED` | `REVIEW_REQUIRED` / still `BLOCKED` |

So the resolve was necessary on both and sufficient on one, and the two are
byte-identical in the field before it. This misreads in the reassuring
direction, which is the property step 8 already flags for `null`: following it
yields a resolve, a re-read expecting `CLEAN`, and a pull request reported as
ready while it waits on a first approving review - the presentation #1905
records for another cause.

Step 8 now carries the second row, prescribes reading the review set beside the
threads, and orders that read *before* the resolve. The ordering is the
load-bearing half: #2328's decision moved from `null` to `REVIEW_REQUIRED` on
the resolve, so the one value identifying which case a reader is in is destroyed
by the action the passage prescribes and cannot be recovered by re-reading. The
bullet also notes that `REVIEW_REQUIRED` itself carries two remedies - a first
approval, or a second account when the only approval came from the pusher -
which is the split `scripts/check_last_push_approval.py --all-open` reports.

`tests/test_review_decision_null_is_not_one_state.py` pins both halves: the
arithmetic, showing a verdict keyed on the decision and the threads alone is
wrong on one of the two recorded rows while one that also reads the review set
is right on both; and the prose, including a guard that the unqualified "needs
no review at all" claim cannot be restored without the qualification naming
#2328. Five of its ten assertions fail against the uncorrected file.

Documentation and test only - no behaviour changes.
