### Docs: a `mergePullRequest` error is not a verdict on whether the squash landed

`AGENTS.md` > PR Workflow step 8 already warned that a `mergePullRequest`
mutation can report `Pull Request is not mergeable` for a merge that succeeded.
It read as a rare edge case, so the documented remedy - confirm with
`state`/`merged` before redoing the work - looked optional, and the reflex it
competes with is a retry.

The error is uninformative rather than rare. #2249 and #2250 were squashed
thirty seconds apart, each by a single call carrying `expectedHeadOid` - so a
stale oid is ruled out - against a pull request reading `CLEAN` / `MERGEABLE` /
`APPROVED` with every required context `SUCCESS`. Both calls returned
`Pull Request is not mergeable` beside `mergePullRequest: null`, and both
squashes were already on `main`: `926beb9` at 19:24:50 and `07a759d` at 19:25:20.
That is three for three with #1756's `4bf139c`, and the payload carries no field
separating a refusal from a success, so only the read-back can tell you which one
you got.

The read-back also names the likely cause, in the field the error is worded
about: after the merge #2249 reports `mergeStateStatus` and `mergeable` as
`UNKNOWN`, consistent with the mutation re-reading a pull request it has just
closed. That makes the retry the expensive reflex rather than the safe one - a
second call against #2249 after it had merged returned the identical error beside
the identical `null`, so retrying manufactures a second confirmation of a failure
that never happened.

`tests/test_merge_mutation_error_is_not_a_verdict.py` derives the wrong verdict
from each recorded payload, so the arithmetic the guidance forbids is executed
rather than described, and pins the prose adjacency separately so the correction
cannot drift away from the instruction it qualifies.
