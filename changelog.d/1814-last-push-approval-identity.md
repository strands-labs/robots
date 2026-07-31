### Docs: pushing to a PR branch consumes the approval of whoever owns the token

The `default` branch ruleset sets `require_last_push_approval: true`, so the most
recent push must be approved by someone other than whoever pushed it. When this
agent pushes to a PR branch it uses `PAT_TOKEN`, and GitHub attributes the push
to the token's *owner* - so that account's approval stops counting, and no number
of further approvals from it can clear the PR. #1722 is in that state: one
current `APPROVED` review post-dating its head commit, all four review threads
resolved, `call-test-lint` green, and `reviewDecision` still `REVIEW_REQUIRED`.

The commit metadata asserts the opposite. `d938686`'s author and committer are
both `strands-robots`, an identity distinct from the approver, so reading the
commit list says the rule is satisfied. Nor is the pusher in the fields you would
check - `reviewDecision` `REVIEW_REQUIRED` and `mergeStateStatus` `BLOCKED` are
exactly what a PR with no approval at all looks like. It is legible in one place,
`actions/runs?head_sha=<head>` -> `triggering_actor`.

#1035 is the control: same author, same fork, same `strands_robots/mesh/` files,
one approval from the same account post-dating its head commit, threads clear,
checks green, no CODEOWNERS file in the tree. It differs in exactly one input -
its head commit was pushed by the contributor rather than by the approver - and
reads `APPROVED`. Every other rule in the ruleset is satisfied identically on
both PRs, so `require_last_push_approval` is the only one that accounts for the
difference.

`AGENTS.md` > PR Workflow > step 8 now records the mechanism, the one API field
that shows the pusher, and the consequence: prefer leaving a change for the
contributor to push so they stay the last pusher, and when the agent must push,
say on the PR that it now needs a second approver.

Documentation only; no production code or test behaviour changes.
