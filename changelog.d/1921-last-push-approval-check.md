### Docs: the last-push-approval deadlock is now surfaced by a check, not only described

`require_last_push_approval: true` on the `default` ruleset means an approval from
the account that pushed the head does not count. AGENTS.md has recorded that
mechanism since #1814 and refined it in #1852, and #1905 recorded it a third time
as a process issue. Nothing enforced or surfaced it, which is the shape the
changelog rule had before #1784.

What makes it worth a check rather than a third paragraph is that the state is
**invisible**. It presents as `reviewDecision: REVIEW_REQUIRED` with
`mergeStateStatus: BLOCKED` - byte for byte what a pull request nobody has
reviewed yet looks like - so no field a status sweep reads separates "waiting on a
reviewer" from "cannot merge without a second one", and the two need opposite
actions. It stood in eight consecutive scheduled scan summaries as "reviewer
bandwidth is the sole constraint" while #1722 and #1035 sat permanently
unmergeable.

`scripts/check_last_push_approval.py` reads the pusher from
`actions/runs?head_sha=<head>` -> `triggering_actor`, the one place it is legible,
and compares it against the accounts holding a current approval. #1920 sharpens
why commit metadata cannot stand in: its head was committed under the
`strands-robots` identity, which has no linked GitHub account, so
`commit.author.login` is `None` - there the metadata does not mislead, it declines
to answer, while #1722's names an identity distinct from the approver and reads as
satisfied when it is not. Opposite metadata, same verdict.

Three outcomes, because they ask for three different things: `pusher-only-approval`
is the finding; `awaiting-first-review` passes deliberately, being the ordinary
state of an open pull request, so that a red check means one specific thing rather
than appearing on every branch; and a pusher the lookup cannot attribute passes
rather than guessing from the commit. Verified against six live pull requests with
no false positive - #1722 and #1035 the finding, #1894 and #1920 (both merged)
satisfied, #1899 and #1901 awaiting review.

Only workflow runs whose event a push produces attribute a pusher. That filter
is load-bearing, not defensive: the check's own `pull_request_review` trigger
creates a run on the same head sha attributed to the *reviewer*, newer than any
run the push produced, so reading the newest run unfiltered named the approver as
the pusher and reported a deadlock on every approved pull request. It was caught
on #1921 itself, where GitHub read `APPROVED` / `UNSTABLE` and the check
disagreed - the six-PR verification above had passed only because no head sha
carried a review-triggered run until this workflow existed.

It reports and does not gate, and is absent from the required set on purpose.
Unlike `merge-base-overlap.yml`, whose remedy is self-clearing, this one's remedy
is a second reviewer and no work the author does turns it green. A gate a branch
cannot clear by doing anything is a report.

Tooling and documentation only; no production code or test behaviour changes.
Neither #1722 nor #1035 is unblocked by this - both still need an approver who is
not the account that pushed their heads.
