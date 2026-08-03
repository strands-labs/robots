### Quality: the required check is no longer cancelled by events that change no code

`pr-and-push.yml` was the only workflow overriding `pull_request`'s `types`, and
three of the six it listed - `ready_for_review`, `review_requested`,
`review_request_removed` - cannot change `pull_request.head.sha`, which is the ref
it hands to the reusable test-lint workflow. Because it also cancels its own
in-flight run for the same pull request, each of those events discarded a run of
the repository's one required check and started an identical one over the same
commit.

Measured on #1899, #1901 and #1902, where two reviewers were requested seconds
after the pull request opened: every sibling check ran once on the head sha while
this workflow ran three times, leaving each sha with both a CANCELLED and a
SUCCESS `call-test-lint / Test and Lint`.

That a cancelled check aggregates into a `FAILURE` roll-up is #1800's finding and
is already in `AGENTS.md`; that entry describes the other producer of it - pushes
to `main` sharing one concurrency group - which is left alone, because a
superseded `main` run is of no interest. This producer discards work for no
possible gain, and the aggregate it leaves behind is not merely red but unstable:
the same three shas read `FAILURE FAILURE FAILURE` and then
`FAILURE SUCCESS SUCCESS` ten minutes later with no new run in between, which no
reading discipline covers. The ~20s each cancellation actually cost is
incidental; the same event arriving at minute 25 of a 27-minute run discards
all 25.

The `types` key is removed rather than shortened, because GitHub's default is
exactly the three sha-changing events, so the fix leaves one definition instead
of a second copy - and it makes this workflow's trigger shape identical to the
seven `pull_request` workflows that never overrode it.
