### Quality: a pull request's superseded checks no longer run ahead of its current head

A concurrency group is per workflow, so the required check cancelling its own
superseded run said nothing about the ten other workflows a pull request starts.
Five of them declared no group at all - `codeql.yml`, `dependency-review.yml`,
`breaking-change-check.yml`, `agent-api-check.yml` and `llm-input-safety.yml` -
and a push replaces the head sha they read, so each kept computing a verdict about
a commit no reviewer would open, on a runner the new head was queued behind.
Measured over the last 100 `pull_request` runs of each: 45 superseded runs and 67
runner-minutes, 48 of them CodeQL's.

They now use the group the other pull-request workflows already had.
`codeql.yml`'s also runs on `push` and `schedule`, so it keys the push side on the
commit and separates the events, leaving each pushed commit's analysis
uncancellable by the next merge or by the Monday scan.
`closing-reference.yml` keeps declaring none: it is the only workflow that can
have two runs on one head, where a cancelled context lands on a head that
satisfies the check. `tests/test_pull_request_concurrency_group.py` grades the
rule and records that exemption.
