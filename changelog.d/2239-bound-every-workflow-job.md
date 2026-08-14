### Fixed: every workflow job that runs on a runner is bounded, so a hung step cannot hold a runner silently

A GitHub Actions job with no `timeout-minutes` inherits a 6-hour default, and no
job in this repository set one -- including `Test and Lint`, the required check
every merge waits on. On #2236 that job read `IN_PROGRESS` for 103 minutes while
pytest had in fact printed `1 failed, 22095 passed` 18 minutes in and then
refused to exit; the run ended because a human cancelled it, not because
anything bounded it. A red verdict presented as "still running", which is the
one direction that misleads a reviewer into waiting.

Each job now declares a bound sized from observed durations over the last 300
runs: 45 min for the suite (measured band 19-27, max 26.5 over 20 runs), 20 for
CodeQL, 10-15 for the gates. Two structural exemptions, both asserted rather
than assumed: a job that calls a reusable workflow cannot carry the key at all
(the bound lives in the callee, which is checked to have one), and a job gated
on a deployment `environment:` can be waiting on a human. Pinned tree-wide by
`tests/test_workflow_jobs_are_bounded.py`.
