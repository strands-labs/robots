### Docs: a cancelled check aggregates into a `FAILURE` rollup, and a rapid merge batch verifies only the tip

`AGENTS.md` > PR Workflow > step 8 said how to verify a pull request before and
after merging, but not how to read `main` afterwards. The obvious reading is
wrong in a way that invites reverting a good change: `statusCheckRollup.state`
reports `FAILURE` when a check was merely **cancelled**, and the checks UI draws
that the same red as a genuine failure.

`pr-and-push.yml` keys its concurrency group on
`github.event.pull_request.number || github.ref`. A push carries no pull request
number, so every push to `refs/heads/main` collapses into one group under
`cancel-in-progress: true` and each merge kills the run of the merge before it.
Four PRs merged inside 22 minutes left three consecutive commits (#1788, #1794,
#1796) each reporting rollup `FAILURE` whose only non-`SUCCESS` context was
`call-test-lint / Test and Lint` = `CANCELLED`, killed 1m07s, 15m00s and 5m38s
into their runs. Nothing had failed.

The timings also record a real cost rather than a misread: the suite had not
finished in 15m00s, so merging faster than it runs leaves only the tip verified
and no intermediate commit attributable. The new sub-bullet says to read each
context's own `conclusion` before believing the rollup, and to price a batch
knowingly - a red tip then costs a manual bisect.

Documentation only; no production code or test behaviour changes.
