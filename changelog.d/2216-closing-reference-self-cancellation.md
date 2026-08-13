### Fixed: the closing-reference check no longer leaves a red context on a pull request that satisfies it

`closing-reference.yml` is the one workflow subscribed to a `pull_request`
activity type that cannot change the head sha (`edited`, which its own remedy
produces), so it is the one workflow that can be started twice on a single head.
Its concurrency group was keyed on the pull request number with
`cancel-in-progress: true`, so when those two runs overlapped one was cancelled -
and a cancelled check run is permanent, attaches to the live head, and aggregates
into `statusCheckRollup.state == FAILURE` from behind a same-named `SUCCESS`.

Measured on #1722 and #2205: every context on both heads was `SUCCESS` except a
cancelled duplicate of this check, and #2205's roll-up read `SUCCESS`, then
`FAILURE`, then `SUCCESS` across three reads of one unchanged sha. Whether it
happened was a race - #2204's two runs are 40s apart and both completed.

The workflow now carries no concurrency block, and no longer passes `PR_TITLE`:
with cancellation gone, two runs can overlap, and the payload's title outranks the
API's copy for the life of a run, so a `synchronize` run handed it would report a
pre-edit verdict after an `edited` run had already passed. Reading the title from
the API makes concurrent runs agree, and removes the interpolation seam that
passing author-controlled text existed to work around.
