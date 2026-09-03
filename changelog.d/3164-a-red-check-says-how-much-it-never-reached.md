### Fixed: a red required check says how much of the suite it never reached

The one required check runs the suite under `-x`, so it stops at the first
failure. That is deliberate, but it means **a red `call-test-lint` is not a
count**: it names one failing cell, and the number of failing cells stays
unknown until the run is repeated on a tree where that one passes.

Measured on run 33690980247, the `Test and Lint` job of #3161 at `48881ea`:
`collected 46583 items`, then `1 failed, 34268 passed, 278 skipped`. So 34547
items executed and **12036 -- 25.8% of the suite -- never ran**. The failure
that was hidden was neither hypothetical nor distant: the next one was four
lines below the first, in the same class, with the same cause, so the
truncation turned a one-round fix into two. `--cov-fail-under=80` is no
backstop either -- that same truncated run reported
`Total coverage: 81.54%` and the gate read as a pass.

pytest does print a `stopping after 1 failures` banner, so this was not an
unrecorded event. What was missing was the subtraction: the banner does not say
how much was skipped, the `collected` line it needs sits **34,927 lines
earlier** in a 6.4 MB log, and neither number reaches a surface short of
downloading that log.

`scripts/report_truncated_test_run.py` now pairs those two numbers and writes
the result to the job summary and to a check annotation, so a reviewer reads
"12036 never ran, the failure count is a lower bound" without opening the log.
It reports three outcomes -- `complete`, `truncated`, and
`incomplete-no-summary` for a run killed before pytest finished reporting --
and returns 0 for every input including an unreadable one, because it runs
inside the required check's own job where a nonzero exit would either duplicate
the suite's own red or, on a parsing bug, invent one on a green tree.

**`-x` itself stays, and that is a measurement rather than a preference.**
Dropping it would make a red run cost what a green one costs, and a green one
is already close to the bound: over the last 40 `main` pushes the `Test and
Lint` job took 32.6 to 57.5 minutes on the 37 runs that completed, median 46.9,
against `timeout-minutes: 60` -- with run 33631903156 already reaped at 60.12.
The comment on that bound records that a further raise cannot be spent there
(#2457, #2239), so letting a red run continue would put it in reach of the
reap. Describing the truncation is the change that costs no runner time at all.

Capturing the output needs a pipe, and the pipe brings its own hazard worth
naming: the default shell for `run:` is `bash -e`, which does **not** set
`pipefail`, so a pipeline exits with the status of its last command -- `tee`,
which always succeeds. Without `set -o pipefail` a failing suite would report
SUCCESS on the one required check, and nothing else in this repository reads
the suite's verdict independently. That invariant is pinned rather than left to
review.

Closes #3164.
