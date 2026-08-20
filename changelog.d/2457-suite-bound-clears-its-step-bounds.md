### Fixed: the suite's job bound now covers every step bound inside it

`Test and Lint` is the one required check, and its 45-minute job bound had
stopped being compatible with the 12-minute bound on its own apt step. The two
were sized independently, months apart, and neither edit had occasion to read
the other.

Measured over 663 successful jobs in the 10 days to 2026-08-19: `Run tests`
costs at most 29.1 min, the steps carrying no bound of their own cost 4.4 min,
so a run in which apt spends its entire 12-minute allowance costs 45.5 min. That
run was reaped at 45 having done nothing wrong -- and a job-level reap renders as
a rollup `FAILURE` naming no step, which is exactly the diagnosis the step bound
was added to provide. The worst four successful jobs in the window were
44.4/44.0/42.0/39.6 min, and in each the excess is apt (12.4/11.9/10.5/9.1),
never the suite.

The job bound is now 60. `tests/test_workflow_jobs_are_bounded.py` asserts the
budget rather than describing it: the job bound must cover the sum of its step
bounds plus the measured suite ceiling and unbounded overhead, so a future edit
to either number that breaks the relationship fails instead of silently
narrowing the margin. The resampled band also replaces the stale 20-run table
the old bound was sized from, and the suite floor rises from 30 to 46, which had
sat below the observed maximum.

60 is also the guard's own ceiling, deliberately. The suite grew +0.95 min/day
across that window (p50 23.0 -> 32.7, R^2 0.93) as `tests/` gained 3054 test
functions, so this is the last raise available here and the next one has to be
argued as suite runtime instead.
