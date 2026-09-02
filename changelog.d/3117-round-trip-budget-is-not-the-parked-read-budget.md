### Fixed: a round trip that must complete is not timed by the budget for a read that must not

`tests/inference/test_a_failed_exchange_does_not_leave_the_connection_cached.py`
built its fixture client with a single `REQUEST_TIMEOUT = 0.3`, whose comment
stated two requirements without noticing they pull in opposite directions on one
number: short enough that a parked reply misses its deadline promptly, long
enough that an unparked round trip never does. Several scenarios in that file
park a reply server-side and assert the read misses its deadline; the same client
also serves the calls that have to *succeed*. One value cannot serve both, and it
was resolved in favour of the requirement whose cost is visible -- suite speed --
so `0.3` was a budget calibrated for the calls that must fail, applied silently
to the calls that must succeed.

That turned `main` red at `ee1fb7f`:
`test_a_successful_handshake_is_cached` makes one ordinary `get_actions_sync`
call and reported `TimeoutError: timed out in 0.3s`.

The tension is not real, because only one of the two budgets is ever waited out.
A successful call returns as soon as its reply lands -- the deadline is an upper
bound it never reaches -- so a generous value there costs a passing run nothing.
A missed read waits out its budget in full on every run, so that is the only one
with a reason to be small. The single constant coupled a budget that is free to
be generous to one that is not.

Measured on the round trip the 0.3s budget had to cover: 2.4ms median idle
(5% of the budget), 26.6ms median with cores oversubscribed 4x, and 654.6ms
median / 1851ms max at 24x. The nominal cost sits ~125x inside the budget, so the
failure is entirely tail latency under contention -- which is what a full suite
creates, and why this reproduces in CI and not on an idle checkout. On the exact
failing pattern under that load, the old budget breached 4 times in 40 trials;
the new ones breached 0 in 40.

`ROUND_TRIP_TIMEOUT` (5.0) now times the calls that must complete and
`PARKED_READ_TIMEOUT` (0.3) the one read per scenario that must miss, narrowed
around just that read and unwound in a `finally` -- callers keep using the client
after the strand returns, so a failed assertion inside the window would otherwise
leave the short deadline on it and relocate the flake into whichever test ran
next. The round-trip budget is generous but bounded: a genuinely hung call must
still report `TimeoutError` from the client rather than be killed by the suite's
`--timeout=120`, which reports nothing useful.

The concurrent-discard scenario keeps one shared budget
(`SHARED_DISCARD_TIMEOUT`, 3.0), because thread A's completing call and thread
B's missed read run on the same client and `_request` takes its deadline from
instance state with no per-call override. It is bounded on both sides: above a
round trip on a loaded runner, below the join that observes B time out.

`PARK_HOLD` names the hold that makes a missed read deterministic rather than a
race, and a new premise test pins the ordering the scenarios rest on:
`PARKED_READ_TIMEOUT < SHARED_DISCARD_TIMEOUT < THREAD_JOIN_TIMEOUT <
PARK_HOLD`. A budget that must expire but exceeds the hold gets its reply
*delivered*, so the scenario passes having asserted nothing about the discard it
names -- indistinguishable from a real pass. That relationship previously lived
only in the arithmetic between two unrelated literals, and the motive that
produced the original single budget applies just as well to shortening the hold,
so the guard is on the relationship rather than on any one value. It pins the
meaning of those scenarios, not their freedom from timing flakes: that a
completing call fits inside its budget depends on the runner and is not
statically assertable.

Test-only. The library's 60s default is a sensible deadline; these tests needed a
probe. Green-path cost is +2.76s, all of it the one concurrent test whose two
budgets provably cannot be separated.
