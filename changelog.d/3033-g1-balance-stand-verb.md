### Added: `g1_balance_stand` verb for the driver's BalanceStand write

`G1Driver.balance_stand` is the driver-side BalanceStand entry point: a caller
passes a balance-mode id and the driver publishes `LocoClient.BalanceStand`
(which internally reaches the SDK's `SetBalanceMode` handler) over the same
DDS singleton `ensure_dds` opens. The Python SDK exposes `BalanceStand` as a
public `LocoClient` method that admits a small set of pre-programmed modes:
`0` (static balance, the default) and `3` (dynamic balance, from the neon
bundle's field notes against the real robot). The neon bundle's
`g1_balance_stand` verb (`cagataycali/neon-the-g1/tools/g1_posture.py`)
wrapped the call under a single-writer lock and coerced the argument through
`int(...)` before dispatch; the read-only half of that envelope already landed
as `strands_robots.tools.g1.g1_balance_modes` (refs #358), and this module is
the write-side companion that hands the target to the driver.

The driver's method itself is not yet plumbed on `G1Driver` today (refs #358
for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a `balance_stand`
accessor with a message naming the verb, the `driver` parameter and the
accessor. Once the driver method lands the same call returns the driver's
envelope verbatim — this is the same shape `g1_set_fsm` (refs #3025),
`g1_set_stand_height` (refs #3031) and `g1_set_swing_height` (refs #3032)
already ship.

`strands_robots.tools.g1.g1_balance_stand.g1_balance_stand` is the
agent-facing side of that write: one duck-typed call on
`driver.balance_stand`, the envelope the driver produced returned verbatim,
and the same live-handle refusals every write-side verb in this package owes
(`driver` is `None`, a robot *name*, or any object without a callable
`balance_stand`). The verb adds four data-parameter shape refusals on top —
a `None` `balance_mode`, a `bool` payload (which would coerce to a silent
`0` or `1`), a `float` payload (which the neon bundle's own `int(...)` would
silently truncate to a mode the caller did not name), and a non-integer
`str` shape — using inline `isinstance` shape checks rather than a shared
validator because the value-domain here is an integer mode id with a
neon-bundle-observed admitted set `{0, 3}`, and no existing shared validator
in `strands_robots.utils` fits that shape (`positive_count_error` refuses
`0`, which is the static-balance default; `finite_number_error` admits
floats the SDK's handler cannot use). The in-set admission `{0, 3}` itself
is not enforced by this verb — the module docstring names "does not refuse
a `balance_mode` outside the admitted set" as one of the things this verb
does not do, because refusing an unlisted mode here would fork the neon
bundle's admission set into a second source of truth this module would then
have to keep in sync with the read-only envelope `g1_balance_modes` module.
Importing the module pulls no `unitree_sdk2py` submodule (the package's
SDK-load-hygiene contract, refs #358), and the module docstring names the
six things this verb does not do (refuse an out-of-set mode, encode the
`BalanceStand` dispatch, decode the SDK's `rc`, restate the driver's
refusal wording, schedule the FSM 801 prerequisite, check the `WALK_FSMS`
gate) so a caller reading it does not misread the surface.

The test suite
(`tests/drivers/test_g1_balance_stand_writes_the_driver_envelope.py`) grades
sixteen shapes: the SDK-load-hygiene pin, two driver-side envelopes as
pass-through (a driver-side refusal, a future success envelope), the two
live-handle refusals through the shared `live_handle_refusal` guard, four
data-parameter shape refusals (missing `balance_mode`, non-integer `str`
`balance_mode`, `float` `balance_mode`, `bool` `balance_mode`), four
admitted-value pins (a `0` static-balance target, a `3` dynamic-balance
target, a `7` out-of-set target passed through unchanged, and a negative
value passed through unchanged - all four reach the driver verbatim because
the module docstring names "does not refuse a `balance_mode` outside the
admitted set" as one of the things this verb does not do), one exactly-once
write pin, and one arguments-pass-through pin. The universal auto-discovery
test at `tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py`
grades the verb the moment it lands (its first parameter is `driver: Any`),
so the two live-handle refusal rules are held by that suite too.
