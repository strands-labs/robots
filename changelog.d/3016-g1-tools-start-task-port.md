### Added: `g1_start_task` verb over `G1Driver.start_task`

Ports [`g1_start_task`][gh-neon-g1] from `cagataycali/neon-the-g1` into
`strands_robots.tools.g1`. The verb is the provider-registry entry point
to the driver's 500 Hz control loop, sitting alongside `g1_run_policy`
(which starts the loop against an already-built policy) and
`g1_send_action` (which writes one arm-SDK frame). The provider
registry in `strands_robots.policies` is not yet plumbed to
`G1Driver`, so the driver's `start_task` runs the FSM/battery gate
and refuses with a fixed `start_task: provider registry not wired yet;
use run_policy(policy_object=...) to drive the control loop today`
message; the verb passes that envelope through unchanged.

The driver's `start_task` re-runs the same `_check_motion_gates`
gate `run_policy` and `send_action` route through (refs
strands-labs/robots#2916), so the verb is a thin duck-typed wrapper
that surfaces the driver's envelope through the `@tool` shape - the
today-shape registry-not-wired refusal, a gate refusal if the FSM
left the arm-SDK admission set first, and (once the registry lands)
the loop-start envelope `run_policy` produces today. The verb refuses
a `None` / wrong-type `driver` before the accessor is called (four-
invariant refusal envelope naming the parameter and the remedy)
through the shared `live_handle_refusal` guard that every live-handle
verb in this package already routes through.

`import strands_robots.tools.g1.g1_start_task` still pulls no
`unitree_sdk2py` submodule (the package's SDK-load-hygiene contract,
refs strands-labs/robots#358). The five arguments the driver's method
takes (`instruction`, `policy_port`, `policy_host`, `policy_provider`,
`duration`) reach the driver verbatim; a caller who names only
`driver` reaches the driver with the same defaults the driver's
method would have filled on its own.

The test suite (`tests/drivers/test_g1_start_task_writes_the_driver_envelope.py`)
grades the three envelope shapes the driver produces today or in the
future (the registry-not-wired refusal, a gate refusal, a future
success envelope matching `run_policy`) as pass-through, the two
live-handle refusals through the shared guard, "the verb writes the
driver exactly once", "the verb passes the five arguments through
unchanged", and the default-parity check. The universal live-handle
auto-discovery test at
`tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py`
grades the verb the moment it lands (its first parameter is
`driver: Any`).

[gh-neon-g1]: https://github.com/cagataycali/neon-the-g1
