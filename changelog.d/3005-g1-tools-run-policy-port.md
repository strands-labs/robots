### Added: `g1_run_policy` verb over `G1Driver.run_policy`

Ports [`g1_run_policy`][gh-neon-g1] from `cagataycali/neon-the-g1` into
`strands_robots.tools.g1`. The verb starts the driver's 500 Hz control
loop against a caller-supplied policy - the loop the driver's
`send_action` writes one frame into and `stop_task` requests the exit
of - and returns the start envelope the driver produced verbatim.

The driver's `run_policy` already re-gates through `_check_motion_gates`
with scope `"motion"` on every step and publishes a zero-torque frame
before exiting if the FSM leaves the admission set mid-rollout (refs
strands-labs/robots#2916), so the verb is a thin duck-typed wrapper
that surfaces the driver's envelope through the `@tool` shape. The
verb refuses a `None` / non-callable / no-`.step()` `policy_object`
before the driver is called (four-invariant refusal envelope naming
the parameter and the remedy) and surfaces every one of the driver's
own refusal shapes verbatim (a `duration` / `n_steps` validation
refusal, a gate refusal, a "task already running" refusal).

`import strands_robots.tools.g1.g1_run_policy` still pulls no
`unitree_sdk2py` submodule (the package's SDK-load-hygiene contract,
refs strands-labs/robots#358).

[gh-neon-g1]: https://github.com/cagataycali/neon-the-g1
