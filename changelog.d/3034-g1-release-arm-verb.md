### Added: `g1_release_arm` verb for the driver's ExecuteAction(99) write

`G1Driver.release_arm` is the driver-side release-arm entry point: a caller
invokes the verb and the driver publishes the SDK's
`G1ArmActionClient.ExecuteAction(99)` call over the same DDS singleton
`ensure_dds` opens, using the release-arm action id (`99`) the SDK's
`action_map` reserves for "drop the arm-action hold and let the driver's
`send_action` path resume". The action-id lookup lives one file over in
`strands_robots.tools.g1.g1_arm_actions` where `_ARM_RELEASE_ACTION_ID` names
the number and the neon bundle's `g1_release_arm` verb (a single-purpose
`ExecuteAction(99)` wrapper) is called out as the verb the release-side
driver method will front (refs #2959). That lookup is the one source of
truth for the id; this verb does not re-name it.

The driver's method itself is not yet plumbed on `G1Driver` today (refs #358
for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a `release_arm`
accessor with a message naming the verb, the `driver` parameter and the
accessor. Once the driver method lands the same call returns the driver's
envelope verbatim — this is the same shape `g1_set_stand_height` (refs
#3031), `g1_set_swing_height` (refs #3032), `g1_start_task` and
`g1_send_action` already ship.

`strands_robots.tools.g1.g1_release_arm.g1_release_arm` is the agent-facing
side of that write: one duck-typed call on `driver.release_arm`, the
envelope the driver produced returned verbatim, and the same live-handle
refusals every write-side verb in this package owes (`driver` is `None`, a
robot *name*, or any object without a callable `release_arm`). The verb
adds no data-parameter refusals — the release-arm request has no caller-
facing shape beyond the driver handle, which is the one thing the neon
bundle's verb also asked for.

The FSM gate is not consulted here. The driver's `_check_motion_gates` is
the arm-write gate (`send_action` / `run_policy` / `start_task`); a release
is the *end* of the arm-write window a prior action opened, not a new
arm-write frame, so admitting a release under a widened gate (or refusing
one under a narrower gate) would leave the arm holding when the driver's
own release path already knows the exact frame the SDK accepts (refs
#358, #2916).

The SDK-load-hygiene contract every file under `strands_robots.tools.g1`
carries holds here too: `import strands_robots.tools.g1.g1_release_arm`
pulls no `unitree_sdk2py` submodule; the SDK loads only inside function
bodies (through the driver's own `release_arm` write path once it lands).

Contract-graded off the driver, no real `unitree_sdk2py` or DDS bus. Ten
tests: SDK-load hygiene, driver-side refusal round-trip, future success
envelope round-trip, three live-handle refusals (`None`, `str`, `int`),
single-call ordering, the SDK's `rc=7401` holding-code refusal round-trip,
and the SDK's `rc=7400` topic-busy refusal round-trip. Each cell names the
one shape a caller can rely on; none quote the driver's refusal wording
verbatim (refs #2874).
