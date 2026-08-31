### Added: `g1_stop_task` signals `G1Driver.stop_task` and reports the join

`G1Driver.stop_task` signals the driver's running control loop to exit,
joins its thread, and lets the loop publish
`_build_zero_torque_lowcmd` on the way out (a soft *controlled* stop
rather than a Disable that would let the named joints fall freely).
Idempotent: no running task returns a success envelope naming the state,
and a join that outlasts the budget surfaces as `status="error"` with the
snapshot's `stopped=False` in the payload, so a caller reading only
`status` cannot count the task as stopped while the loop is still writing
frames.

This verb is the agent-facing side of that write. It calls the driver's
method once, unwraps the envelope (either the `content[0]["text"]`
no-task sentinel or the `content[0]["json"]` snapshot dict) and reshapes
it into a flat dict carrying `status` / `present` / `stopped` /
`running` and every field `_ControlLoop.snapshot` writes: `steps`,
`refusals`, `elapsed_s`, `duration_budget_s`, `n_steps_budget`,
`exit_reason`, `exit_detail`, `hz`, `fsm_refresh_hz`, `fsm_reads`. On
the "no task is running" shape `reason` quotes the driver's own text
verbatim.

The verb is the write-side sibling of `g1_get_task_status`
(`strands-labs/robots#2955`) - together they let an agent poll the loop's
snapshot on any thread and, when a supervisor decides to halt it, request
the controlled stop and hear back the honest join outcome.  Refs
`strands-labs/robots#358`.
