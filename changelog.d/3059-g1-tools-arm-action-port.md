### Added: `g1_arm_action` verb for the driver's arm-gesture write

`G1Driver.arm_action` is the driver-side arm-action entry point: a
caller invokes the verb with an action name (`"clap"`, `"heart"`,
`"two-hand kiss"`) or a numeric `action_id` and the driver publishes
the SDK's
`G1ArmActionClient.ExecuteAction(id)` call over the same DDS singleton
`ensure_dds` opens, using the id the SDK's `action_map` reserves for
that gesture. The name-to-id table already lives one file over on
`g1_arm_actions._ARM_ACTION_MAP` (refs `strands-labs/robots#2959`) and
`g1_list_arm_actions` surfaces it, so this verb does not re-name the
map; it hands `action` and `action_id` to the driver's method
verbatim and returns the envelope the driver produced.

The driver's method itself is not yet plumbed on `G1Driver` today
(refs `strands-labs/robots#358` for the SDK-facing gate work the
write belongs on); `live_handle_refusal` refuses a handle without an
`arm_action` accessor with a message naming the verb, the `driver`
parameter and the accessor it read for, and once the driver lands
the same call returns the envelope the driver wrote verbatim. This
is the same shape `g1_release_arm` (refs
`strands-labs/robots#3034`), `g1_send_action` (refs
`strands-labs/robots#3004`) and `g1_start_task` (refs
`strands-labs/robots#3016`) already ship.

The verb adds two refusal envelopes at the tool layer instead of
raising: a live-handle refusal (`driver` is `None`, a robot *name*,
or an object without the accessor), and a name/id refusal when
neither `action` nor `action_id` is passed. The neon bundle's
`g1_arm_action` verb accepted a default empty `action=""` and would
reach the SDK's map with the empty string; this port refuses the
shape at the tool layer so a caller who reached the verb with
neither parameter set sees a message naming both parameters and the
remedy (call `g1_list_arm_actions` to see the map), rather than a
driver-side refusal that names only the id it received. The
driver's own refusals - the topic-busy code (`rc=7400`) from
concurrent writers on `rt/armsdk`, the FSM/battery gate that
`_check_motion_gates` writes with scope `"arm"`, the SDK-side raise
- round-trip through this verb verbatim (refs
`strands-labs/robots#2874`).

`import strands_robots.tools.g1.g1_arm_action` still pulls no
`unitree_sdk2py` submodule (the package's SDK-load-hygiene contract,
refs `strands-labs/robots#358`). The neon bundle's own
`time.sleep(hold_seconds)` + `ExecuteAction(99)` release path is
*not* ported here: the hold is a concern of the write scheduler
(the caller's own timer, or the driver's `run_policy` loop), not of
a one-frame write. A caller who wants a hold-then-release calls
`g1_release_arm` after the hold from the same schedule that fired
this verb.
