### Security: `use_ros` gates every verb that can command a robot, not just `publish`

Publishing to `/cmd_vel`, calling `/emergency_stop` and sending a
`/navigate_to_pose` goal all move a robot, but only the topic `publish` path
consulted the safety-critical blocklist. `/navigate_to_pose` and `/follow_path`
are ROS 2 **actions** and the e-stop / motor-enable surfaces are usually
**services**, so 7 of the 12 blocklisted entries could not be enforced at all -
an agent asked to drive somewhere reaches for `action_send_goal`, which was
ungated. The gate is now keyed on the surface name and consulted from `publish`,
`service_call` and `action_send_goal` alike; reading a blocked surface (`echo`,
`info`, the `list_*` queries) stays ungated so telemetry is unaffected.

A blocklisted name is also compared in the form rclpy resolves it to, so the
unrooted `cmd_vel` and the trailing-separator `/cmd_vel/` no longer slip past a
literal membership test. Case is deliberately not folded: ROS 2 graph names are
case-sensitive, so `/CMD_VEL` is a different topic that no `/cmd_vel` subscriber
receives, and refusing it would block a legitimate surface without closing a path
to the robot. The gate now runs after the action's required arguments are
validated, so an operator is never prompted to approve a call that could not have
run. The headless pre-approval variable is `STRANDS_ROS2_COMMAND_ALLOW`, matching
the surfaces it now covers.

Everything that wraps `use_ros` has to forward an operator context or it inherits
the fail-closed path for every command it sends, so `RosBridgedRobot` now does:
`drive_<node>`, `stop_<node>` and `navigate_<node>` are declared
`@tool(context=True)` and hand the context through `drive()` / `stop()` /
`navigate_to()` into `use_ros`, and an agent driving a bridged robot is prompted
per command instead of refused. **Migration:** a *programmatic*
`robot.drive(...)` / `stop()` / `navigate_to(...)` has no operator to prompt, so
it now needs its surface named in `STRANDS_ROS2_COMMAND_ALLOW` (or
`BYPASS_TOOL_CONSENT=true`) - `/cmd_vel` and a Nav2 `nav_action` are both
blocklisted. Reads (`get_pose`, `get_scan`) are unaffected. `stop()` is gated
like any other publish to the same topic: the gate is keyed on the surface and
not on the payload, because "zero is harmless" holds for a `Twist` and not for
`/joint_command`, where zero commands motion to the zero pose - so a deployment
that must always be able to halt unattended pre-approves its `cmd_vel` topic
rather than relying on a carve-out.
