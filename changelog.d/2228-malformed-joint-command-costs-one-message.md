### Fixed: a malformed `joint_command` position no longer discards the commands queued behind it

`RosTelemetryBase._command_action` promises an action dict or `None`, but a
position `float()` could not read escaped as `ValueError`/`TypeError`. What that
cost depended on the transport: rclpy delivers one callback per `spin_once`, so
the raise cost that one message, while the cyclonedds `_poll_loop` calls
`take(N=10)` and has therefore already consumed a batch - the raise aborted the
dispatch loop, so every sample *behind* the malformed one was dropped with it.
Measured on a three-sample batch, one of two valid commands was applied.

The parser now reports the offending joint and value and returns `None`,
rejecting the message whole exactly as it already did for a length mismatch and
an out-of-range joint, so the failure costs itself on either transport.
Finiteness is unchanged: a `nan` position is a readable number that
`send_action` already refuses naming the joint.

Also drives the three cyclonedds poll-loop contracts its rclpy sibling has
pinned all along - a taken sample is dispatched, a reader failure costs only
that tick, and a second start spawns no second thread.
