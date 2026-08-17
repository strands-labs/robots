### Fixed

- **mesh/input**: a failed `get_teleop_events()` read is no longer published as `events: None`
  with no other trace. `null` is also what a leader arm with no event surface publishes, so a
  teleoperator whose event surface stopped answering dropped the operator's `terminate_episode`
  signal while joint commands kept flowing and `stats` reported a clean session. The read stays
  best-effort and never drops the frame the follower is tracking, but a failure now increments
  `InputPublisher.stats["event_read_errors"]` and logs a warning naming the device and the cause.
