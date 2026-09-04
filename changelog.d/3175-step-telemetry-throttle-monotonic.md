### Fixed

- A simulation rollout's per-step mesh telemetry keeps streaming across a
  wall-clock correction. `MuJoCoSimEngine._make_run_policy_hook` rate-limits
  `Mesh.publish_step` against the period from
  `strands_robots.mesh.session.stream_min_period_from_env`, and measured that
  elapsed interval on `time.time()`: a backward step -- an NTP correction, a
  `date -s`, a resume from suspend -- landing between two publishes made the
  difference negative, so the throttle refused every later step of the rollout
  until the date caught up. A real 0.9s rollout at 50 Hz lost 10 of its 15
  publishes to a 2s step and still reported `status="success"`, and the
  publishes that did land stayed correctly spaced, so nothing distinguished the
  shortfall afterwards from a rollout that simply ran for less time. The
  interval is now read from `time.monotonic()`, and the "never published"
  sentinel is `-inf` rather than `0.0` so the first step of a rollout is due
  wherever the platform's monotonic epoch sits. Because a sentinel below every
  reading makes the throttle's subtraction `inf` on that first step -- which
  clears even the infinite period that spells `STRANDS_MESH_STREAM_HZ=0` -- the
  hook now reads the period itself rather than relying on the subtraction, so
  the operator's opt-out still publishes nothing at all. The hardware control
  loop throttles the same publish on the same period and already read this
  clock; the absolute `TrajectoryStep.timestamp` the same hook writes stays on
  the wall clock.
