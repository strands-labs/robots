### Added: a remote LeRobot policy reconnects when its server comes back

When the policy server goes away mid-run, the robot holds its last target and the session is reopened in the
background with backoff, for `reconnect_s` (default 10 s), instead of the run ending with a fault at once. The new
session must serve the same checkpoint (LeRobot's `expected_artifact` check) over the same pinned channel with the same
features, semantics, rate and execution; a different deployment at that address ends the run with a fault naming what
changed. `policy.reconnects` lists each recovered drop. `reconnect_s=0` keeps the old behaviour.
