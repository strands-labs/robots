### Added: fleet failover and degraded-ops example (peer loss + dispatcher down)

`examples/fleet/03_failover_and_degraded_ops.py` drills the two failure modes
a fleet orchestrator must survive. Part 1: a robot's heartbeat dies
mid-transport, the orchestrator observes the presence timeout, closes that
robot's rollout bookkeeping, and re-dispatches the remaining legs to a peer
chosen by the shared `capabilities.py` hard-constraint filter - a capability
match, never a name match, and a reply failure from a robot that is still
heartbeating is a structured task failure rather than a failover. Part 2: the
orchestrator peer dies, robots keep answering peer to peer, `emergency_stop`
still propagates over `strands/safety/estop` with the dispatcher dead, and the
restarted orchestrator re-syncs from presence plus the signed audit log. The
smoke test asserts the estop-during-dispatcher-outage claim through the real
safety handlers with no dispatcher object in existence. (#2182, epic #2179)
