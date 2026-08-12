# Fleet orchestration examples

Multi-robot orchestration on top of the pieces the rest of `examples/`
introduces one at a time: multi-robot worlds, the Zenoh mesh with its safety
protocol, capability-based dispatch, and the signed audit log. Tracked by epic
[#2179](https://github.com/strands-labs/robots/issues/2179).

| # | Example | Status |
|---|---|---|
| 01 | `01_skill_dispatch_multi_vendor.py` - capability-based dispatch across heterogeneous robots | tracked by [#2180](https://github.com/strands-labs/robots/issues/2180) |
| 02 | `02_cross_zone_transport.py` + read-only Rerun fleet dashboard | tracked by [#2181](https://github.com/strands-labs/robots/issues/2181) |
| 03 | [`03_failover_and_degraded_ops.py`](03_failover_and_degraded_ops.py) - peer-loss reassignment + dispatcher-down safety | here |
| 04 | `04_emergency_evacuation.py` - three-phase evacuate protocol, benchmark-scored | tracked by [#2183](https://github.com/strands-labs/robots/issues/2183) |
| 05 | `05_work_order_dispatch.py` - structured task ingress mapped onto per-site capability manifests | tracked by [#2185](https://github.com/strands-labs/robots/issues/2185) |

Shared across the suite:

- [`capabilities.py`](capabilities.py) - the per-robot, per-site capability
  manifest schema (`{robot, site, skills: [{name, payload_kg, fixture,
  zones}]}`) plus the deterministic hard-constraint filter. Consumed by
  examples 01, 03 and 05.

## 03 - failover and degraded operations

Two failure drills on one fleet (a quadruped and a wheeled base that can both
transport, plus an arm that cannot - so the failover is demonstrably a
capability match, not a name match):

1. **Robot failure mid-task**: a 4-leg transport loses its robot's heartbeat
   after leg 1. The orchestrator observes the presence timeout
   (`PEER_TIMEOUT`, `strands_robots.mesh.session`), closes that robot's
   rollout bookkeeping, re-runs the `capabilities.py` filter over the
   surviving manifests, and re-dispatches the remaining legs - including the
   unconfirmed one, so delivery is at-least-once. A failed reply from a robot
   that is *still heartbeating* is a structured task failure instead: presence
   is the failover trigger, never a failed reply on its own.
2. **Dispatcher failure**: the orchestrator peer dies. The robots' local
   control loops keep answering `status` peer to peer, and `emergency_stop`
   issued by a surviving robot still propagates over `strands/safety/estop` -
   safety does not depend on the dispatcher. The example *asserts* both halves
   (status answered, execute refused) and raises if either fails. The
   orchestrator then restarts and re-syncs from the two things that survived
   the outage: the presence registry and the signed audit log.

```bash
# No simulator, no mesh - scripted presence + loopback transport:
python examples/fleet/03_failover_and_degraded_ops.py --dry-run

# Live: three MuJoCo worlds (go2 + lekiwi + so101), one mesh peer per robot,
# both drills end to end (the presence timeout is honoured in full, twice):
python examples/fleet/03_failover_and_degraded_ops.py

# Signed audit trail:
STRANDS_MESH_AUDIT_PSK=demo-psk python examples/fleet/03_failover_and_degraded_ops.py --dry-run
```

Two live-mode notes. The estop drill leaves the surviving robots in safety
lockout by design; resuming needs the operator override code
(`STRANDS_MESH_OVERRIDE_CODE`). And the estop broadcast logs a CRITICAL
`peers_not_stopped` line for sim peers - `Simulation` exposes no `stop_task`,
so the mesh honestly refuses to report those peers as halted; the asserted
property here is the lockout propagation, which engages regardless.

The read-only Rerun fleet dashboard ([#2181](https://github.com/strands-labs/robots/issues/2181))
attaches to the same mesh and shows the peer loss and the recovery live once
it lands.

## Environment variables

| Variable | Effect |
|---|---|
| `STRANDS_MESH_AUDIT_PSK` | HMAC-sign every audit record; `verify_audit_integrity()` then attests the trail. |
| `STRANDS_MESH_AUDIT_DIR` | Relocate the audit log (default `~/.strands_robots/`). |
| `STRANDS_MESH_OVERRIDE_CODE` | Operator override code required to resume a peer out of estop lockout. |
| `STRANDS_MESH_LOCAL_DEV=1` | Skip TLS for local development (defaulted by the examples). |
| `STRANDS_MESH=0` | Disable the mesh entirely; use `--dry-run` in that posture. |
