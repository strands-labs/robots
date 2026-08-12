# Fleet orchestration examples

Multi-robot orchestration on top of the pieces the rest of `examples/`
introduces one at a time: multi-robot worlds, the Zenoh mesh with its safety
protocol, capability-based dispatch, and the signed audit log. Tracked by epic
[#2179](https://github.com/strands-labs/robots/issues/2179).

| # | Example | Status |
|---|---|---|
| 01 | `01_skill_dispatch_multi_vendor.py` - capability-based dispatch across heterogeneous robots | tracked by [#2180](https://github.com/strands-labs/robots/issues/2180) |
| 02 | `02_cross_zone_transport.py` + read-only Rerun fleet dashboard | tracked by [#2181](https://github.com/strands-labs/robots/issues/2181) |
| 03 | `03_failover_and_degraded_ops.py` - peer-loss reassignment + dispatcher-down safety | tracked by [#2182](https://github.com/strands-labs/robots/issues/2182) |
| 04 | [`04_emergency_evacuation.py`](04_emergency_evacuation.py) - three-phase evacuate protocol, benchmark-scored | here |
| 05 | `05_work_order_dispatch.py` - structured task ingress mapped onto per-site capability manifests | tracked by [#2185](https://github.com/strands-labs/robots/issues/2185) |

Shared across the suite (each lands with its sibling PR): `capabilities.py`,
the per-robot capability manifest schema consumed by 01/03/05, and
`dashboard.py`, the read-only Rerun fleet dashboard (#2181) that attaches to
the MESH, not a simulator, so it serves every example here unchanged.

## 04 - emergency evacuation

During an emergency, robots must clear the evacuation path and never block
personnel. A plain e-stop is NOT sufficient - a frozen robot in a corridor is
itself the blocking hazard, and mesh lockout refuses everything except
`status`/`resume`. So the protocol is three phases on existing primitives,
with the LLM outside the safety path (epic decisions D2/D5/D7):

1. **Abort**: an injected alarm broadcast (a mesh event, not a simulated
   sensor) stops every policy rollout fleet-wide - rate-limited, so an alarm
   flood cannot re-trigger the protocol, and audited either way.
2. **Clear path**: each robot runs a pre-validated deterministic retreat -
   scripted base/joint setpoints, never a learned policy - to its muster
   pose. Priority conflicts resolve by deterministic corridor-distance
   ordering: closest to the path moves first, the others hold. The retreat
   sits behind the small `EvacuationWorld` seam so the Isaac adapter
   ([#2123](https://github.com/strands-labs/robots/issues/2123)) drops in
   later.
3. **Lockout + HITL resume**: mesh lockout engages only AFTER the path is
   asserted clear (D7 - lockout first would freeze the hazard in place).
   Resume goes through the existing HMAC override protocol with operator
   approval; a wrong-code or declined resume leaves the lockout engaged, and
   the example proves it with a live probe.

Scored, not narrated: a `DeclarativeBenchmark` (predicate DSL) fails the run
if any robot re-enters the clearance-inflated corridor at any tick after the
path is declared clear, and succeeds only when the personnel proxy reaches
the exit unimpeded (a blocked proxy waits - it never teleports through) AND
the fleet-wide abort met its deadline. Post-event, an incident report is
built deterministically from the signed audit log.

```bash
# No simulator, no mesh - scripted kinematics through the same protocol core:
python examples/fleet/04_emergency_evacuation.py --dry-run

# Live: one MuJoCo corridor world (lekiwi + go2 mid-task in the corridor,
# so101 overhanging it), mesh alarm -> abort -> retreat -> lockout -> resume:
STRANDS_MESH_HITL_ACTIONS=none python examples/fleet/04_emergency_evacuation.py

# Artifacts: Rerun replay (.rrd: clearance plot + camera tiles) and a GIF of
# the robots vacating the corridor; signed audit trail for the report:
STRANDS_MESH_AUDIT_PSK=demo-psk python examples/fleet/04_emergency_evacuation.py \
  --rrd evacuation.rrd --gif evacuation.gif
```

Interactive operator approval is the documented default for the resume; the
CI/smoke posture is `STRANDS_MESH_HITL_ACTIONS=none` (epic D4). A declined
approval sends nothing - the lockout stays engaged, and (same property the
`robot_mesh` tool pins) a decline consumes no rate-limit slot anywhere, so
nuisance prompts can never lock the fleet out of a genuine emergency action.

The mesh ACL template ([`examples/mesh/mesh_acl_example.json5`](../mesh/mesh_acl_example.json5))
now ships a read-only `dashboard` role for the fleet dashboard: subscribe to
presence/health/safety/camera plus its own presence announcement, and no
`cmd`/`broadcast`/`safety` publish grant - a compromised dashboard cert can
watch, not actuate.

## Environment variables

| Variable | Effect |
|---|---|
| `STRANDS_MESH_AUDIT_PSK` | HMAC-sign every audit record; `verify_audit_integrity()` then attests the trail. |
| `STRANDS_MESH_AUDIT_DIR` | Relocate the audit log (default `~/.strands_robots/`). |
| `STRANDS_MESH_OVERRIDE_CODE` | Operator override code required to resume a peer out of estop lockout. |
| `STRANDS_MESH_HITL_ACTIONS` | `none` = unattended approvals (CI posture); unset = interactive prompts. |
| `STRANDS_MESH_LOCAL_DEV=1` | Skip TLS for local development (defaulted by the examples). |
| `STRANDS_MESH=0` | Disable the mesh entirely; use `--dry-run` in that posture. |
