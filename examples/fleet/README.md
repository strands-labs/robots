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
| 04 | `04_emergency_evacuation.py` - three-phase evacuate protocol, benchmark-scored | tracked by [#2183](https://github.com/strands-labs/robots/issues/2183) |
| 05 | [`05_work_order_dispatch.py`](05_work_order_dispatch.py) - structured task ingress mapped onto per-site capability manifests | here |

Shared across the suite:

- [`capabilities.py`](capabilities.py) - the per-robot, per-site capability
  manifest schema (`{robot, site, skills: [{name, payload_kg, fixture,
  zones}]}`) plus the deterministic hard-constraint filter. Consumed by
  examples 01 and 05.

## 05 - work-order dispatch

Work orders arrive on a JSONL file queue speaking business vocabulary only
(`{order_id, material, operation, qty, from: "site-a/litho", to:
"site-a/etch", due}`). The translation to robot vocabulary is two-stage:

1. **Deterministic**: schema validation, then hard-constraint filtering
   (site, payload, fixture, zones) against the capability manifests. An order
   no robot can serve is NACKed back onto the queue with a machine-readable
   reason - never silently dropped.
2. **Agent**: choosing among the capable robots and sequencing multi-step
   orders. A choice outside the feasible set is refused (`guard_choice`), so
   the agent can never invent a capability.

Dispatch goes over the mesh behind a human-in-the-loop gate; the `order_id`
is threaded through the signed audit log end to end; a structured
completion/failure event lands back on the queue
(`work_order_events.jsonl`).

```bash
# No simulator, no mesh - validate/filter/sequence with a loopback transport:
STRANDS_MESH_HITL_ACTIONS=none python examples/fleet/05_work_order_dispatch.py --dry-run

# Live: one MuJoCo world (so101 + lekiwi + go2), dispatch over the mesh,
# interactive HITL approval per dispatch:
python examples/fleet/05_work_order_dispatch.py

# Signed audit trail:
STRANDS_MESH_AUDIT_PSK=demo-psk STRANDS_MESH_HITL_ACTIONS=none \
    python examples/fleet/05_work_order_dispatch.py --dry-run
```

The example's boundary is the JSON schema, not the transport: a production
ingress can front the same schema with a queue service or an API. The file
queue keeps the suite air-gapped (epic decision D1: everything here passes
with the network off, given cached robot assets).

## Environment variables

| Variable | Effect |
|---|---|
| `STRANDS_MESH_HITL_ACTIONS=none` | Auto-approve dispatches (CI/smoke mode; logged loudly). The default is an interactive prompt per dispatch. |
| `STRANDS_MESH_AUDIT_PSK` | HMAC-sign every audit record; `verify_audit_integrity()` then attests the trail. |
| `STRANDS_MESH_AUDIT_DIR` | Relocate the audit log (default `~/.strands_robots/`). |
| `STRANDS_MESH_LOCAL_DEV=1` | Skip TLS for local development (defaulted by the examples). |
| `STRANDS_MESH=0` | Disable the mesh entirely; use `--dry-run` in that posture. |
