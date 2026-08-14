# Fleet orchestration examples

Multi-robot orchestration on top of the pieces the rest of `examples/`
introduces one at a time: multi-robot worlds, the Zenoh mesh with its safety
protocol, capability-based dispatch, and the signed audit log. Tracked by epic
[#2179](https://github.com/strands-labs/robots/issues/2179).

| # | Example | Status |
|---|---|---|
| 01 | [`01_skill_dispatch_multi_vendor.py`](01_skill_dispatch_multi_vendor.py) - capability-based dispatch across heterogeneous robots | here |
| 02 | `02_cross_zone_transport.py` + read-only Rerun fleet dashboard | tracked by [#2181](https://github.com/strands-labs/robots/issues/2181) |
| 03 | [`03_failover_and_degraded_ops.py`](03_failover_and_degraded_ops.py) - peer-loss reassignment + dispatcher-down safety | here |
| 04 | [`04_emergency_evacuation.py`](04_emergency_evacuation.py) - three-phase evacuate protocol, benchmark-scored | here |
| 05 | [`05_work_order_dispatch.py`](05_work_order_dispatch.py) - structured task ingress mapped onto per-site capability manifests | here |

Shared across the suite:

- [`capabilities.py`](capabilities.py) - the per-robot, per-site capability
  manifest schema (`{robot, site, skills: [{name, payload_kg, fixture,
  zones}]}`) plus the deterministic hard-constraint filter. Consumed by
  examples 01, 03 and 05.

Air-gap acceptance (epic decision D1): every example here passes with the
network fully off, given a pre-populated asset cache (`HF_HUB_OFFLINE=1` +
cached robot assets). The first live run downloads robot models from GitHub /
robot_descriptions; after that, no network is required anywhere in the suite.

## 01 - skill dispatch across vendors

One MuJoCo world, three robots from three vendors (SO-101 arm, LeKiwi wheeled
base, Unitree Go2 quadruped - repeated `add_robot` on the same world). A
skills table maps each skill name to capability requirements over registry
metadata (category, joint count, gripper) and to an execution binding
(`create_policy` provider or motion primitive); each robot's capability
manifest is derived from that metadata alone, so nothing in the dispatch path
branches on an embodiment. Matching runs through the shared `capabilities.py`
filter: a task no robot can serve is rejected with a per-robot,
machine-readable reason - never silently dropped. Every dispatch passes a
human-in-the-loop gate first.

Execution on MuJoCo is a `move_to` motion primitive for the staging skill
plus ONE synchronized `run_multi_policy` loop for every policy-bound skill.
`--backend isaac` runs the identical dispatch layer with execution falling
back to sequential per-robot `run_policy` (the base-ABC contract every
backend implements) - the fallback deliberately shows where the portability
boundary sits today ([#2122](https://github.com/strands-labs/robots/issues/2122)
tracks `run_multi_policy` parity, [#2123](https://github.com/strands-labs/robots/issues/2123)
the motion primitives).

```bash
# No simulator - match, gate, and reject with a loopback execution seam:
STRANDS_MESH_HITL_ACTIONS=none python examples/fleet/01_skill_dispatch_multi_vendor.py --dry-run

# Live: one MuJoCo world (so101 + lekiwi + unitree_go2), interactive HITL
# approval per dispatch:
python examples/fleet/01_skill_dispatch_multi_vendor.py

# Watch it (needs a local display):
python examples/fleet/01_skill_dispatch_multi_vendor.py --view

# Same dispatch layer, Isaac execution fallback (needs Isaac Sim):
python examples/fleet/01_skill_dispatch_multi_vendor.py --backend isaac

# Let a Strands Agent drive the [list_robots, match_skill, dispatch] tools
# (needs model-provider credentials; the scripted path needs none):
STRANDS_MESH_HITL_ACTIONS=none python examples/fleet/01_skill_dispatch_multi_vendor.py --dry-run --agent
```

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
| `STRANDS_MESH_HITL_ACTIONS=none` | Auto-approve dispatches and resumes (CI/smoke mode; logged loudly). The default is an interactive prompt per gated action. |
| `MUJOCO_GL` | GL backend for headless rendering (the examples default it to `egl`). |
| `STRANDS_MESH_AUDIT_PSK` | HMAC-sign every audit record; `verify_audit_integrity()` then attests the trail. |
| `STRANDS_MESH_AUDIT_DIR` | Relocate the audit log (default `~/.strands_robots/`). |
| `STRANDS_MESH_OVERRIDE_CODE` | Operator override code required to resume a peer out of estop lockout. |
| `STRANDS_MESH_LOCAL_DEV=1` | Skip TLS for local development (defaulted by the examples). |
| `STRANDS_MESH=0` | Disable the mesh entirely; use `--dry-run` in that posture. |
