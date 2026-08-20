# Fleet orchestration examples

Multi-robot orchestration on top of the pieces the rest of `examples/`
introduces one at a time: multi-robot worlds, the Zenoh mesh with its safety
protocol, capability-based dispatch, and the signed audit log. Tracked by epic
[#2179](https://github.com/strands-labs/robots/issues/2179).

| # | Example | Status |
|---|---|---|
| 01 | [`01_skill_dispatch_multi_vendor.py`](01_skill_dispatch_multi_vendor.py) - capability-based dispatch across heterogeneous robots | here |
| 02 | [`02_cross_zone_transport.py`](02_cross_zone_transport.py) + read-only Rerun fleet dashboard ([`dashboard.py`](dashboard.py)) | here |
| 03 | [`03_failover_and_degraded_ops.py`](03_failover_and_degraded_ops.py) - peer-loss reassignment + dispatcher-down safety | here |
| 04 | [`04_emergency_evacuation.py`](04_emergency_evacuation.py) - three-phase evacuate protocol, benchmark-scored | here |
| 05 | [`05_work_order_dispatch.py`](05_work_order_dispatch.py) - structured task ingress mapped onto per-site capability manifests | here |

Shared across the suite:

- [`capabilities.py`](capabilities.py) - the per-robot, per-site capability
  manifest schema (`{robot, site, skills: [{name, payload_kg, fixture,
  zones}]}`) plus the deterministic hard-constraint filter. Consumed by
  examples 01, 02, 03 and 05.
- [`dashboard.py`](dashboard.py) - the read-only fleet dashboard. It attaches
  to the MESH, not to a simulator, so the same dashboard serves every example
  here and any backend.

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

## 02 - cross-zone transport

A site split into two ownership zones, each run by its own zone orchestrator -
an in-process mesh peer (`init_mesh(..., peer_id="zone-a")`, the multi-peer
pattern from `notebooks/06_fleet_orchestration.ipynb`) that owns that zone's
robots and nothing else. A fleet coordinator (a third peer, holding no robot)
handles a `stock -> etch` transport request that no single zone can serve:

1. **Decompose**: the request splits into per-zone legs joined at the zones'
   handoff dock (`dock-ab`). A same-zone request stays one leg; a request
   touching a location no zone covers, or a zone pair with no dock, is refused
   with a machine-readable reason - never guessed.
2. **Select, zone-side**: each leg's robot is chosen by running the ONE shared
   `transport` skill definition through the `capabilities.py` filter over that
   zone's manifests only. The identical skill artifact executes in both zones
   by different robot types (a LeKiwi wheeled base in zone-a, a Unitree Go2
   quadruped in zone-b) - there is zero per-zone skill code to fork, and the
   coordinator never sees another zone's robots.
3. **Dispatch, gated**: every leg goes over `mesh.send` behind a
   human-in-the-loop gate. Custody is explicit: leg 2 is never dispatched
   before leg 1's success reply (until then the tote has not physically
   reached the dock), and an aborted or declined handoff reports exactly where
   the payload is. Every transition lands in the signed audit log
   (`handoff_dispatch` / `handoff_custody` / `handoff_complete` / ...).

```bash
# No simulator, no mesh - decompose, gate, and hand off over a loopback seam:
STRANDS_MESH_HITL_ACTIONS=none python examples/fleet/02_cross_zone_transport.py --dry-run

# Live: one MuJoCo world per zone, each joined as its zone's mesh peer,
# interactive HITL approval per leg:
python examples/fleet/02_cross_zone_transport.py

# Signed audit trail:
STRANDS_MESH_AUDIT_PSK=demo-psk python examples/fleet/02_cross_zone_transport.py --dry-run
```

A declined approval is a first-class outcome: the leg is recorded as declined,
nothing executes, and - same property the `robot_mesh` tool pins - a decline
consumes no rate-limit slot, so nuisance prompts an operator turns down can
never lock the fleet out of a genuine emergency action.

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
   the outage: the presence registry and the signed audit log. Both are
   awaited on a bounded deadline rather than sampled once - a peer's own
   `remote_estop_engaged` row is written by that peer's safety handler when
   the broadcast reaches it, so it can land after the issuing call has
   already returned.

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

The read-only Rerun fleet dashboard ([`dashboard.py`](dashboard.py), next
section) attaches to the same mesh and shows the peer loss and the recovery
live.

## The read-only fleet dashboard

`dashboard.py` joins the mesh as its own peer (`peer_type="dashboard"`),
subscribes to presence, health and safety topics, tails the signed audit log,
and renders a fleet table (peer, type, presence age, battery, last safety
event, current task) plus an event timeline (dispatch / estop / resume / HITL
decisions). Rendering uses [Rerun](https://rerun.io) when `rerun-sdk` is
installed and degrades gracefully to a terminal table when it is not.

READ-ONLY is enforced, not narrated: `restrict_to_subscribe_only` replaces
every command-capable method on the peer (`send` / `tell` / `broadcast` /
`emergency_stop` / `publish_step`) with a refusal and confines raw `publish`
to the peer's own presence namespace, so the dashboard cannot publish a
command, an estop, or a resume. HITL approvals stay in the operator terminal;
a write-capable UI is an explicit epic non-goal.

```bash
# Terminal 1 - the dashboard (Rerun viewer when available). Two processes
# need a discovery channel: multicast scouting is off by default, so enable
# it on BOTH sides (trusted networks only) or configure connect endpoints:
STRANDS_MESH_MULTICAST=true python examples/fleet/dashboard.py

# Terminal 2 - any fleet example on the same mesh:
STRANDS_MESH_MULTICAST=true python examples/fleet/02_cross_zone_transport.py

# Headless / CI posture: terminal renderer, bounded runtime:
python examples/fleet/dashboard.py --no-rerun --duration 10
```

On a headless or remote host - where a fleet dashboard naturally runs - the
native viewer has no display to open on. `--serve-web` instead serves the
Rerun web viewer and the live log stream from the dashboard process and
prints the ready-to-open URL (the `?url=rerun%2Bhttp...` form):

```bash
# Remote box - no display needed. Binds 127.0.0.1 by default per the repo's
# network-exposure convention; --bind 0.0.0.0 deliberately opts into wider
# exposure and the startup output says so.
STRANDS_MESH_MULTICAST=true python examples/fleet/dashboard.py --serve-web
# ports: --web-port 9090 (viewer HTTP) / --grpc-port 9876 (log stream)

# Local machine - tunnel BOTH ports (the browser fetches the viewer from one
# and dials the log stream on the other), then open the printed URL:
ssh -N -L 9090:127.0.0.1:9090 -L 9876:127.0.0.1:9876 user@remote-box
```

Any loopback address gets that tunnel recipe, not just the default one:
`--bind` takes an IP literal, and the whole `127.0.0.0/8` block is reachable
only from the host, so `--bind 127.0.0.2` (isolating the viewer on its own
loopback address) prints the recipe forwarded to `127.0.0.2`. The
network-exposure warning is for a bind that really is wider.

The web viewer is a view, not a surface: it changes only the render
transport, and the mesh peer stays subscribe-only exactly as above. With
rerun-sdk absent, `--serve-web` fails with the install hint rather than
silently falling back to the terminal renderer - the explicit ask for a web
viewer must not degrade into tables nobody is watching.

Because it attaches to the mesh and not the simulator, the dashboard works
unchanged across every example in this suite and every backend (epic decision
D9).


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
| `STRANDS_MESH_HITL_ACTIONS=none` | Auto-approve dispatches and resumes (CI/smoke mode; logged loudly). The default is an interactive prompt per gated action (per leg in example 02). |
| `MUJOCO_GL` | GL backend for headless rendering (the examples default it to `egl`). |
| `STRANDS_MESH_AUDIT_PSK` | HMAC-sign every audit record; `verify_audit_integrity()` then attests the trail. |
| `STRANDS_MESH_AUDIT_DIR` | Relocate the audit log (default `~/.strands_robots/`). Point the dashboard and the examples at the same directory. |
| `STRANDS_MESH_OVERRIDE_CODE` | Operator override code required to resume a peer out of estop lockout. |
| `STRANDS_MESH_LOCAL_DEV=1` | Skip TLS for local development (defaulted by the examples). |
| `STRANDS_MESH_MULTICAST=true` | Enable multicast scouting so separate processes (e.g. the dashboard and an example) discover each other. Off by default; trusted networks only. |
| `STRANDS_MESH=0` | Disable the mesh entirely; use `--dry-run` in that posture. |
| _(mesh extra absent)_ | With `eclipse-zenoh` not installed the mesh stays off (`mesh.alive` is `False`): the live paths refuse at start-up naming the peer and the remedy. Install `strands-robots[mesh]` or use `--dry-run`. |
