# Fleet orchestration examples

Multi-robot orchestration on top of the pieces the rest of `examples/`
introduces one at a time: multi-robot worlds, the Zenoh mesh with its safety
protocol, capability-based dispatch, and the signed audit log. Tracked by epic
[#2179](https://github.com/strands-labs/robots/issues/2179).

| # | Example | Status |
|---|---|---|
| 01 | [`01_skill_dispatch_multi_vendor.py`](01_skill_dispatch_multi_vendor.py) - capability-based dispatch across heterogeneous robots | here |
| 02 | [`02_cross_zone_transport.py`](02_cross_zone_transport.py) + read-only Rerun fleet dashboard ([`dashboard.py`](dashboard.py)) | here |
| 03 | `03_failover_and_degraded_ops.py` - peer-loss reassignment + dispatcher-down safety | tracked by [#2182](https://github.com/strands-labs/robots/issues/2182) |
| 04 | `04_emergency_evacuation.py` - three-phase evacuate protocol, benchmark-scored | tracked by [#2183](https://github.com/strands-labs/robots/issues/2183) |
| 05 | `05_work_order_dispatch.py` - structured task ingress mapped onto per-site capability manifests | tracked by [#2185](https://github.com/strands-labs/robots/issues/2185) |

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

Because it attaches to the mesh and not the simulator, the dashboard works
unchanged across every example in this suite and every backend (epic decision
D9).

## Environment variables

| Variable | Effect |
|---|---|
| `STRANDS_MESH_HITL_ACTIONS=none` | Auto-approve dispatches (CI/smoke mode; logged loudly). The default is an interactive prompt per dispatch (per leg in example 02). |
| `STRANDS_MESH_AUDIT_PSK` | HMAC-sign every audit record; `verify_audit_integrity()` then attests the trail. |
| `STRANDS_MESH_AUDIT_DIR` | Relocate the audit log (default `~/.strands_robots/`). Point the dashboard and the examples at the same directory. |
| `STRANDS_MESH_LOCAL_DEV=1` | Skip TLS for local development (defaulted by the examples). |
| `STRANDS_MESH_MULTICAST=true` | Enable multicast scouting so separate processes (e.g. the dashboard and an example) discover each other. Off by default; trusted networks only. |
| `STRANDS_MESH=0` | Disable the mesh entirely; use `--dry-run` in that posture. |
| `MUJOCO_GL` | GL backend for headless rendering (the examples default it to `egl`). |
