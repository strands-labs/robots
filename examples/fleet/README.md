# Fleet orchestration examples

Multi-robot orchestration on top of the pieces the rest of `examples/`
introduces one at a time: multi-robot worlds, the Zenoh mesh with its safety
protocol, capability-based dispatch, and the signed audit log. Tracked by epic
[#2179](https://github.com/strands-labs/robots/issues/2179).

| # | Example | Status |
|---|---|---|
| 01 | [`01_skill_dispatch_multi_vendor.py`](01_skill_dispatch_multi_vendor.py) - capability-based dispatch across heterogeneous robots | here |
| 02 | `02_cross_zone_transport.py` + read-only Rerun fleet dashboard | tracked by [#2181](https://github.com/strands-labs/robots/issues/2181) |
| 03 | `03_failover_and_degraded_ops.py` - peer-loss reassignment + dispatcher-down safety | tracked by [#2182](https://github.com/strands-labs/robots/issues/2182) |
| 04 | [`04_emergency_evacuation.py`](04_emergency_evacuation.py) - three-phase evacuate protocol, benchmark-scored | here |
| 05 | `05_work_order_dispatch.py` - structured task ingress mapped onto per-site capability manifests | tracked by [#2185](https://github.com/strands-labs/robots/issues/2185) |

Shared across the suite:

- [`capabilities.py`](capabilities.py) - the per-robot, per-site capability
  manifest schema (`{robot, site, skills: [{name, payload_kg, fixture,
  zones}]}`) plus the deterministic hard-constraint filter. Consumed by
  examples 01 and 05.

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
| `STRANDS_MESH_HITL_ACTIONS=none` | Auto-approve dispatches and resumes (CI/smoke mode; logged loudly). The default is an interactive prompt per gated action. |
| `MUJOCO_GL` | GL backend for headless rendering (the examples default it to `egl`). |
| `STRANDS_MESH_AUDIT_PSK` | HMAC-sign every audit record; `verify_audit_integrity()` then attests the trail. |
| `STRANDS_MESH_AUDIT_DIR` | Relocate the audit log (default `~/.strands_robots/`). |
| `STRANDS_MESH_OVERRIDE_CODE` | Operator override code required to resume a peer out of estop lockout. |
| `STRANDS_MESH_LOCAL_DEV=1` | Skip TLS for local development (defaulted by the examples). |
| `STRANDS_MESH=0` | Disable the mesh entirely; use `--dry-run` in that posture. |
