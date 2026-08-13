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
| 04 | `04_emergency_evacuation.py` - three-phase evacuate protocol, benchmark-scored | tracked by [#2183](https://github.com/strands-labs/robots/issues/2183) |
| 05 | `05_work_order_dispatch.py` - structured task ingress mapped onto per-site capability manifests | tracked by [#2185](https://github.com/strands-labs/robots/issues/2185) |

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

## Environment variables

| Variable | Effect |
|---|---|
| `STRANDS_MESH_HITL_ACTIONS=none` | Auto-approve dispatches (CI/smoke mode; logged loudly). The default is an interactive prompt per dispatch. |
| `MUJOCO_GL` | GL backend for headless rendering (the examples default it to `egl`). |
| `STRANDS_MESH_AUDIT_PSK` | HMAC-sign every audit record; `verify_audit_integrity()` then attests the trail. |
| `STRANDS_MESH_AUDIT_DIR` | Relocate the audit log (default `~/.strands_robots/`). |
| `STRANDS_MESH_OVERRIDE_CODE` | Operator override code required to resume a peer out of estop lockout. |
| `STRANDS_MESH_LOCAL_DEV=1` | Skip TLS for local development (defaulted by the examples). |
| `STRANDS_MESH=0` | Disable the mesh entirely; use `--dry-run` in that posture. |
