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
| 04 | `04_emergency_evacuation.py` - three-phase evacuate protocol, benchmark-scored | tracked by [#2183](https://github.com/strands-labs/robots/issues/2183) |
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

## Environment variables

| Variable | Effect |
|---|---|
| `STRANDS_MESH_HITL_ACTIONS=none` | Auto-approve dispatches (CI/smoke mode; logged loudly). The default is an interactive prompt per dispatch. |
| `MUJOCO_GL` | GL backend for headless rendering (the examples default it to `egl`). |
