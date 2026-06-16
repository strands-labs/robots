# Examples

Each example demonstrates ONE strands-robots primitive in under 60 lines.
No raw lerobot wrangling - the SDK handles configuration, feature schemas,
and hardware abstraction internally.

## Quick start

```bash
pip install "strands-robots[sim-mujoco,lerobot,mesh]"
MUJOCO_GL=egl python examples/01_sim_hello_world.py
```

## Index

| # | File | Primitive | Hardware | GPU |
|---|------|-----------|----------|-----|
| 01 | [`01_sim_hello_world.py`](01_sim_hello_world.py) | `Robot()` + `Simulation` | No | No |
| 02 | [`02_policy_abstraction.py`](02_policy_abstraction.py) | `create_policy()` | No | No |
| 03 | [`03_record_dataset.py`](03_record_dataset.py) | `start/stop_recording` | No | No |
| 04 | [`04_mesh_peer_discovery.py`](04_mesh_peer_discovery.py) | `Mesh` + peer discovery | No | No |
| 05 | [`05_agent_natural_language.py`](05_agent_natural_language.py) | `Agent` + `Robot` tool | No | No (needs LLM API) |

## What each example shows vs raw lerobot

| Task | Raw lerobot | strands-robots |
|------|------------|----------------|
| Sim setup | Manual MjSpec, XML parsing, actuator config | `Robot("so100")` (world + robot in one call) |
| Policy loading | Import provider, build config, handle embodiment mapping | `create_policy("mock")` or `create_policy("hf/repo")` |
| Dataset recording | `LeRobotDataset.create(features={...}, ...)` + manual frame loop | `start_recording()` / `stop_recording()` |
| Multi-robot networking | Custom pub/sub, IP management, serialization | `Mesh` auto-joins, `get_peers()` discovers |
| Agent control | N/A (lerobot has no agent layer) | `Agent(tools=[Robot(...)])` |

## Advanced examples

| File | What it shows |
|------|--------------|
| [`molmoact2_so101_pickplace.py`](molmoact2_so101_pickplace.py) | Real hardware + MolmoAct2 VLA policy on SO-101 |
| [`cosmos3_sim_rollout.py`](cosmos3_sim_rollout.py) | Cosmos 3 VLA in MuJoCo with WebSocket policy server |
| [`wbc_g1_torque_deploy.py`](wbc_g1_torque_deploy.py) | GR00T-WBC (SONIC) locomotion on the Unitree G1 via the torque-control deploy loop |
| [`lerobot/hub_to_hardware.py`](lerobot/hub_to_hardware.py) | Full agent-driven pipeline: record, train, deploy |

## Environment variables

- `MUJOCO_GL=egl` - headless rendering (required on servers without display)
- `STRANDS_MESH_LOCAL_DEV=1` - skip TLS for mesh examples in local dev
- `STRANDS_MESH=0` - disable mesh entirely
- `HF_TOKEN` - push datasets to Hugging Face Hub
- `STRANDS_TRUST_REMOTE_CODE=1` - required for some HF policy checkpoints
