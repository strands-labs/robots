# Examples

Each example demonstrates ONE strands-robots primitive in under 60 lines.
No raw lerobot wrangling - the SDK handles configuration, feature schemas,
and hardware abstraction internally.

## Quick start

```bash
pip install "strands-robots[sim-mujoco,lerobot,mesh]"
MUJOCO_GL=egl python examples/01_sim_hello_world.py
```

## Directory map

The numbered `01_*`..`15_*` scripts below are the core primitive walkthroughs and
live at the top level. Everything else is grouped by topic:

- [`vla/`](vla/) - vision-language-action provider examples (Cosmos 3, MolmoAct2)
- [`wbc/`](wbc/) - whole-body control on the Unitree G1 (torque deploy, gait, composite)
- [`locomotion/`](locomotion/) - G1 locomotion and the VLA-on-G1 record→tune→deploy workflow
- [`training/`](training/) - from-scratch RL trainers (PPO, FastSAC)
- [`mesh/`](mesh/) - Zenoh mesh ACL config templates
- [`registry/`](registry/) - robot / hardware catalog discovery
- [`lerobot/`](lerobot/) - LeRobot hub-to-hardware companion scripts
- [`ros2/`](ros2/) - ROS 2 bridge demos
- [`vera_mimicgen_panda/`](vera_mimicgen_panda/) - VERA MimicGen → Panda rollout
- [`notebooks/`](notebooks/) - Jupyter getting-started series (CPU-only)

## Index

Prefer a click-and-run walkthrough? The [`notebooks/`](notebooks/) folder has the
getting-started series (`Robot()` basics, record + stream, and the full
record→train→deploy loop) as Jupyter notebooks - all CPU-only, no hardware or GPU.

| # | File | Primitive | Hardware | GPU |
|---|------|-----------|----------|-----|
| 01 | [`01_sim_hello_world.py`](01_sim_hello_world.py) | `Robot()` + `Simulation` | No | No |
| 02 | [`02_policy_abstraction.py`](02_policy_abstraction.py) | `create_policy()` | No | No |
| 03 | [`03_record_dataset.py`](03_record_dataset.py) | `start/stop_recording` | No | No |
| 04 | [`04_mesh_peer_discovery.py`](04_mesh_peer_discovery.py) | `Mesh` + peer discovery | No | No |
| 05 | [`05_agent_natural_language.py`](05_agent_natural_language.py) | `Agent` + `Robot` tool | No | No (needs LLM API) |
| 06 | [`06_agent_collect_and_stream.py`](06_agent_collect_and_stream.py) | `Agent` record + `stream_dataset` | No | No (needs LLM API) |
| 07 | [`07_post_tune_any_policy.py`](07_post_tune_any_policy.py) | `create_trainer` + `TrainSpec` (record→train→load) | No | No |
| 08 | [`08_discover_lerobot.py`](08_discover_lerobot.py) | `use_lerobot` tool: discover LeRobot robots, policies, teleoperators, cameras | No | No |
| 09 | [`09_procedural_terrain.py`](09_procedural_terrain.py) | `create_world(terrain=...)` heightfield ground + `difficulty` curriculum | No | No |
| 10 | [`10_evaluate_benchmark.py`](10_evaluate_benchmark.py) | `register_builtin_benchmarks` + `evaluate_benchmark` (success rate / reward) | No | No |
| 11 | [`11_author_a_benchmark.py`](11_author_a_benchmark.py) | `DeclarativeBenchmark.from_dict` + predicate DSL (author a task) | No | No |
| 12 | [`12_domain_randomization.py`](12_domain_randomization.py) | `randomize` + `set_obs_noise` (sim2real appearance/physics/sensor noise) | No | No |
| 13 | [`13_physics_introspection.py`](13_physics_introspection.py) | `get_jacobian` / `get_mass_matrix` / `inverse_dynamics` / `get_energy` | No | No |
| 14 | [`14_save_state_and_perturb.py`](14_save_state_and_perturb.py) | `save_state`/`load_state` + `apply_force` + `raycast` | No | No |
| 15 | [`15_robot_catalog.py`](15_robot_catalog.py) | `list_robots` / `get_robot` registry discovery (no sim) | No | No |
| 16 | [`16_harness_memory.py`](16_harness_memory.py) | `harness_memory` tool: save a solution trace, reuse it under spatial perturbation | No | No |
| -- | [`locomotion/vla_g1_workflow.py`](locomotion/vla_g1_workflow.py) | VLA-on-G1: record -> GR00T fine-tune -> WBC deploy | No | Optional (tune) |
| — | [`vera_mimicgen_panda/`](vera_mimicgen_panda/) | VERA MimicGen → Panda (eef-delta + IK bridge) | No | **Yes** (server) |
| — | [`isaac/isaac_replicator_synthdata.py`](isaac/isaac_replicator_synthdata.py) | `IsaacSimulation` + Omniverse Replicator synthetic-data generation | No | **Yes** (Isaac Sim / RTX) |
| — | [`isaac_gs/`](isaac_gs/) | Isaac RTX robot z-composited over a 3DGS / panorama backdrop (digital-twin) | No | **Yes** (Isaac Sim / RTX) |
| — | [`mujoco_gs/`](mujoco_gs/) | MuJoCo + 3D Gaussian Splatting hybrid render (depth-aware composite) driven by the `Simulation` AgentTool | No | Optional (`gsplat`) |
| -- | [`registry/lerobot_hardware_catalog.py`](registry/lerobot_hardware_catalog.py) | `Robot()` covers the whole LeRobot hardware catalog (name -> lerobot_type) | No | No |

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
| [`vla/molmoact2_so101_pickplace.py`](vla/molmoact2_so101_pickplace.py) | Real hardware + MolmoAct2 VLA policy on SO-101 |
| [`vla/cosmos3_sim_rollout.py`](vla/cosmos3_sim_rollout.py) | Cosmos 3 VLA in MuJoCo with WebSocket policy server |
| [`wbc/wbc_g1_torque_deploy.py`](wbc/wbc_g1_torque_deploy.py) | GR00T-WBC (SONIC) locomotion on the Unitree G1 via the torque-control deploy loop |
| [`lerobot/hub_to_hardware.py`](lerobot/hub_to_hardware.py) | Full agent-driven pipeline: record, train, deploy |
| [`so101_curobo/`](so101_curobo/) | SO-101 tabletop pick-and-place: cuRobo motion planning + LeRobot dataset capture. Backend-agnostic (`SimEngine`): MuJoCo today, Isaac when `strands-robots[sim-isaac]` is installed. **GPU: Optional** (cuRobo / Isaac) |
| [`libero/run_mujoco.py`](libero/run_mujoco.py) | LIBERO benchmark eval on the default MuJoCo backend with whole-run MP4 recording. **GPU: Optional** (`--policy mock` is CPU-only; `--policy groot` needs Docker + NVIDIA GPU) |
| [`libero/run_mujoco_agent.py`](libero/run_mujoco_agent.py) | LIBERO-on-MuJoCo driven by a Strands `Agent` in natural language. **GPU: Optional** (needs LLM API) |
| [`libero/run_isaac.py`](libero/run_isaac.py) | LIBERO benchmark eval on the Isaac Sim backend (`create_simulation("isaac")`) with rollout-MP4 recording. **GPU: Yes** (Isaac Sim 6.0+) |
| [`libero/run_isaac_agent.py`](libero/run_isaac_agent.py) | LIBERO-on-Isaac driven by a Strands `Agent` in natural language. **GPU: Yes** (Isaac Sim 6.0+, needs LLM API) |
| [`libero/libero_backend_matrix.py`](libero/libero_backend_matrix.py) | Run one LIBERO task across every installed backend, side-by-side `success_rate` / `wall_time` table |

## Environment variables

- `MUJOCO_GL=egl` - headless rendering (required on servers without display)
- `STRANDS_ISAAC_RTX_PATHTRACING=1` - Isaac Sim LIBERO examples: photoreal RTX pathtracing render mode (default is `rtx_realtime`)
- `STRANDS_MESH_LOCAL_DEV=1` - skip TLS for mesh examples in local dev
- `STRANDS_MESH=0` - disable mesh entirely
- `HF_TOKEN` - push datasets to Hugging Face Hub
- `STRANDS_TRUST_REMOTE_CODE=1` - required for some HF policy checkpoints
