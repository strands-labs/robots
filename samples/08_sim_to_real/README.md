# 🎓 Sample 08: Sim-to-Real Transfer — Cosmos Transfer & Hardware Deployment

**Level:** 3 (Advanced / High School, Grades 9–12) · **Time:** 30 minutes · **Hardware:** GPU + Physical Robot

> Close the loop: **sim train → Cosmos Transfer → real deploy**.

---

## 🧠 The Sim-to-Real Gap

Training policies in simulation is fast and safe, but simulated images don't look
like the real world.  A policy that works perfectly in MuJoCo often fails when
deployed on a real robot because textures, lighting, and camera noise are completely
different.

```
Simulation (MuJoCo)         Real World
┌───────────────┐           ┌───────────────┐
│ Perfect physics│           │ Noisy sensors  │
│ Clean images   │   GAP     │ Varied lighting│
│ No latency     │ <=======> │ Network delays │
│ Deterministic  │           │ Stochastic     │
└───────────────┘           └───────────────┘
                 │
     Cosmos Transfer 2.5 bridges this gap
     by making sim footage look real
```

**NVIDIA Cosmos Transfer 2.5** is a 2B-parameter diffusion model with
ControlNet conditioning.  It takes a simulation video plus ground-truth
control signals (depth, edges, segmentation) from the physics engine and
generates a **photorealistic** version — frame-by-frame, with robot motion
preserved exactly.

## 📦 What's in This Sample

| File | Description |
|------|-------------|
| [`cosmos_transfer.py`](cosmos_transfer.py) | Part 1: Record sim video → Cosmos Transfer → photorealistic output |
| [`train_on_transferred.py`](train_on_transferred.py) | Part 2: Collect demos → transfer → fine-tune GR00T → compare results |
| [`deploy_to_hardware.py`](deploy_to_hardware.py) | Part 3: Sim rehearsal → connect robot → run policy → record |
| [`safety_wrapper.py`](safety_wrapper.py) | Safety utilities: joint limits, velocity caps, workspace bounds |
| [`configs/cosmos_depth.yaml`](configs/cosmos_depth.yaml) | Cosmos Transfer configuration |
| [`configs/deploy_so100.yaml`](configs/deploy_so100.yaml) | SO-100 deployment safety config |
| [`configs/deploy_g1.yaml`](configs/deploy_g1.yaml) | Unitree G1 deployment safety config |
| [`configs/deploy_reachy_mini.yaml`](configs/deploy_reachy_mini.yaml) | Reachy Mini deployment safety config |

## 🚀 Quick Start

```bash
# Part 1: Cosmos Transfer (needs GPU)
python samples/08_sim_to_real/cosmos_transfer.py

# Part 2: Train on transferred data (needs GPU)
python samples/08_sim_to_real/train_on_transferred.py

# Part 3: Deploy to hardware (needs physical robot)
python samples/08_sim_to_real/deploy_to_hardware.py
```

> **No GPU?** Parts 1 and 2 will run the simulation steps and verify the
> API contracts, but skip the actual Cosmos Transfer inference.  Part 3
> requires a physical robot connection.

## 🏗️ The Full Pipeline

```
┌──────────────────────────────────────────────────────┐
│  1. Train in simulation (Sample 06: Newton/Isaac)     │
│       └─ MuJoCo / Newton / Isaac Sim                  │
│                                                        │
│  2. Record sim rollouts (Sample 05: Data Collection)   │
│       └─ sim.record_video()                            │
│                                                        │
│  3. Cosmos Transfer 2.5 (THIS SAMPLE - Part 1)        │
│       └─ sim images → photorealistic images            │
│       └─ Robot motion preserved via ControlNet         │
│                                                        │
│  4. Fine-tune on transferred data (THIS SAMPLE - Part 2)│
│       └─ GR00T N1.6 sees "real" images during training │
│                                                        │
│  5. Deploy to real robot (THIS SAMPLE - Part 3)        │
│       └─ Same Policy ABC, same create_policy()         │
│       └─ Zero code changes between sim and real!       │
└──────────────────────────────────────────────────────┘
```

## 🔑 The Key Insight: Same Policy ABC

The `Policy` abstract base class is **identical** between simulation and
real hardware.  The same `policy_provider` and `model_path` arguments work
in both contexts:

```python
# In simulation (Samples 01-06):
sim.run_policy(robot_name="so100", policy_provider="groot", ...)

# On real hardware (this sample):
real_robot.start_task(instruction="...", policy_provider="groot", ...)

# Same policy, same weights, same interface.
```

## 🤖 Hardware Deployment Matrix

| Robot | DOF | Connection | Camera Setup | Best For |
|-------|-----|-----------|-------------|----------|
| **SO-100** | 6 | USB serial (Feetech) | 1-2 webcams | Easiest to start with |
| **Unitree G1 EDU Plus** | 29 | Ethernet + SDK | Built-in stereo | Most capable, full humanoid |
| **Reachy Mini** | 6+2 | Zenoh (native) | Head cameras | Best Zenoh integration demo |

## ⚠️ Safety Checklist (BEFORE running on real hardware)

- [ ] **Sim rehearsal passes** — run `deploy_to_hardware.py` sim rehearsal step first
- [ ] **Workspace cleared** — no people, objects, or cables in the robot's reach
- [ ] **Emergency stop** within arm's reach (physical button or Ctrl+C)
- [ ] **Velocity scale reduced** — start at 30-50% (`max_velocity_scale` in config)
- [ ] **Joint limits verified** — check the YAML config matches your robot's actual limits
- [ ] **Someone watching** — never run unattended during first deployments
- [ ] **Camera connected** — verify camera feed before starting the policy

## 📚 SDK Surface Covered

| Class / Function | Module | Purpose |
|-----------------|--------|---------|
| `CosmosTransferPipeline` | `strands_robots.cosmos_transfer` | Reusable pipeline for sim→real visual transfer |
| `CosmosTransferConfig` | `strands_robots.cosmos_transfer` | Configuration (model variant, control types, guidance) |
| `transfer_video()` | `strands_robots.cosmos_transfer` | Convenience function for one-shot transfer |
| `Robot` (sim) | `strands_robots` | MuJoCo simulation factory |
| `Robot` (hardware) | `strands_robots.robot` | Physical robot control |
| `TaskStatus` | `strands_robots.robot` | Task execution state enum |
| `RobotTaskState` | `strands_robots.robot` | Real-time task monitoring dataclass |
| `SafetyWrapper` | (this sample) | Joint/velocity/workspace safety checks |

## 🧪 Exercises

1. **Control type comparison** — Run Cosmos Transfer with depth-only, edge-only,
   and depth+edge.  Which produces the most realistic output?

2. **Training comparison** — Measure policy success rate with:
   - Sim-only training data
   - Cosmos-transferred training data
   - Domain randomisation (Sample 03) + Cosmos Transfer
   Which combination works best?

3. **Cross-robot transfer** — Train on SO-100 demos, deploy on Reachy Mini.
   Does the policy generalise across different hardware?

## 🔗 Connections

| Direction | Sample | Topic |
|-----------|--------|-------|
| ← Previous | [Sample 07](../07_groot_finetune/) | GR00T Fine-Tuning |
| → Next | [Sample 09](../09_multi_robot_fleet/) | Multi-Robot Fleet Orchestration |
| Related | [Sample 03](../03_domain_randomization/) | Domain Randomisation |
| Related | [Sample 05](../05_data_collection/) | Data Collection |
| Related | [Sample 06](../06_gpu_training/) | GPU Training (Newton/Isaac) |

## 📖 Further Reading

- [Cosmos Transfer docs](../../docs/simulation/cosmos-transfer.md) — Full API reference
- [Cosmos Transfer 2.5 paper](https://arxiv.org/abs/2503.14492) — Architecture details
- [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook) — NVIDIA examples
- Issue [#126](../../issues/126) — Cosmos Video Foundation Model Training
