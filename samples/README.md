# 🎓 Strands Robots — Educational Curriculum

> **10 progressive learning samples** following K12 universal learning format.
> From "Hello Robot" to GPU-accelerated training pipelines in ~4 hours.

See [#148](../../issues/148) for the full curriculum plan.

## Learning Path

| # | Sample | Level | Time | Hardware |
|---|--------|-------|------|----------|
| 01 | [Hello Robot](01_hello_robot/) | 🟢 Elementary | 10 min | CPU |
| 02 | [Policy Playground](02_policy_playground/) | 🟢 Elementary | 15 min | CPU |
| 03 | [Build a World](03_build_a_world/) | 🟠 Middle | 20 min | CPU |
| 04 | [Gymnasium Training](04_gymnasium_training/) | 🟠 Middle | 25 min | CPU (+GPU) |
| 05 | [Data Collection](05_data_collection/) | 🟠 Middle | 20 min | CPU |
| 06 | [Ray-Traced Training](06_raytraced_training/) | 🔴 Advanced | 30 min | GPU |
| 07 | [GR00T N1.6 Fine-Tuning](07_groot_finetuning/) | 🔴 Advanced | 30 min | GPU |
| 08 | [Sim-to-Real Transfer](08_sim_to_real/) | 🔴 Advanced | 30 min | GPU + HW |
| 09 | [Multi-Robot Fleet](09_multi_robot_fleet/) | 🔴 Advanced | 25 min | Multi-device |
| 10 | [Autonomous Repo Case Study](10_autonomous_repo_casestudy/) | 🟣 Meta | 20 min | CPU |

## Quick Start

```bash
# Install
pip install strands-robots[sim]

# Level 1 — Elementary (CPU only)
cd 01_hello_robot && python hello_robot.py        # Your first robot in 1 line
cd 02_policy_playground && python policy_playground.py  # Explore 17+ policy providers

# Level 2 — Middle School (CPU only)
cd 03_build_a_world && python build_world.py      # Create 3D scenes with robots + objects
cd 04_gymnasium_training && python gym_basics.py   # RL environment basics
cd 05_data_collection && python record_episodes.py # Record robot demonstrations

# Level 3 — Advanced (GPU recommended)
cd 06_raytraced_training && python newton_parallel.py  # Newton GPU-parallel training
cd 07_groot_finetuning && python inference_demo.py     # GR00T N1.6 inference
cd 08_sim_to_real && python cosmos_transfer.py         # Cosmos Transfer sim-to-real
cd 09_multi_robot_fleet && python zenoh_mesh_demo.py   # Multi-robot Zenoh mesh

# Meta
cd 10_autonomous_repo_casestudy && python analyze_commits.py  # How this repo self-evolved
```

## Architecture Overview

```
Natural Language → Strands Agent → Robot() Factory
                                    ├── Simulation (MuJoCo CPU)     ← Samples 01-05
                                    ├── IsaacSimBackend (GPU)       ← Sample 06
                                    ├── NewtonBackend (GPU)         ← Sample 06
                                    └── HardwareRobot (real)        ← Sample 08
                                          ↓
                                    PolicyRegistry (plugin-based)
                                    ├── mock (sine wave, testing)   ← Sample 01
                                    ├── lerobot_local (HF models)   ← Sample 02
                                    ├── groot (NVIDIA GR00T)        ← Sample 07
                                    └── 15+ more providers          ← Sample 02
```

## Prerequisites

- **Level 1 (Elementary):** Python 3.10+, `pip install strands-robots[sim]`
- **Level 2 (Middle):** + PyTorch basics, optional GPU
- **Level 3 (Advanced):** + ML fundamentals, GPU required (NVIDIA L40S or Thor)

## Partner Integration

| Partner | Samples | Technology |
|---------|---------|-----------|
| **NVIDIA** | 06, 07, 08 | GR00T N1.6, Isaac Sim, Cosmos Transfer, Newton |
| **ARM** | 09 | `device_connect` via Zenoh mesh |
| **HuggingFace** | 02, 05, 07 | LeRobot datasets, model hosting |
