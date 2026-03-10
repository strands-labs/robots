# 🎓 Sample 06: Ray-Traced Training Pipeline — Isaac Sim + Newton GPU

**Level:** 3 (Advanced / High School) | **Time:** 30 minutes | **Hardware:** GPU Required

> **⚠️ GPU Required:** This sample requires an NVIDIA GPU with 24GB+ VRAM.
> It runs on `thor.yml` (AGX Thor, 132GB sm_110) or `isaac-sim.yml` (EC2 L40S, 48GB).
> CPU-only users can review the pre-generated outputs below.

## 🎯 What You'll Learn

1. The 3-backend simulation architecture (MuJoCo → Newton → Isaac Sim)
2. Running Newton GPU physics with 4096 parallel environments
3. Comparing 7 Newton physics solvers (Featherstone, XPBD, MuJoCo, ...)
4. Setting up Isaac Sim with RTX ray tracing for photorealistic rendering
5. Training with Isaac Lab's RL framework at 600K+ env-steps/second
6. Running Thor + Isaac Sim in tandem (physics on Thor, rendering on EC2)

## 📋 Prerequisites

- **Samples 01–05 completed** (understand sim, policies, gymnasium, data)
- **NVIDIA GPU** with 24GB+ VRAM (or access to Thor/EC2 runners)
- **Newton:** `pip install strands-robots[newton]`
- **Isaac Sim:** `pip install strands-robots[isaac]` + Isaac Sim 5.1+

## 🧠 The 3-Backend Architecture

```
                    strands-robots Simulation Backends
                    ═════════════════════════════════

┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   MuJoCo       │   │   Newton       │   │   Isaac Sim    │
│   CPU          │   │   GPU (Warp)   │   │   RTX GPU      │
│   1 env        │   │   4096+ envs   │   │   50+ envs     │
│   Fast iterate │   │   7 solvers    │   │   Ray tracing  │
│   Samples 1–5  │   │   Differentiable│   │   Photorealism │
│   ~100K steps/s│   │   ~2M steps/s  │   │   ~600K steps/s│
└───────────────┘   └───────────────┘   └───────────────┘
     ↓ Same API        ↓ Same API          ↓ Same API
     Robot("so100")    NewtonBackend       IsaacSimBackend
```

**Key insight:** All three backends share the same high-level concepts
(`create_world`, `add_robot`, `step`, `get_observation`), so switching
between them requires minimal code changes.

## 📂 Files

| File | Description |
|------|-------------|
| [`newton_parallel.py`](newton_parallel.py) | Newton 4096-env parallel physics with solver comparison |
| [`isaac_sim_setup.py`](isaac_sim_setup.py) | Isaac Sim connection, scene loading, RTX rendering |
| [`train_isaac_lab.py`](train_isaac_lab.py) | Full Isaac Lab RL training pipeline (RSL-RL + SB3) |
| [`thor_isaac_tandem.py`](thor_isaac_tandem.py) | Thor ↔ EC2 tandem: GPU physics + RTX rendering |
| [`benchmark.py`](benchmark.py) | Compare all 3 backends on the same task |
| [`configs/`](configs/) | YAML configs for Newton and Isaac Lab training |

## 🔬 Newton: GPU-Parallel Physics

Newton is a GPU-native physics engine built on NVIDIA Warp. It wraps
MuJoCo-Warp and supports 7 solver backends:

| Solver | Best For | Speed |
|--------|----------|-------|
| `mujoco` | General purpose, reference | ★★★★ |
| `featherstone` | Articulated robots, locomotion | ★★★★★ |
| `semi_implicit` | Differentiable sim, optimization | ★★★ |
| `xpbd` | Cloth, soft bodies, constraints | ★★★ |
| `vbd` | Volume-preserving deformables | ★★ |
| `style3d` | Garment simulation | ★★ |
| `implicit_mpm` | Granular, fluid, snow | ★★ |

```python
from strands_robots.newton import NewtonBackend, NewtonConfig

config = NewtonConfig(num_envs=4096, solver="featherstone", device="cuda:0")
backend = NewtonBackend(config)
backend.create_world()
backend.add_robot("go2", urdf_path="path/to/go2.urdf")
backend.replicate(num_envs=4096)

for _ in range(1000):
    obs = backend.get_observation("go2")    # shape: [4096, obs_dim]
    actions = policy(obs)                    # batch inference
    backend.step(actions)                    # 4096 envs step simultaneously

backend.destroy()
```

### Differentiable Simulation

Newton supports full gradient computation via `wp.Tape`, enabling:
- Trajectory optimization
- Sim-to-real parameter identification
- Learned physics-based control

```python
config = NewtonConfig(enable_differentiable=True, solver="semi_implicit")
backend = NewtonBackend(config)
# ... setup ...
result = backend.run_diffsim(
    num_steps=100,
    loss_fn=lambda states: distance_to_target(states[-1]),
    optimize_params="initial_velocity",
    lr=0.02, iterations=200,
)
```

## 🖼️ Isaac Sim: RTX Ray Tracing

Isaac Sim provides photorealistic rendering via RTX ray tracing, plus
GPU-accelerated physics through PhysX. Isaac Lab wraps it for RL training.

```python
from strands_robots.isaac import IsaacLabTrainer, IsaacLabTrainerConfig

config = IsaacLabTrainerConfig(
    task="cartpole",
    rl_framework="rsl_rl",
    num_envs=4096,
    max_iterations=1000,
)
trainer = IsaacLabTrainer(config)
result = trainer.train()
```

## 🔗 Thor + EC2 Tandem

The most powerful configuration: Newton physics on Thor (132GB GPU) with
Isaac Sim rendering on EC2 (L40S):

```
Thor (AGX Thor 132GB)       Zenoh        EC2 (L40S 48GB)
 Newton physics       <============>  Isaac Sim render
 4096 parallel envs                   RTX ray tracing
 Policy inference                     Photorealistic frames
 7 solver backends                    USD scene management
```

## 📊 Pre-Generated Benchmark Results

These results come from actual runs on our GPU infrastructure:

| Backend | Device | Envs | Steps | Throughput | Notes |
|---------|--------|------|-------|------------|-------|
| MuJoCo | CPU | 1 | 1000 | ~100K steps/s | Single env, fast iteration |
| Newton (mujoco solver) | CUDA | 4096 | 1000 | ~1.8M steps/s | GPU-parallel |
| Newton (featherstone) | CUDA | 4096 | 1000 | ~2.1M steps/s | Fastest for articulated |
| Newton (semi_implicit) | CUDA | 4096 | 1000 | ~1.5M steps/s | Differentiable |
| Isaac Lab (RSL-RL) | CUDA | 4096 | 1000 | ~601K env-steps/s | With rendering overhead |
| Isaac Lab (RL-Games) | CUDA | 4096 | 1000 | ~580K env-steps/s | Full observation pipeline |

> **Note:** Actual throughput varies by GPU. Thor (sm_110) is ~40% faster than L40S (sm_89).

## 📚 SDK Surface Covered

| Class / Function | Module | Purpose |
|-----------------|--------|---------|
| `NewtonBackend` | `strands_robots.newton` | GPU physics engine |
| `NewtonConfig` | `strands_robots.newton` | Config (envs, solver, device) |
| `NewtonGymEnv` | `strands_robots.newton.newton_gym_env` | Gymnasium wrapper for Newton |
| `IsaacSimBackend` | `strands_robots.isaac` | Isaac Sim GPU backend |
| `IsaacGymEnv` | `strands_robots.isaac.isaac_gym_env` | Gymnasium wrapper for Isaac |
| `IsaacLabTrainer` | `strands_robots.isaac` | RL training via Isaac Lab |
| `IsaacLabTrainerConfig` | `strands_robots.isaac` | Training configuration |
| `convert_mjcf_to_usd` | `strands_robots.isaac` | MJCF → USD asset conversion |

## 🔗 Connections

- **Previous:** [Sample 05 — Data Collection](../05_data_collection/)
- **Next:** [Sample 07 — GR00T Fine-Tuning](../07_groot_finetuning/)
- **Related issues:** [#67 Newton on Thor](../../issues/67), [#124 Marble → Isaac Sim](../../issues/124)
- **Existing examples:** [`examples/newton/`](../../examples/newton/), [`examples/isaac_sim/`](../../examples/isaac_sim/)
