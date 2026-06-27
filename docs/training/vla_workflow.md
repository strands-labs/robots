---
description: End-to-end VLA workflow on the Unitree G1 - collect teleop data, fine-tune Isaac-GR00T N1.7, deploy with SONIC whole-body control.
---

# VLA-on-G1 Workflow

The full Vision-Language-Action (VLA) pipeline on the Unitree G1 humanoid:
**collect teleop data** (LeRobot recording) -> **fine-tune Isaac-GR00T N1.7**
(GR00T Trainer) -> **deploy with SONIC whole-body control** (WBC provider).

Each piece ships individually in `strands-robots`; this page documents how they
compose into one coherent pipeline. The companion example script runs the chain
end-to-end:

```bash
# Quick demo (record + deploy with mock, ~10s on CPU):
python examples/vla_g1_workflow.py

# Full pipeline with real fine-tuning (Docker + GPU):
python examples/vla_g1_workflow.py --tune --base-model nvidia/GR00T-N1.7-3B

# Deploy-only with downloaded SONIC weights:
python examples/vla_g1_workflow.py --checkpoint /path/to/GEAR-SONIC
```

## Pipeline stages

### 1. Record  - collect locomotion data

Drive the G1 (in sim or on real hardware via LeRobot teleop) and capture a
`LeRobotDataset`. The recording pipeline is the same one the existing
[`03_record_dataset.py`](https://github.com/strands-labs/robots/blob/main/examples/03_record_dataset.py)
hero example demonstrates  - adapted for the 29-DOF humanoid:

Drive the G1 with **WBC** (the merged SONIC whole-body controller) so the
captured dataset is genuine walking motion:

```python
from strands_robots import Robot
from strands_robots.policies import create_policy
from strands_robots.policies.wbc import install_wbc_torque_control

sim = Robot("unitree_g1", mesh=False)
policy = create_policy("wbc", checkpoint="/path/to/GEAR-SONIC", walk=True)

# WBC emits joint-POSITION targets, but the G1 scene's actuators are
# position-servos (uniform kp=500) that override SONIC's tuned per-joint PD -
# so writing the targets directly makes the robot fall. install_wbc_torque_control
# flips the G1's actuators to torque mode and applies the SONIC PD law at the
# correct decimation, so the G1 actually WALKS. Pair it with control_frequency=50.
install_wbc_torque_control(sim, policy, "unitree_g1")

sim.start_recording(
    repo_id="local/g1_locomotion",
    root="/tmp/g1_dataset",
    fps=30, task="walk forward", overwrite=True,
)
sim.run_policy(
    robot_name="unitree_g1",
    policy_object=policy,
    instruction="walk forward",
    policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},  # [vx, vy, omega]
    action_horizon=1,
    control_frequency=50.0,
    n_steps=200,
)
sim.stop_recording()
```

The `vla_g1_workflow.py` example wires exactly this up behind a flag:

```bash
python examples/vla_g1_workflow.py --record-checkpoint /path/to/GEAR-SONIC
```

Two ingredients make WBC close its loop through `sim.run_policy`:

1. The MuJoCo backend's observation surfaces the joint velocities and base IMU
   signals (`<joint>.vel`, `base_quat`, `base_ang_vel`) that WBC's balance
   controller consumes - no manual observation wiring.
2. `install_wbc_torque_control` converts WBC's position targets into joint
   torques via the SONIC PD law on torque-mode actuators (the standard scene
   ships stiff position-servos that the gait cannot drive).

For data collection from a different source, swap the WBC policy for a LeRobot
teleop driver, a VR controller, or `MockPolicy` (synthetic, runs with no weights
or hardware - the quick-demo default). The dataset format is identical either way.

### 2. Fine-tune  - post-train Isaac-GR00T N1.7

Use the [`Trainer` abstraction](overview.md) with the `"groot"` provider to
post-train a GR00T N1.7 base model on the recorded G1 data:

```python
from strands_robots.training import create_trainer, TrainSpec

trainer = create_trainer("groot")
spec = TrainSpec(
    dataset_root="/tmp/g1_dataset",
    base_model="nvidia/GR00T-N1.7-3B",
    output_dir="/tmp/g1_finetuned",
    steps=1000,
    extra={"embodiment": "unitree_g1", "data_config": "unitree_g1"},
)
result = trainer.train(spec)
checkpoint = trainer.export(spec, result.checkpoint_dir)
```

Under the hood, `Gr00tTrainer` orchestrates the `gr00t_inference` Docker tool's
training pipeline. This stage requires Docker + a GPU and takes minutes to hours
depending on dataset size and step count.

> **Note:** The `gr00t_inference` tool's `unitree_g1` embodiment is marked
> `[posttrain]`  - meaning it requires a fine-tuned checkpoint, not the base
> model directly. The base model (`nvidia/GR00T-N1.7-3B`) is the starting point
> for fine-tuning; the output is the checkpoint you deploy.

### 3. Deploy  - SONIC whole-body control

Load the fine-tuned (or pre-trained SONIC) checkpoint with the `wbc` provider
and drive the G1's 15 leg+waist DOFs:

```python
from strands_robots import Robot
from strands_robots.policies import create_policy

sim = Robot("unitree_g1", mesh=False)
policy = create_policy("wbc", checkpoint="/tmp/g1_finetuned", walk=True)

sim.run_policy(
    robot_name="unitree_g1",
    policy_object=policy,
    instruction="walk forward",
    policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},
    duration=10.0,
    control_frequency=50.0,
    action_horizon=1,
)
```

For real deploy-grade locomotion (with the upstream torque-PD law), use the
[torque-control harness](../policies/wbc.md#watching-it-walk-torque-control-deploy):

```bash
python examples/wbc_g1_torque_deploy.py --checkpoint /tmp/g1_finetuned --vx 0.5
```

## Prerequisites

| Stage | Install | External |
|-------|---------|----------|
| Record | `pip install "strands-robots[sim-mujoco,lerobot]"` | None (sim) |
| Fine-tune | `pip install "strands-robots[groot-service]"` | Docker + GPU |
| Deploy | `pip install "strands-robots[wbc,sim-mujoco]"` | None (CPU ONNX) |

## Upstream references

- [GR00T Whole-Body-Control VLA workflow tutorial](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)
- [GR00T-WholeBodyControl repo](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [WBC policy docs](../policies/wbc.md)
- [Training overview](overview.md) (the `Trainer` abstraction)
- [Dataset recording example](https://github.com/strands-labs/robots/blob/main/examples/03_record_dataset.py)

## See also

- [`07_post_tune_any_policy.py`](https://github.com/strands-labs/robots/blob/main/examples/07_post_tune_any_policy.py)  - the same record->train->deploy loop for arm manipulation (SO-100 + LeRobot ACT)
- [WBC provider](../policies/wbc.md)  - the deploy-stage policy (observation layout, command kwargs, torque harness)
- [GR00T provider](../policies/groot.md)  - the inference-stage policy (ZMQ + Docker)
