---
description: NVIDIA GR00T Whole-Body-Control (SONIC) humanoid locomotion - in-process ONNX, no GPU required, goal via target_velocity kwargs.
---

# WBC (Whole-Body-Control)

[`WBCPolicy`](https://github.com/strands-labs/robots/blob/main/strands_robots/policies/wbc/policy.py)
wraps NVIDIA's
[GR00T Whole-Body-Control](https://github.com/NVlabs/GR00T-WholeBodyControl)
(SONIC / decoupled-WBC) ONNX controllers for deploy-grade humanoid locomotion
on the Unitree G1. Like [cuRobo](curobo.md) it runs **in the same process**
(via ONNX Runtime - no sidecar, no network round-trip), but unlike cuRobo it
needs no GPU: the ONNX sessions run on CPU.

It is a non-VLA, locomotion controller: it reads its goal from the well-known
locomotion `**kwargs` (`target_velocity`), ignores camera frames
(`requires_images = False`), and never parses the instruction string for
control. The controller drives the **15 leg+waist DOFs** of the G1; the arm
joints are held at their nominal defaults. Layering an upper-body manipulation
policy (e.g. GR00T) on top of WBC locomotion is the job of a future
`CompositePolicy`, out of scope for this provider.

## Install

```bash
pip install "strands-robots[wbc]"            # onnxruntime only - light, no torch
pip install "strands-robots[wbc,sim-mujoco]" # + MuJoCo to drive the G1 in sim
```

No model weights are bundled. Download a GR00T-WBC (SONIC) checkpoint under the
NVIDIA Open Model License (e.g.
[`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC)) into a
directory containing `policy.onnx` (plus an optional `walk_policy.onnx` and
`config.json`).

## Quickstart

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "wbc",                                  # shorthand: "sonic"
    checkpoint="/path/to/GEAR-SONIC",       # dir with policy.onnx (+ walk_policy.onnx)
    walk=True,
)

actions = policy.get_actions_sync(
    observation_dict={"observation.state": [0.0] * 29},  # G1 joint positions
    instruction="walk forward",             # ignored by the controller
    target_velocity=[0.5, 0.0, 0.0],        # [vx, vy, omega] (m/s, m/s, rad/s)
)
# actions == [{"left_hip_pitch_joint": .., ..., "waist_pitch_joint": ..}]
# one per-tick dict of 15 leg+waist joint targets (closed-loop, not a chunk)
```

## Parameters

```python
WBCPolicy(
    checkpoint="/path/to/GEAR-SONIC",  # dir with policy.onnx, a direct .onnx path, or an HF id
    config=None,                       # WBCConfig | path | dict | None (None -> config.json in checkpoint)
    walk=True,                         # load + prefer walk_policy.onnx for locomotion
    target_velocity=None,              # constructor-time default [vx, vy, omega] (per-call kwarg overrides)
    allow_missing_models=False,        # test seam: skip eager ONNX load (inject a stub session)
)
```

A missing `onnxruntime` or a missing checkpoint raises `RuntimeError` at
construction - WBC never falls back to silent zero torques.

## Goal kwargs

WBC reads locomotion commands from `**kwargs`, sharing the non-VLA goal
vocabulary so a command can flow through `run_policy` / mesh `tell()` without
coupling to a backend:

| Key | Type | Meaning |
|-----|------|---------|
| `target_velocity` | `list[float]` | Locomotion command `[vx, vy, omega]` (m/s, m/s, rad/s) |
| `target_orientation` | `list[float]` | Optional extra command/style channels appended after `[vx, vy, omega]` |

A per-call `target_velocity` overrides the constructor-time default. With no
command at all the controller holds a standing balance (zero velocity).

## Control contract

WBC reproduces the upstream `GearWbcController` loop:

- **Two ONNX sessions** - a main `policy.onnx` and an optional `walk_policy.onnx`,
  loaded once at construction. The walk policy is selected when `walk=True` and
  the command requests motion; otherwise the main (balance) policy runs.
- **Observation** - an 86-dim frame (command + base angular velocity +
  projected gravity + scaled joint positions/velocities + previous action),
  stacked over `obs_history_len` via a deque.
- **Action** - the network emits a 15-dim joint-position *offset*; the policy
  forms absolute targets `target_q = default_angles + action_scale * raw` and
  returns them keyed by actuator name. For torque-actuated MuJoCo, convert with
  the upstream PD law via `policy.compute_torques(target, q, dq)`.

## Actuator mapping

WBC output index `i` drives `WBC_G1_LEG_WAIST_JOINTS[i]` - an explicit table
(no positional guessing). `set_robot_state_keys` validates that the robot's
first 15 joints match this order and raises otherwise, so a mismatched model
can never silently actuate the wrong joints:

```
left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint,
left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint,
right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint,
right_knee_joint, right_ankle_pitch_joint, right_ankle_roll_joint,
waist_yaw_joint, waist_roll_joint, waist_pitch_joint
```

## In simulation

```python
from strands_robots import Robot

sim = Robot("unitree_g1")             # sim-by-default; CPU ONNX, no GPU needed
sim.run_policy(
    robot_name="unitree_g1",
    instruction="walk forward",       # ignored by the controller
    policy_provider="wbc",
    policy_config={"checkpoint": "/path/to/GEAR-SONIC", "walk": True},
    policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},   # per-call locomotion goal
    duration=10.0,
    control_frequency=50.0,
    action_horizon=1,                 # WBC is closed-loop per tick
)
```

The per-call command rides through `run_policy`'s `policy_kwargs` to
`policy.get_actions(..., target_velocity=[...])`. A *static* walk can also be
set once at construction via `policy_config={"checkpoint": ..., "target_velocity":
[0.5, 0.0, 0.0]}` (the value forwarded to the policy constructor), which is how a
command reaches the policy over the mesh `tell()` path.

## See also

- [Policy overview](overview.md)
- [cuRobo](curobo.md) - in-process CUDA collision-aware planning (non-VLA).
- [MoveIt2](moveit2.md) - ROS 2 sidecar collision-aware planning (non-VLA).
- [GR00T](groot.md) - ZMQ service VLA (manipulation upper body).
- [Custom policies](custom-policies.md) - implement the non-VLA goal-kwargs contract.
- [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
