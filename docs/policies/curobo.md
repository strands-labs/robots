---
description: NVIDIA cuRobo collision-aware motion planning - in-process CUDA, no network round-trip, goal via target_pose / target_joints kwargs.
---

# cuRobo

[`CuroboPolicy`](https://github.com/strands-labs/robots/blob/main/strands_robots/policies/curobo/policy.py)
wraps NVIDIA [cuRobo](https://curobo.org/)'s `MotionPlanner`. Unlike the
sidecar VLA providers (GR00T, Cosmos 3), cuRobo runs **in the same process**
as a CUDA library: there is no network round-trip, but a CUDA-capable GPU is
required. It is a non-VLA, collision-aware motion planner - it reads its goal
from `**kwargs` (`target_pose` / `target_joints`), ignores camera frames
(`requires_images = False`), and never parses the instruction string for
control.

## Install

cuRobo is **not** on PyPI (the `nvidia-curobo` PyPI package is an unrelated
v0.1 squatter). Install from the upstream source repository, then this
package:

```bash
git clone https://github.com/NVlabs/curobo.git
pip install -e ./curobo
pip install "strands-robots[curobo]"   # extra is currently empty;
                                       # reserved for a future stable
                                       # cuRobo PyPI wheel
```

This policy targets cuRobo's restructured `main` API (`MotionPlanner` /
`MotionPlannerCfg` / `DeviceCfg` / `JointState` / `GoalToolPose`). The
on-device cuRobo APIs are still moving on `main` until upstream cuts a stable
release; if you hit a fresh API shift, pin to a known-good commit or open an
issue with the cuRobo SHA you tested.

## Quickstart

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "curobo",                      # alias: "cumotion"
    robot_config="franka.yml",     # any cuRobo built-in YAML, or a dict
    action_horizon=16,
)

actions = policy.get_actions_sync(
    observation_dict={
        "observation.state": [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854],
    },
    instruction="reach for the red block",   # ignored by planners
    target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],   # [x,y,z, qw,qx,qy,qz]
)
```

## Parameters

```python
CuroboPolicy(
    robot_config="franka.yml",     # cuRobo built-in YAML name, a path, or a dict
    world_config=None,             # initial collision world (dict | None)
    action_horizon=16,             # waypoints streamed per get_actions() call
    device_cfg=None,               # None | "cuda:0" | torch.device | DeviceCfg
    motion_planner_kwargs=None,    # extra kwargs forwarded to MotionPlannerCfg.create
    motion_gen=None,               # pre-built planner (tests / advanced); skips build
    warmup=True,                   # warm the planner at construction
    # Legacy 0.7.x aliases (pass only one of each pair):
    tensor_args=None,              # alias of device_cfg=
    motion_gen_kwargs=None,        # alias of motion_planner_kwargs=
)
```

Supplying both a canonical kwarg and its legacy alias (e.g. `device_cfg=` and
`tensor_args=`) raises `ValueError` rather than silently picking one.

## Goal kwargs

cuRobo shares the non-VLA goal vocabulary with the rest of the planner family,
so a goal can flow across providers without coupling to a backend:

| Key | Type | Meaning |
|-----|------|---------|
| `target_pose` | `list[float]` | Cartesian goal `[x, y, z, qw, qx, qy, qz]` in the base frame |
| `target_joints` | `dict[str, float]` | Joint-space goal keyed by joint name (rad / m) |
| `world_update` | `dict \| None` | Per-call collision-scene refresh |
| `replan` | `bool` | Force a fresh plan even if cached waypoints remain |

Pass exactly one of `target_pose` / `target_joints`. When neither is given,
the policy makes a best-effort parse of a JSON `target_pose` / `target_joints`
payload embedded in the instruction (for LLM-agent flows); if none is found it
raises `ValueError`. Each `{...}` object in the instruction is decoded on its
own, so prose braces before or after the payload are harmless, and only a
**top-level** goal field counts - a goal nested inside a wrapper object
(`{"goal": {"target_pose": [...]}}`) is not read as a goal.

## Trajectory chunking

The full collision-free trajectory is planned and cached on the first call.
Each subsequent call yields up to `action_horizon` waypoints from the cache so
the 50Hz execution loop in `Robot` streams per-step joint targets without
re-planning. Force a fresh plan with `replan=True` (or `policy.reset()`) when
the world changes mid-rollout. `world_update` is forwarded to
`MotionPlanner.update_scene` (with a legacy `update_world` fallback) for
per-call collision-scene refresh.

## In simulation

```python
from strands_robots import Robot

sim = Robot("panda")              # sim-by-default; needs a CUDA GPU for cuRobo
sim.run_policy(
    robot_name="panda",
    instruction="",               # ignored by the planner
    policy_provider="curobo",
    policy_config={"robot_config": "franka.yml", "action_horizon": 16},
    policy_kwargs={"target_pose": [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]},
    duration=10.0,
    control_frequency=50.0,
)
```

`policy_config` and `policy_kwargs` are two different sinks. `policy_config`
is expanded into the policy **constructor**; the per-call goal belongs in
`policy_kwargs`, which the runner hands to every `get_actions()` call. Passing
`target_pose=` directly to `run_policy` raises `TypeError` - it has no such
parameter and no `**kwargs`.

The mesh path forwards the same goal vocabulary:
`mesh.tell(peer, "...", policy_provider="curobo", target_pose=[...])`.
`Robot.start_task` does not - it accepts only `instruction`, `policy_port`,
`policy_host`, `policy_provider` and `duration`.

## See also

- [Policy overview](overview.md)
- [MoveIt2](moveit2.md) - ROS 2 sidecar collision-aware planning (non-VLA).
- [GR00T](groot.md) - ZMQ service VLA.
- [Cosmos 3](cosmos3.md) - WebSocket VLA.
- [LeRobot Local](lerobot-local.md) - in-process HF models.
- [Custom policies](custom-policies.md) - implement the non-VLA goal-kwargs contract.
- [cuRobo project](https://curobo.org/)
