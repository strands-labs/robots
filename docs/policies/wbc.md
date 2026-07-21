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

No model weights are bundled and there is no default download: a bare
`create_policy("wbc")` raises an actionable error instead of fetching the wrong
model family. The decoupled-WBC G1 policies live in the
[`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl)
git-LFS tree at `decoupled_wbc/sim2mujoco/resources/robots/g1/policy/`:

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git   # 4.6G LFS
mkdir -p /path/to/grootwbc-g1
cp GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/\
GR00T-WholeBodyControl-{Balance,Walk}.onnx /path/to/grootwbc-g1/
```

The canonical `GR00T-WholeBodyControl-Balance.onnx` / `-Walk.onnx` filenames are
accepted verbatim - no rename is needed. The conventional `policy.onnx` /
`walk_policy.onnx` names also work and take precedence when both are present, so
either layout is valid:

```text
/path/to/grootwbc-g1/
    GR00T-WholeBodyControl-Balance.onnx   # main (Balance) policy
    GR00T-WholeBodyControl-Walk.onnx      # optional walk policy
    config.json                           # optional
```

Note: the HuggingFace repo
[`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC) is the SONIC VLA
inference stack (encoder/decoder/planner ONNX), **not** this decoupled-WBC
Balance/Walk family; passing it as a checkpoint raises a clear error.

## Quickstart

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "wbc",                                  # shorthand: "sonic"
    checkpoint="/path/to/grootwbc-g1",       # dir with policy.onnx (+ walk_policy.onnx)
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
    checkpoint="/path/to/grootwbc-g1",  # dir with policy.onnx, a direct .onnx path, or an HF id
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
| `target_velocity` | `list[float]` | Locomotion command `[vx, vy, omega]` (m/s, m/s, rad/s). Scaled by `cmd_scale` (`[2.0, 2.0, 0.5]`) into the observation's command block. |
| `target_orientation` | `list[float]` | Target base `[roll, pitch, yaw]` (rad), written to command slots `[4:7]`. Defaults to the config `rpy_cmd` (`[0,0,0]`). |
| `height` | `float` | Target base height (m), written to command slot `[3]`. Defaults to the config `height_cmd` (`0.74`). |

A per-call `target_velocity` overrides the constructor-time default. With no
command at all the controller holds a standing balance (zero velocity, default
height + level orientation).

## Control contract

WBC reproduces the upstream `GearWbcController` loop (NVlabs/GR00T-WholeBodyControl
`decoupled_wbc/sim2mujoco`, `run_mujoco_gear_wbc.py` + `g1_gear_wbc.yaml`):

- **Two ONNX sessions** - a main `policy.onnx` and an optional `walk_policy.onnx`,
  loaded once at construction. Selection matches upstream: when the **raw**
  velocity-command norm is `<= 0.05` the robot is "standing" and the main policy
  runs; above that the walk policy runs (when `walk=True`).
- **Observation** - an 86-dim frame stacked over `obs_history_len` (default 6,
  so the network input is `86 * 6 = 516`):
  - command `[0:7]` = `[vx*2.0, vy*2.0, omega*0.5, height, roll, pitch, yaw]`
  - base angular velocity `[7:10]` (scaled by `ang_vel_scale=0.5`)
  - projected gravity `[10:13]`
  - joint positions `[13:28]` (minus `default_angles`, scaled by `dof_pos_scale`)
  - joint velocities `[28:43]` (scaled by `dof_vel_scale=0.05`)
  - previous action `[43:58]`; indices `[58:86]` are a reserved (zero) tail.
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
    policy_config={"checkpoint": "/path/to/grootwbc-g1", "walk": True},
    policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},   # per-call locomotion goal
    duration=10.0,
    control_frequency=50.0,
    action_horizon=1,                 # WBC is closed-loop per tick
)
```

`run_policy` writes a policy's joint-position **targets** straight to the sim's
actuators, but the stock Menagerie G1 ships *position-servo* actuators with a
uniform `kp=500` gain that overrides SONIC's tuned per-joint PD - so driving
those servos directly diverges and the robot falls. To make this quickstart
just work, `run_policy` auto-detects a `WBCPolicy` on a position-servo scene and
installs the torque shim (the `WBCTorqueController` PD->torque loop) for the
duration of the call, then restores the actuators afterwards. With the real
`GR00T-WholeBodyControl-{Balance,Walk}.onnx` weights and `target_velocity =
[0.5, 0, 0]` the base advances ~1.9 m over 5 s while holding pelvis height
~0.75 m and staying upright. Pass `wbc_install_torque_control=False` to opt out
(e.g. to drive a torque-actuated scene directly or manage the controller
yourself):

```python
sim.run_policy(..., policy_provider="wbc", wbc_install_torque_control=False)
```

The per-call command rides through `run_policy`'s `policy_kwargs` to
`policy.get_actions(..., target_velocity=[...])`. A *static* walk can also be
set once at construction via `policy_config={"checkpoint": ..., "target_velocity":
[0.5, 0.0, 0.0]}` (the value forwarded to the policy constructor), which is how a
command reaches the policy over the mesh `tell()` path.

> **Note:** `policy_kwargs` is wired on the control/deploy path
> (`run_policy` / `start_policy` / mesh `tell()`), not the evaluation path.
> `eval_policy` / `evaluate_benchmark` are instruction-driven (built for
> task-success benchmarks); to evaluate WBC at a fixed velocity, set it once via
> the constructor `target_velocity` (above). Per-episode velocity variation in
> eval is out of scope for this provider.

## Watching it walk (torque-control deploy)

The `sim.run_policy` path above already produces a stable gait: it auto-installs
the torque shim that converts the policy's joint-position **targets** to
**torque** via SONIC's per-joint PD law (the stock position-servo actuators
alone diverge - see the note in [In simulation](#in-simulation)).

For a self-contained, low-level reference - building the torque-actuated G1,
running `policy.compute_torques(...)` at `control_decimation=4`, and rendering -
[`examples/wbc/wbc_g1_torque_deploy.py`](https://github.com/strands-labs/robots/blob/main/examples/wbc/wbc_g1_torque_deploy.py)
reproduces the upstream deploy loop directly - torque motors, `policy.compute_torques(...)` at
`control_decimation=4`, whole-body observation with real joint velocities + base
IMU - and is the right way to see the G1 actually locomote:

```bash
python examples/wbc/wbc_g1_torque_deploy.py --checkpoint /path/to/grootwbc-g1 \
    --duration 5 --vx 0.5 --mp4 /tmp/g1_walk.mp4
```

With the real `GR00T-WholeBodyControl-{Balance,Walk}.onnx` weights this produces
a stable forward walk (the base advances ~0.38 m/s for a 0.5 m/s command while
holding height); a standing command (`--vx 0`) holds balance in place.

<figure markdown>
  ![Unitree G1 walking forward under WBC torque control](../assets/wbc/g1_walk.gif)
  <figcaption>Unitree G1 under <code>WBCPolicy</code> (GR00T-WBC SONIC, <code>walk_policy.onnx</code>)
  commanded at <code>vx = 0.5 m/s</code> — the torque-PD deploy loop in MuJoCo (headless).
  The base advances ~2.3 m over 6 s (~0.38 m/s) while holding pelvis height ~0.75 m and
  staying upright. Produced by <code>examples/wbc/wbc_g1_torque_deploy.py --vx 0.5 --mp4</code>
  (<a href="../../assets/wbc/g1_walk.mp4">MP4</a>).</figcaption>
</figure>

## Composing an upper body (manipulation on top of WBC)

`WBCPolicy` drives the 15 leg+waist DOFs and holds the arms at their defaults.
To layer a manipulation policy on the arms while WBC keeps the robot balanced
and walking, wrap both in
[`CompositePolicy`](custom-policies.md): the lower policy owns the
legs+waist joints, the upper policy owns the arms, and the composite queries
both each tick and merges their action dicts by joint name.

```python
from strands_robots.policies import CompositePolicy, create_policy
from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, WBC_G1_LEG_WAIST_JOINTS

ARM_JOINTS = WBC_G1_ALL_JOINTS[len(WBC_G1_LEG_WAIST_JOINTS):]  # the 14 arm DOFs

lower = create_policy("wbc", checkpoint="/path/to/grootwbc-g1")
upper = create_policy("groot", port=5555)        # or pi0 / MolmoAct / any Policy
policy = CompositePolicy(
    lower=lower,
    upper=upper,
    lower_joints=WBC_G1_LEG_WAIST_JOINTS,   # legs + waist
    upper_joints=ARM_JOINTS,                 # both arms
)
```

Routing is explicit: with `lower_joints` / `upper_joints` set, each child
contributes only its own joint group (foreign keys are dropped). Left default,
the lower policy takes precedence on any name it emits and the upper policy
fills the rest. A genuine ownership conflict (both children claim the same
joint) is raised, never silently resolved. The merged chunk length is the
shorter of the two, so a per-tick controller (WBC, `execution_horizon == 1`) is
never starved by a slower chunk-emitting manipulation policy.

[`examples/wbc/wbc_g1_composite.py`](https://github.com/strands-labs/robots/blob/main/examples/wbc/wbc_g1_composite.py)
runs the composite in the torque-deploy loop with a zero-dependency scripted
arm-wave as the upper body (swap in `--upper-port` for a real GR00T server):

```bash
python examples/wbc/wbc_g1_composite.py --checkpoint /path/to/grootwbc-g1 \
    --duration 5 --vx 0.4 --mp4 /tmp/g1_composite.mp4
```

<figure markdown>
  ![Unitree G1 walking under WBC while the composite upper body waves its arms](../assets/wbc/g1_composite.gif)
  <figcaption>Unitree G1 under <code>CompositePolicy</code>: <code>WBCPolicy</code> (GR00T-WBC SONIC)
  drives the legs+waist for a <code>vx = 0.4 m/s</code> walk while the upper-body policy drives the
  arms - the base advances ~1.55 m over 5 s while the arms move, in MuJoCo (headless).
  Produced by <code>examples/wbc/wbc_g1_composite.py --vx 0.4 --mp4</code>
  (<a href="../../assets/wbc/g1_composite.mp4">MP4</a>).</figcaption>
</figure>

## Gait-clock variant

NVIDIA's reference repo ships a second G1 controller - a single-policy
**gait-clock** variant (95-dim observation, an 8-wide command with a
`freq_cmd` step-frequency slot, and a 2-dim bipedal phase clock). It is
implemented by `WBCGaitPolicy` (provider `wbc_gait`). See
[WBC gait-clock variant](wbc_gait.md).

## See also

- [Policy overview](overview.md)
- [cuRobo](curobo.md) - in-process CUDA collision-aware planning (non-VLA).
- [MoveIt2](moveit2.md) - ROS 2 sidecar collision-aware planning (non-VLA).
- [GR00T](groot.md) - ZMQ service VLA (manipulation upper body).
- [Custom policies](custom-policies.md) - implement the non-VLA goal-kwargs contract.
- [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
