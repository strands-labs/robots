# ProtoMotions — reference-motion tracking for the Unitree G1

`ProtoMotionsPolicy` wraps a ProtoMotions **Generalist Tracking Policy** (GTP)
ONNX export. Given a reference motion clip it emits balanced PD joint targets
for the Unitree G1's 29 actuators, tracking that clip while keeping the robot
upright.

It is the tracking half of a two-stage pipeline. A *kinematic* generator such as
[`KimodoPolicy`](./kimodo.md) (text-to-motion diffusion) or
[`MotionBricksPolicy`](./motionbricks.md) produces a `qpos` sequence with no
notion of balance; the tracker turns that sequence into physics. Compare
[`WBC`](./wbc.md), which takes a velocity/height *command* rather than a
reference clip and so cannot follow a whole-body pose trajectory.

## Install

```bash
pip install "strands-robots[protomotions]"
```

That pulls `onnxruntime` (runs the graph), `pyyaml` (reads the
`unified_pipeline.yaml` sidecar) and `huggingface_hub` (fetches a checkpoint from
a model id). Weights are not bundled. Building a reference clip from `qpos` with
[`qpos_to_motion_data`](#bridging-a-qpos-clip) additionally needs MuJoCo, which
ships in `strands-robots[sim-mujoco]`.

## Run a clip in simulation

```python
from strands_robots import Robot
from strands_robots.policies.protomotions import (
    ProtoMotionsPolicy,
    qpos_to_motion_data,
)

sim = Robot("unitree_g1", mode="sim")

motion = qpos_to_motion_data(qpos, fps=30, proto_mjcf_path=mjcf)
policy = ProtoMotionsPolicy(
    onnx_path="unified_pipeline.onnx",
    yaml_path="unified_pipeline.yaml",
    motion=motion,
)

sim.run_policy(
    robot_name="unitree_g1",
    policy_object=policy,
    n_steps=1000,
    control_frequency=50.0,
)
```

## The observation contract

The tracker consumes four inputs per tick. Three are ordinary proprioception —
joint positions, joint velocities, and the floating base's angular velocity —
and the runtime already publishes all three (`<joint>`, `<joint>.vel` and
`base_ang_vel`, which is the base freejoint's `qvel[3:6]` and so is already in
the body frame the network wants).

The fourth is the **world orientation of the anchor link**, `torso_link` on the
G1. This one is not derivable from the observation's floating-base signals:
`base_quat` is the *pelvis*, and the torso differs from it by the three waist
joints. On a G1 sweeping its waist through 0.6 rad the two frames diverge by up
to 42 degrees, so substituting the base would feed the network a wrong frame
rather than an approximation of the right one.

The policy therefore declares the link it needs:

```python
policy.required_bodies        # ('torso_link',)
```

and the runtime resolves that name once per rollout and merges
`body.torso_link.quat` into every observation it hands to `get_actions`. Nothing
is required of the caller — this is the same "policy declares, runtime supplies"
contract as [`requires_images`](./overview.md).

A caller assembling observations by hand (a hardware loop reading an IMU, for
instance) can pass the signals directly instead:

```python
await policy.get_actions(
    obs,
    "",
    anchor_rot_xyzw=[x, y, z, w],   # anchor link, world frame
    root_ang_vel_local=[wx, wy, wz],
)
```

If neither the declared body pose nor an explicit override is present the policy
raises, naming the key it wanted. It will not fall back to `base_quat` unless
the config's anchor body *is* the floating base, in which case the two are the
same frame by definition.

Joint velocities are treated the same way: an absent `<joint>.vel` is refused
rather than substituted with zero. Velocity is tracker feedback, and zeros are a
plausible-looking value that quietly degrades tracking.

## Bridging a qpos clip

`qpos_to_motion_data` turns a `[T, 7 + 29]` MuJoCo `qpos` sequence into the
cache the tracker plays. It runs forward kinematics through the ProtoMotions
MJCF to recover per-body world poses, finite-differences velocities at the
source rate, and resamples onto the tracker's `control_dt` (0.02 s):

```python
cache = qpos_to_motion_data(qpos, fps=30, proto_mjcf_path=mjcf)
cache["num_frames"], cache["control_dt"]
```

`MotionPlayer` accepts that dict, an `.npz` written by
`MotionPlayer.save_cache_npz`, or a raw ProtoMotions `.pt`.

### Motion files are read with a restricted unpickler

An `.npz` cache is read by NumPy and needs no torch. A raw `.pt` is read with
`torch.load(..., weights_only=True)`, which accepts tensors and plain scalars -
the documented payload above - and refuses anything else.

That restriction matters because a motion file travels: clips get downloaded,
shared between machines and committed to dataset repos. The unrestricted
unpickler runs whatever `__reduce__` a file names *while reading it*, so
accepting one would make playing a third-party motion enough to execute code on
the machine that plays it. If a `.pt` is refused, re-save it as a dict of
tensors, or convert it once with `save_cache_npz` and load the `.npz`.

## Per-episode reset

`reset(seed=...)` rewinds the playhead and clears the historical-action buffer.
The runtime calls it once per episode, so a multi-episode `eval_policy` run
replays the clip from frame 0 each time. The seed is accepted for interface
parity and ignored: given a reference clip and an observation the tracker is
deterministic, holding no RNG state.

## Configuration

`ProtoMotionsConfig` is a frozen dataclass mirroring the checkpoint's
`unified_pipeline.yaml`: the 29 joint names in ONNX action order, the 33 body
names, the anchor and root body indices, per-joint PD gains, timing, and the
lookahead offsets for the future-reference window. Pass `yaml_path=` to load a
sidecar; omit it to use the defaults, which match the shipped export.

| Field | Default | Meaning |
| --- | --- | --- |
| `joint_names` | 29 G1 joints | ONNX action order |
| `anchor_body_index` | `16` (`torso_link`) | Link whose world rotation the network reads |
| `root_body_index` | `0` (`pelvis`) | Floating base |
| `control_dt` | `0.02` | Seconds per control tick (50 Hz) |
| `future_step_indices` | `(1, 2, 4, 8)` | Lookahead offsets, in control steps |

## Testing without weights

`session=` injects anything satisfying the `ProtoMotionsSession` protocol
(`run(output_names, inputs) -> list[np.ndarray]`), so the observation to action
mapping can be exercised with no onnxruntime, no weights and no GPU:

```python
policy = ProtoMotionsPolicy(session=stub, motion=cache)
```

Outputs are paired to the names in `config.onnx_out_names` rather than read
positionally, so an export that declares them in another order still feeds
`joint_pos_targets` to the PD loop; an export missing that output is refused by
name.
