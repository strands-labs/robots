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

### Orientations need not be exactly unit

Every rotation the tracker derives — body-framing the root angular velocity,
and extracting a yaw for heading alignment — is computed from a formula that
mixes a quadratic term in the quaternion components with a constant, so the
quaternion's scale does not cancel. Both helpers therefore normalise their
input, and a quaternion scaled by any positive factor gives the same answer:

```python
from strands_robots.policies.protomotions import extract_yaw_quat

extract_yaw_quat(q)          # same heading as
extract_yaw_quat(q * 0.92)   # this
```

That is worth knowing when assembling observations by hand. An IMU reading
drifts off unit, and an orientation obtained by *linearly* interpolating two
samples is short by up to about 8% — for two samples 90 degrees apart the
midpoint has `|q| = 0.924`, which read as-is would be a heading 6.2 degrees off
and an angular velocity 29% short of its true magnitude. (Linear interpolation
is why the clip loader slerps and renormalises rather than lerping.)

An orientation that cannot define a rotation at all — all zeros, which is how a
never-written or dropped orientation reads, or a non-finite component — is
refused with a `ValueError` naming the helper and the value. Scaling cannot
recover a direction from it, and standing in an arbitrary rotation would feed
the network a plausible-looking wrong frame.

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

Each channel is `[num_frames, ...]`, so a cache states its frame count several
times over and `MotionPlayer` checks those statements agree before playing it.
That matters when you edit a cache by hand - trimming or concatenating the
channels and leaving `num_frames` behind is refused with both counts named,
rather than overrunning the arrays part-way through the clip (the frame index is
clamped to `num_frames`, and the tracker's future window reads ahead of the
playhead) or silently hiding the tail. Drop `num_frames` and the channels' own
row count is used:

```python
cache["dof_pos"] = cache["dof_pos"][:100]   # ... and the other five channels
del cache["num_frames"]                     # or set it to 100
player = MotionPlayer(cache)
```

### The MJCF has to be the tracker's own embodiment

`proto_mjcf_path` is not just any G1. The tracker reads a body out of the cache
by **row index**, never by name: `anchor_body_index` and `root_body_index` are
offsets into `GTP_G1_BODY_NAMES`, the 33-name list pinned from the checkpoint's
sidecar. `qpos_to_motion_data` fills those rows by name from the model you hand
it, so the model has to carry all 33 - plus a free root and the 29
`GTP_G1_JOINT_NAMES` joints, for a `qpos` width of 36.

The G1 family does not agree on that body set. The widely-shipped fingerless
models omit `head` and both `rubber_hand` placeholders and expose 30 bodies;
`g1_29dof_with_hand` and `g1_with_hands` expose 44 and a `qpos` width of 50.
Passing one of those is refused, with a message that names what is missing:

```text
ValueError: ProtoMotions G1 MJCF .../g1_29dof.xml is missing 3 of the 33 bodies
the tracker reads by row index: ['head', 'left_rubber_hand', 'right_rubber_hand'].
```

The refusal is the point. Read positionally, a 30-body model shifts every row
after the gap by one, so the tracker asks for `torso_link` at row 16 and is
handed `left_shoulder_pitch_link` - on a walking clip, an anchor orientation
some 20 degrees out, with nothing in the cache to say so. A model whose `qpos`
layout differs is refused separately and says so, so a model-side mismatch is
not read as a bad `qpos` argument.

### An MJCF that declares its own ground is used as-is

Forward kinematics needs a floor, and the ProtoMotions G1 MJCF names one from
`<contact><pair geom2="floor">`, so the bridge appends a plane geom called
`floor` when - and only when - the model does not already declare a ground.

Whether it does is decided from the geom list MuJoCo itself parses out of the
file, so every way of declaring a ground counts: a plane in any of the
`<worldbody>` sections MuJoCo merges, one nested inside a body, and one spliced
in through `<include>`. Pass any G1 MJCF that already ships a floor - the
`unitree_ros` G1 descriptions declare theirs in a second `<worldbody>`, and
menagerie-style `scene.xml` wrappers `<include>` the robot and add their own -
and it is used unchanged.

### Cache velocities are world-frame

`body_pos` and `body_rot` are world poses, and `body_vel` and `body_ang_vel`
are **world-frame** velocities derived from them - the same convention a raw
ProtoMotions motion library uses for `rigid_body_ang_vel`. That matters when
you hand-build a cache instead of bridging one: the tracker's root input is a
*local*-frame angular velocity, and `compute_root_local_ang_vel` produces it by
rotating a world-frame row into the root's frame. Storing local-frame rows here
gets them rotated a second time. The frames are not interchangeable - on a
walking G1 clip they differ by whole rad/s, the same class of silent wrong-frame
error as substituting the base for the anchor link above.

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
