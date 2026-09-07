---
description: NVIDIA Cosmos 3 omnimodal VLA - WebSocket service, droid/umi/av/bridge/openarm embodiments, MuJoCo rollout.
---

# Cosmos 3

```bash
uv pip install "strands-robots[cosmos3-service]"   # adds msgpack + websockets; no openpi-client needed
```

```python
from strands_robots.policies import create_policy

policy = create_policy("cosmos3", embodiment="droid", port=8000)
# or: create_policy("cosmos3://localhost:8000")
```

## Start the server

```bash
python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000
# embodiment is selected client-side via create_policy(..., embodiment="droid")
```

## Parameters

```python
Cosmos3Policy(
    embodiment="droid",          # droid | umi | av | bridge | openarm
    host="localhost",         # bare hostname or IP literal; IPv6 bracketed "[::1]"
    port=8000,                   # int in [1, 65535]
    action_space=None,
    observation_mapping=None,
    action_mapping=None,
    robot=None,                  # "franka" or "panda" for built-in DROID→sim mapping
    prompt="",
    api_key=None,
    client=None,
    transport="raw",
    backend="service",          # "service" (default) | "diffusers" (in-process)
    mode="policy",              # "policy" | "forward_dynamics" | "inverse_dynamics" (diffusers only)
    model=None,                 # HF repo id / path for the diffusers backend
)
```

`host` and `port` are the two halves of the one address this client dials
(`ws://<host>:<port>`), and both are refused before the endpoint is built rather
than surfacing later as an unreachable server. `host` must be a bare hostname or
IP literal - no `/`, `:`, scheme or credentials, IPv6 bracketed as `"[::1]"` -
because the URI parse gives a delimiter to a later component and takes the port
with it: `host="localhost/foo"` reads as port **80**, so the configured `8000`
lands in the path. Both are read only when this constructor builds the client; an
injected `client=` owns its own address. `host="0.0.0.0"` reaches a server bound
on every interface.

## Embodiments

| Embodiment | Robot hardware | Strands sim asset |
|------------|----------------|-------------------|
| `droid` | Franka / DROID dataset | `"panda"` or `"franka"` |
| `umi` | UMI gripper | - |
| `av` | Autonomous vehicle cameras | - |
| `bridge` | Bridge dataset robots | - |
| `openarm` | Enactic OpenArm (7-DOF + gripper) | `"openarm"` |

### Action spaces

An embodiment serves its action under one or more `action_space` names, and each
one names its own columns:

| `action_space` | Columns | Notes |
|----------------|---------|-------|
| `midtrain` | The model's unified action: `tx,ty,tz` + the 6D rotation `r0..r5` + `grasp` (omitted for `av`, which has no gripper) | Served through un-converted, so the columns are the same as `raw_action_layout` |
| `joint_pos` | `joint_0..joint_6` + `gripper` (DROID only) | The one space the RoboLab server post-processes, converting the effector pose into joint targets |

`action_mapping` renames a column to one of your robot's actuator names, so its
keys must be columns of the active space. Mapping the gripper is therefore
`{"grasp": ...}` under `midtrain` and `{"gripper": ...}` under `joint_pos`; a key
that names no column is refused, listing the ones that are valid.

It has to be a rename, so two columns may not arrive at one actuator name. A
step dict holds one entry per column, so a merge would collapse two columns into
one entry and drop a command - the actuator that lost would hold position while
the model asked it to move. Both spellings are refused at construction, naming
the columns and the target they collide on: two entries sharing a target, and a
single entry aimed at another column's own name (`{"joint_0": "joint_1"}` on
`joint_pos` names only real columns and still costs a joint). Renaming *every*
column is a bijection and is accepted.

`joint_pos` needs seven joint values plus a gripper, and it reads them in the
order you declare with `set_robot_state_keys()`. Declaring that order is what
every example here does, and it is what binds the request to the joints you mean.

Without it the order is inferred from the observation's own scalar keys, and that
inferred ordering is **position-only**: a `<joint>.vel` entry is dropped when the
observation also carries its `<joint>` position companion. Every sim backend
emits a velocity sibling beside each joint position, so taking the keys in
observation order would otherwise put a velocity in every other slot of the
seven and truncate the trailing joints away. A `.vel` key with no position
companion is kept, because some embodiments legitimately declare velocity state
(LeKiwi's body-frame base velocities `x.vel` / `y.vel` / `theta.vel`). Explicit
`robot_state_keys` are never filtered - naming `elbow.vel` there states the
model's input. This is the same rule the LeRobot provider applies to its own
inferred ordering, so one observation reads as one state vector whichever
provider consumes it.

## Backends

Cosmos3Policy can run Cosmos 3 two ways. The default is unchanged.

| backend | how it runs | install | extra outputs |
|---------|-------------|---------|---------------|
| `service` (default) | WebSocket to the Cosmos Framework RoboLab policy server (holds the GPU out-of-process) | `strands-robots[cosmos3-service]` (msgpack + websockets, numpy-agnostic) | none (server video discarded) |
| `diffusers` | in-process via native `diffusers` (`Cosmos3OmniPipeline`) | `strands-robots[cosmos3-diffusers]` (floors diffusers 0.39, the first release shipping the pipeline) | world video + sound on `last_rollout` |

```bash
# in-process backend (heavy GPU stack: diffusers + torch)
uv pip install "strands-robots[cosmos3-diffusers]"
```

`Cosmos3OmniPipeline` and `CosmosActionCondition` first ship in diffusers 0.39.0,
which the extra floors. A checkpoint newer than that floor can need a newer
diffusers still - `nvidia/Cosmos3-Edge` is built against 0.40.0.dev0, which at the
time of writing ships only from source:

```bash
uv pip install 'diffusers @ git+https://github.com/huggingface/diffusers'
```

Loading a checkpoint the installed diffusers cannot build is refused, naming the
tensors it could not fill - `from_pretrained` itself only warns and leaves them
uninitialized, so without that check the pipeline would run on random weights.

The `cosmos3-diffusers` extra is native `diffusers` + `torch` + `transformers`
(no extra wrapper package). It composes with `numpy>=2` (and therefore with
`lerobot` dataset recording in the same env), so it is co-installable with
`cosmos3-service`.

> **Action layout note.** The `diffusers` backend returns the model's **raw
> unified action** (DROID = 9D end-effector pose `tx,ty,tz,r0..r5` + 1D `grasp`
> = 10D), named by the embodiment `raw_action_layout`. This is the pipeline's
> native output, *before* the RoboLab server's `joint_pos` (8D) conversion. Use
> `backend="service"` when you need joint-position commands.

> **Safety checker / `cosmos_guardrail`.** `Cosmos3OmniPipeline` builds a
> `CosmosSafetyChecker` at load time, which requires the heavy optional
> `cosmos_guardrail` package and otherwise raises `ImportError: cosmos_guardrail
> is not installed`. The diffusers backend disables it by default
> (`enable_safety_checker=False`, passed through to `from_pretrained`) so the
> pipeline loads without that extra. To re-enable it, install `cosmos_guardrail`
> and build the backend with `enable_safety_checker=True`, then hand that backend
> to the policy - the flag is a `Cosmos3DiffusersBackend` parameter, not a
> `Cosmos3Policy` one:
>
> ```python
> from strands_robots.policies.cosmos3.embodiments import get_embodiment
> from strands_robots.policies.cosmos3.policy import Cosmos3Policy
> from strands_robots.policies.cosmos3.policy_diffusers import Cosmos3DiffusersBackend
>
> backend = Cosmos3DiffusersBackend(
>     embodiment=get_embodiment("droid"),
>     model="nvidia/Cosmos3-Nano",
>     enable_safety_checker=True,   # needs cosmos_guardrail installed
> )
> policy = Cosmos3Policy(embodiment="droid", backend="diffusers", diffusers_backend=backend)
> ```
>
> `Cosmos3Policy` forwards only `embodiment`, `model` and `mode` to the backend, so
> the same route is how you reach its other load and sampling knobs
> (`resolution_tier`, `view_point`, `device`, `dtype`, `num_inference_steps`,
> `guidance_scale`, `enable_sound`). Note Cosmos runs in `bfloat16`, so the backend
> up-casts the half-precision action tensor to `float32` before returning the chunk.

### `backend="diffusers"` — world video alongside the action chunk

One in-process forward pass returns the predicted world video, optional sound,
**and** the robot action chunk. The action chunk is returned through the normal
`get_actions` -> `list[dict]` contract; the world video/sound are surfaced on
`policy.last_rollout` (the Policy ABC return type is unchanged).

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "cosmos3",
    embodiment="droid",
    backend="diffusers",
    model="nvidia/Cosmos3-Nano",  # HF repo id or local path
)
policy.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])

steps = policy.get_actions_sync(observation, "pick up the red cube")
# steps == [{"tx": .., "ty": .., ..., "r5": .., "grasp": ..}, ...]  (raw unified
# action, one dict per timestep)

# the predicted world video Cosmos rolled out for that action chunk:
print(policy.last_rollout["video"])   # path to an .mp4 / .gif
print(policy.last_rollout["sound"])   # path to a .wav, or None
```

### Action modes (diffusers only)

The diffusers backend exposes Cosmos 3's full physics loop via the `mode` kwarg
(`CosmosActionCondition.mode`). These do **not** exist in service mode — passing
a non-`policy` mode under `backend="service"` raises a clear error.

| `mode` | conditioning | predicts | `get_actions` returns |
|--------|--------------|----------|------------------------|
| `policy` (default) | first frame + task prompt | future video **+ actions** | action chunk (`list[dict]`) |
| `forward_dynamics` | first frame + given `raw_actions` | future video | `[]` (world video on `last_rollout`) |
| `inverse_dynamics` | an observed video | the actions between frames | action chunk (`list[dict]`) |

All three modes are verified live on real `nvidia/Cosmos3-Nano` weights (Thor, bf16/CUDA): `policy` → 32-step action chunk + world video; `forward_dynamics` → world video only (`get_actions` returns `[]`); `inverse_dynamics` → 32-step action chunk recovered from an observed video. See `docs/assets/cosmos3/live_modes_metrics.json`.

```python
# forward dynamics: "what world results if I run these actions?"
fd = create_policy("cosmos3", embodiment="droid", backend="diffusers", mode="forward_dynamics")
fd.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
fd.get_actions_sync(observation, "", raw_actions=my_action_chunk)
print(fd.last_rollout["video"])   # predicted world rollout

# inverse dynamics: "what actions produced this observed video?"
inv = create_policy("cosmos3", embodiment="droid", backend="diffusers", mode="inverse_dynamics")
inv.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
steps = inv.get_actions_sync(observation, "", video="observed.mp4")
```

### Closing the sim loop: de-normalize → IK → MuJoCo

The `diffusers` backend returns the model's **raw unified action** — for the
DROID/Franka domain that is `[tx, ty, tz, r0..r5, grasp]`, **quantile-normalized
to `[-1, 1]`** and encoding a *relative end-effector pose delta* per step, **not
joint radians**. Feeding it straight into MuJoCo joint actuators is physically
meaningless (the normalized columns land arbitrarily inside/outside real joint
limits; MuJoCo silently clamps and the arm doesn't track). Three honest
geometric steps (`cosmos3-sim` extra) turn it into joint targets a MuJoCo arm
actually follows:

1. **De-normalize** — invert the quantile transform with the embodiment's
   bundled `q01`/`q99` action stats:
   `denorm = 0.5 * (a + 1) * (q99 - q01) + q01`
   (`denormalize_quantile`; stats bundled under `policies/cosmos3/stats/`).
2. **Decode poses** — integrate the per-step `[translation(3), rot6d(6)]` deltas
   into an absolute `(T+1, 4, 4)` SE3 trajectory anchored at the robot's current
   EE pose (`decode_pose_trajectory`).
3. **Inverse kinematics** — solve each Cartesian target to joint angles with
   [`mink`](https://github.com/kevinzakka/mink) differential IK on the *same*
   `mujoco.MjModel` (a `FrameTask` on the EE body + a `PostureTask` regularizer),
   warm-starting each step (`MinkIKBridge`).

```python
import mujoco, numpy as np
from robot_descriptions import panda_mj_description
from strands_robots.policies.cosmos3 import (
    Cosmos3Policy, MinkIKBridge, decode_cosmos_chunk_to_targets,
)
from strands_robots.policies.cosmos3.embodiments import get_embodiment

policy = Cosmos3Policy(embodiment="droid", backend="diffusers", model="nvidia/Cosmos3-Nano")
policy.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
chunk_dicts = policy.get_actions_sync(observation, "pick up the red cube")
raw_chunk = policy.last_rollout["action"]          # [T, 10] raw [-1,1] action

model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
bridge = MinkIKBridge(model, ee_frame_name="hand", ee_frame_type="body")
q_init = np.zeros(model.nq); q_init[:7] = [0, -0.3, 0, -2.2, 0, 2.0, 0.79]

out = decode_cosmos_chunk_to_targets(raw_chunk, get_embodiment("droid"), bridge, q_init)
out["qpos"]            # [T, nq] joint targets to send to MuJoCo
out["gripper"]         # [T] grasp column (None for grasp-less embodiments)
out["tracking_error"]  # {"mean_mm", "max_mm"} Cartesian tracking error
```

Verified on Thor against real `nvidia/Cosmos3-Nano` weights, a reachable EE
trajectory tracks to **mean ≈ 11.5 mm / max ≈ 42.8 mm** — the bar pinned by the
`tests/policies/cosmos3/test_sim_ik.py` regression. (Errors grow only when the
de-normalized deltas are scaled past the ~0.85 m Franka reach — a workspace
concern, not an IK one.)

#### De-normalization stats are per domain

The de-normalize step needs that domain's own `q01`/`q99` quantiles. Two domains
ship them bundled; the other three registered embodiments do not:

| embodiment | domain | raw dim | bundled stats |
|---|---|---|---|
| `droid` | `droid_lerobot` | 10 | yes |
| `bridge` | `bridge_orig_lerobot` | 10 | yes |
| `umi` | `umi` | 10 | no |
| `av` | `av` | 9 | no |
| `openarm` | `openarm_lerobot` | 10 | no |

`nvidia/Cosmos3-Edge` documents its forward-dynamics example on `umi` and its
inverse-dynamics example on `av` — both without bundled quantiles — so
driving the sim bridge from Edge means supplying that domain's stats yourself:

```python
out = decode_cosmos_chunk_to_targets(
    raw_chunk, get_embodiment("umi"), bridge, q_init,
    stats={"q01": q01, "q99": q99},   # this domain's own quantiles
    stats_domain="umi",               # required: which domain they describe
)
```

`stats_domain` is required whenever `stats` is passed, and must match the
embodiment's domain. It is not bookkeeping: `umi`, `droid_lerobot`,
`bridge_orig_lerobot` and `openarm_lerobot` are all 10 columns, so the width
check cannot tell one
domain's quantiles from another's, and the two bundled domains disagree by up to
**2.77x** on the physical translation they decode from the same normalized
action. Substituting another domain's stats would rescale every commanded pose
delta with nothing reported.

> **The Cosmos "modes" are not FK/IK.** `policy` / `forward_dynamics` /
> `inverse_dynamics` are world-model *conditioning* modes (video↔action), not a
> kinematics solve. Joint-space IK is this separate geometric layer applied
> *after* Cosmos.
>
> The reverse map — joints → Cartesian pose — is **forward kinematics**, exposed
> as `MinkIKBridge.ee_pose(qpos) -> (4, 4)`. Step 2 above anchors the decoded SE3
> trajectory at the robot's current EE pose via exactly this FK call, and the IK
> solver uses it internally to score each Cartesian target.

![Cosmos 3 -> MuJoCo: Franka tracking the Cosmos action (left) beside the Cosmos predicted world (right)](../assets/cosmos3/cosmos3_mujoco_sidebyside.gif)

*Left: MuJoCo Franka driven by a **real** `nvidia/Cosmos3-Nano` action chunk through de-normalize → decode → IK. Right: the Cosmos predicted world video from the same forward pass. Runnable: `examples/vla/cosmos3_diffusers_mujoco_rollout.py --render out.mp4`.*

> Install the sim bridge: `uv pip install "strands-robots[cosmos3-sim]"`
> (pulls `mink` + `mujoco`; numpy>=2 compatible, co-installable with
> `cosmos3-diffusers` / `cosmos3-service` / `sim-mujoco` / `lerobot`).

## Rollout

```python
from strands_robots import Robot

sim = Robot("panda")
sim.run_policy(
    robot_name="panda",
    instruction="pick up the red block",
    policy_provider="cosmos3",
    policy_config={"embodiment": "droid", "robot": "panda", "port": 8000},
    duration=15.0,
    control_frequency=50.0,
)
# see examples/vla/cosmos3_sim_rollout.py
```

`robot="panda"` activates the built-in DROID-layout mapping (`joint_0..6/gripper` → `joint1..7/finger_joint1`). `requires_images=True`.

## See also

- [Policy overview](overview.md)
- [GR00T](groot.md)
- [LeRobot Local](lerobot-local.md)
- [Custom policies](custom-policies.md)
- [cuRobo](curobo.md)
- [Policy providers](../policies/overview.md)
