# Kimodo — text-to-motion diffusion for the Unitree G1

`KimodoPolicy` wraps NVIDIA's Kimodo (`nvidia/Kimodo-G1-RP-v1`) text-conditioned
motion diffusion model. Given a natural-language prompt it samples per-frame
full-body `qpos` sequences for the Unitree G1 in a single diffusion pass, then
streams them one frame per tick as G1 joint targets.

Kimodo sits in the same seat as [`MotionBricksPolicy`](./motionbricks.md) — it
is a *kinematic motion generator* that emits motion targets, not torques — a
whole-body reference over all 29 leg + waist + arm joints. Applying that
reference under physics needs a controller that *tracks* it; see
[Tracking the reference under physics](#tracking-the-reference-under-physics).

## When to use

| | Kimodo | MotionBricks |
|---|---|---|
| Control input | free-form text prompt | style token + heading |
| Prompt vocabulary | anything English | fixed clip modes |
| Sampler | diffusion (multi-step) | autoregressive one-shot |
| Wall clock (Jetson AGX-class, 100 steps, 120 frames) | ~8 s | ~1 s |
| Best for | novel motions, prompt engineering | known styles, low latency |

## Install

```bash
pip install "strands-robots[kimodo]"
```

The extra installs the `diffusers` loader, which drives any checkpoint published
in *diffusers pipeline layout*. `trust_remote_code=True` is forwarded for a
pipeline that ships custom code, and the factory gates it behind an explicit
opt-in:

```bash
export STRANDS_TRUST_REMOTE_CODE=1
```

Weights are fetched from HuggingFace on first use under the NVIDIA Open Model
License; nothing is bundled with `strands_robots`.

!!! important "`nvidia/Kimodo-G1-RP-v1` is not a diffusers pipeline"

    NVIDIA publishes the Kimodo weights bare — `config.yaml`,
    `model.safetensors` and `stats/`, with `library_name: kimodo` on the Hub.
    There is no `model_index.json`, so `DiffusionPipeline.from_pretrained`
    cannot load it and the default `model_id` is refused at construction. To
    run the NVIDIA checkpoint, supply its sampler through `motion_agent=` — see
    [Driving the NVIDIA checkpoint](#driving-the-nvidia-checkpoint).

## Quick start

```python
import os; os.environ["MUJOCO_GL"] = "egl"  # headless GL on Jetson/Docker
from strands_robots import Robot

sim = Robot("g1", mesh=False)
sim.add_camera(name="front", position=[3.0, 0.0, 1.2], target=[0.0, 0.0, 0.8])

sim.run_policy(
    robot_name="g1",
    policy_provider="kimodo",
    policy_config={
        "diffusion_steps": 100,
        "guidance_scale": 7.5,
        "num_frames": 120,
        "device": "cuda",
        "dtype": "fp16",
    },
    instruction="a person walking forward with confident strides",
    n_steps=200,
    control_frequency=50,
    video={"path": "walk.mp4", "camera": "front", "fps": 25},
)
```

## Tracking the reference under physics

Kimodo is kinematic: it emits joint *targets* for all 29 DOFs, not torques. Run
standalone (the example above) those targets are applied directly, which is the
faithful visualisation of the generated motion.

Making the robot follow that motion under physics requires a controller that
**tracks the reference** — the 29 targets are the tracker's input. Generator and
tracker therefore run in **series over the same joints**:

```text
prompt -> Kimodo -> 29 joint targets -> reference tracker -> torques -> robot
```

That is a cascade, and `CompositePolicy` does not express it.
[`CompositePolicy`](./custom-policies.md) merges two policies over **disjoint**
joint groups (locomotion legs+waist plus manipulation arms, each joint owned by
exactly one child); handing it a whole-body generator and a whole-body controller
gives both children the same joints, so one child's output is discarded entirely.
That configuration is refused with an error naming the shadowed joints rather
than silently returning one child's commands.

[`WBCPolicy`](./wbc.md) in particular is **not** a reference tracker: its only
command input is a target base velocity (`target_velocity`, plus optional
orientation and height), it has no reference-pose input, and it drives 15 of the
same 29 joints Kimodo drives. Composing the two cannot track a Kimodo motion.

`strands_robots` does not currently ship a whole-body reference tracker. A
tracker matched to the generator's motion distribution (an RL tracker trained on
it, or a tuned PD law) is required, and is out of scope for this provider.

## Config reference

`KimodoConfig` (`strands_robots.policies.kimodo.KimodoConfig`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_id` | str | `nvidia/Kimodo-G1-RP-v1` | HF model id |
| `diffusion_steps` | int | 100 | 25–200 useful range |
| `guidance_scale` | float | 7.5 | CFG weight |
| `num_frames` | int | 120 | ≤196 (RP-v1 max) |
| `transition_frames` | int | 5 | Native frames a chained segment is eased over |
| `native_fps` | int | 30 | Sampler native rate |
| `tracker_fps` | int | 50 | SLERP upsample target |
| `device` | str \| None | auto | `"cuda"` / `"cpu"` |
| `dtype` | str | `"fp16"` | `"fp16"` / `"bf16"` / `"fp32"` |
| `seed` | int \| None | None | Reproducible sampling |

Every field above is also an explicit keyword argument of `KimodoPolicy`, so it
can be set three interchangeable ways:

```python
from strands_robots import create_policy
from strands_robots.policies.kimodo import KimodoConfig, KimodoPolicy

create_policy("kimodo", diffusion_steps=25)          # flat, through the factory
KimodoPolicy(config=KimodoConfig(diffusion_steps=25))  # a config object
KimodoPolicy(config={"diffusion_steps": 25})           # a plain dict
```

Precedence is per-field override > `config` field > the default in the table. A
merged value is re-validated by `KimodoConfig`, so `diffusion_steps=0` is
refused whichever way it arrives. There is no `**kwargs`: a misspelled knob
raises `TypeError` at construction instead of being silently ignored.

## When the sampler runs again

One `sample()` call produces a motion buffer that `get_actions` then drains one
frame per control tick, holding the last frame once the buffer is exhausted. The
buffer is identified by the four inputs that determine it - the prompt plus
`diffusion_steps`, `guidance_scale` and `seed` - so the sampler runs again as
soon as any of them differs from the values that produced the buffer in hand,
and otherwise the buffered frames are reused:

```python
await policy.get_actions({}, "walking forward")                     # samples
await policy.get_actions({}, "walking forward")                     # drains
await policy.get_actions({}, "waving")                              # samples
await policy.get_actions({}, "waving", diffusion_steps=25)          # samples
policy.reset()                                                      # rewinds
policy.reset(seed=7); await policy.get_actions({}, "waving")        # samples
```

This is what makes a multi-episode `eval_policy` meaningful for a stochastic
policy. `PolicyRunner.evaluate` derives a distinct seed per episode and forwards
it to `policy.reset(seed=...)`, so each episode samples its own motion while the
whole run stays reproducible: re-running at the same master `seed=` replays the
same per-episode motions. Repeating a seed replays the buffered motion rather
than re-running the sampler for identical frames, and `reset()` without a seed
only rewinds - neither pays for a diffusion run.

## Chaining prompts into a long-horizon sequence

Because a changed prompt samples the next segment and the stream simply
continues, a long-horizon episode is a rollout that changes the instruction as
it goes — no stitching layer required. Anything that can vary the instruction
per tick will do; a `policy_object` driven directly is the smallest version:

```python
import asyncio

from strands_robots import Robot
from strands_robots.policies.kimodo import KIMODO_G1_JOINTS, KimodoPolicy

CHAIN = [
    ("a person walking forward with confident strides", 90),
    ("a person turning to the left", 60),
    ("a person waving with the right hand", 60),
    ("a person crouching down to pick an object off the floor", 90),
    ("a person walking forward with confident strides", 90),
]

sim = Robot("g1", mesh=False)
policy = KimodoPolicy()
policy.set_robot_state_keys(list(KIMODO_G1_JOINTS))

for instruction, ticks in CHAIN:
    for _ in range(ticks):
        action = asyncio.run(policy.get_actions({}, instruction))[0]
        sim.set_joint_positions(action, robot_name="g1")
```

Each segment is sampled once, on the tick its instruction first appears. Kimodo
samples every motion from its own canonical start pose, which has no relation to
wherever the previous segment left the robot, so a new segment is eased off the
pose last commanded across `transition_frames` native frames. Without that the
seam commands every joint to step at once — measured across the 600 ordered
pairs of a 25-motion corpus, the median seam moved a joint 1.6 rad in a single
tick and 84% of transitions exceeded the largest step the motions themselves
ever take. A reference like that is not one a tracker can follow, and it is
reported as a successful rollout.

`transition_frames` defaults to 5, the same length Kimodo's own sampler uses for
`num_transition_frames` when it generates a multi-prompt sequence. Raise it for
wider pose gaps (opposing poses eased over more frames), lower it toward 1 to
keep segment boundaries crisp. It is bounded below at 1, matching the domain the
sampler enforces on its own transition length.

Note the difference in kind from the sampler's native multi-prompt path: Kimodo
conditions the *diffusion* of the next segment on the previous segment's last
frames, so the generated motion itself leads into the transition. The agent
protocol here takes only a prompt and sampling knobs, with no continuation
state, so easing shapes the emitted stream rather than the sample. It removes
the discontinuity; it does not re-plan the motion around it.

An episode boundary is not a seam. `reset()` forgets the last commanded pose, so
the next episode opens on its motion's own start pose rather than being eased
onto wherever the previous episode finished.

## When the checkpoint is not a Kimodo checkpoint

`model_id` is accepted verbatim so an alternate revision can be pinned. Two
distinct refusals guard that freedom.

**At load time**, a target carrying no `model_index.json` is not a diffusers
pipeline at all, so no amount of sampling will help. Rather than surface a bare
404 for a file that will never exist, the loader names the layout mismatch and
the remedy:

```text
RuntimeError: Kimodo model_id 'nvidia/Kimodo-G1-RP-v1' is not a diffusers
pipeline: it carries no model_index.json, so DiffusionPipeline.from_pretrained
cannot load it. NVIDIA's Kimodo checkpoints publish bare weights (config.yaml
plus model.safetensors) for their own runtime - the Hub declares library_name
'kimodo', not 'diffusers'. Pass motion_agent= with a sampler that loads this
checkpoint through its own runtime and returns a (num_frames, 7+29) qpos array,
or point model_id at a checkpoint published in diffusers pipeline layout.
```

A transport failure is *not* reported this way — a 401 or a 503 re-raises
untouched, so a network problem is never misread as a layout problem.

**At sample time**, a pipeline that loaded but names its output something other
than `motion` is refused with a `RuntimeError` naming the `model_id` and the
fields the output *did* carry:

```text
RuntimeError: Kimodo pipeline output for model_id 'acme/not-kimodo' carries no
'motion' field: got _SampleOutput with fields sample. Kimodo emits per-frame
qpos under 'motion' - point model_id at a Kimodo checkpoint, or pass
motion_agent= to adapt a sampler that names its output differently.
```

The remedies are the two the message names: point `model_id` at a Kimodo
checkpoint, or pass a `motion_agent=` adapter that reads the sampler's own
output field and returns the `(num_frames, 7+29)` `qpos` array this policy
expects.

## Driving the NVIDIA checkpoint

`nvidia/Kimodo-G1-RP-v1` loads through NVIDIA's own `kimodo` runtime, which is
distributed with the model rather than on PyPI. Wrap it in a `KimodoMotionAgent`
and hand the policy to `run_policy` as a built object:

```python
import numpy as np
import torch

from strands_robots.policies.kimodo import KimodoPolicy


class NativeKimodoAgent:
    """Samples through NVIDIA's kimodo runtime instead of diffusers."""

    def __init__(self, device: str = "cuda") -> None:
        from kimodo.exports.mujoco import MujocoQposConverter
        from kimodo.model.load_model import load_model

        self._model = load_model("kimodo-g1-rp", device=device)
        self._converter = MujocoQposConverter(self._model.skeleton)
        self._device = device

    def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
        if seed is not None:
            torch.manual_seed(seed)
        output = self._model(
            [prompt.strip().rstrip(".") + "."],
            [num_frames],
            num_denoising_steps=diffusion_steps,
            num_samples=1,
            return_numpy=True,
        )
        qpos = np.asarray(self._converter.dict_to_qpos(output, self._device))
        return qpos[0].astype(np.float32) if qpos.ndim == 3 else qpos.astype(np.float32)


sim.run_policy(
    robot_name="g1",
    policy_object=KimodoPolicy(motion_agent=NativeKimodoAgent()),
    instruction="a person walking forward with confident strides",
    n_steps=200,
    control_frequency=50,
)
```

The runtime emits a dict of rotation matrices and root positions, so the
`MujocoQposConverter` step is what produces the `(num_frames, 7+29)` qpos array
the agent protocol expects. `guidance_scale` has no counterpart in that runtime
(its classifier-free-guidance knob is a per-stage `cfg_weight` list) and is
ignored by this adapter.

`seed` is applied with `torch.manual_seed` because the runtime draws its initial
noise from the global torch generator and accepts no generator or seed argument
of its own. Seeding is what makes the agent reproducible, and it is what an
adapter is most likely to leave out: an agent that accepts `seed` and ignores it
still satisfies the protocol, so nothing raises, but every request samples fresh
noise. That silently defeats the per-episode seed, since `eval_policy` derives
one seed per episode and hands it to `reset()`, and the policy re-samples
whenever a sampler input changes. Every episode would then get an independent
motion no seed can reproduce.
## Driving the real robot

Kimodo names its joint targets the way the URDF does (`left_hip_pitch_joint`);
lerobot's `UnitreeG1` driver names its action keys after its own joint enum
(`kLeftHipPitch.q`). The two vocabularies name the same 29 joints, so the
hardware path is a key rename applied between the policy and the driver:

```python
from strands_robots.policies.kimodo.hardware import build_lerobot_g1_action_dict

for policy_action in await policy.get_actions(observation, instruction):
    robot.send_action(build_lerobot_g1_action_dict(policy_action))
```

`get_joint_map()` returns the table itself (`{"left_hip_pitch_joint":
"kLeftHipPitch.q", ...}`) if you would rather rename in your own loop. Both are
lerobot-only helpers: `pip install "strands-robots[lerobot]"`. Commanding the
physical robot additionally needs Unitree's `unitree_sdk2` runtime, which
lerobot documents separately for its `unitree_g1` robot.

The table pairs joints by name, never by position in the driver enum. That
matters because the driver applies only the action keys it recognises and leaves
every other motor on its previous command, so a key paired with the wrong joint
— or spelled in a way the driver does not know — raises nothing at all and the
robot simply moves wrong. Pairing by name means a driver-side reorder cannot
move a target, and a driver-side rename or DOF change is refused with the
unmatched joints named on both sides instead of being taken on trust:

```text
RuntimeError: Unitree G1 joint sets disagree between the policy and lerobot's
driver. Joints the policy commands that the driver does not name:
['waist_yaw_joint']. Joints the driver names that the policy does not command:
['kTorsoYaw.q']. ...
```

The rename is one-way. The driver's `get_observation()` already reports
`<motor>.q` keys, so the read path needs no inverse table.

## Unit testing without weights

Inject a `KimodoMotionAgent` stub — no torch/diffusers/CUDA needed. See
`tests/policies/kimodo/test_kimodo_policy.py` for the pattern.

## References

* Kimodo: <https://huggingface.co/nvidia/Kimodo-G1-RP-v1>
* Sibling policy: [`motionbricks`](./motionbricks.md)
