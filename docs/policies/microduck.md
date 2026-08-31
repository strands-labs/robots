# Microduck — locomotion policies for the Pollen 14-DOF biped

`MicroduckPolicy` wraps one of Pollen Robotics' shipped **Microduck** ONNX
policies (`alpha_walking`, `alpha_stand`, `alpha_sitstand`, `roulade`,
`ball_kick_left`/`ball_kick_right`, `roller`/`roller_crouch`,
`alpha_ground_pick`) and drives the open 14-DOF biped through the standard
`Robot(...).run_policy` seam — in MuJoCo or on hardware.

Each export is an actor with its input **normaliser fused into the graph**, so
the provider feeds the observation **raw** and never re-normalises. The policy
also **self-configures from the ONNX metadata**: `joint_names`,
`default_joint_pos`, `action_scale` and `command_names` are read from the file's
`custom_metadata_map` on first inference, so pointing it at a different weight
file reconfigures it. Explicit constructor arguments always win.

## Walking in MuJoCo

![Microduck walking in MuJoCo](../assets/microduck/microduck_walk.gif){ width=480 }

_`alpha_walking.onnx` driven forward at `vx=0.3 m/s`, filmed with a
body-tracking chase camera. Reproduce with
[`examples/microduck/render_video.py`](https://github.com/strands-labs/robots/blob/main/examples/microduck/render_video.py):_

```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib  # macOS: Homebrew ffmpeg
python examples/microduck/render_video.py \
    --onnx ../microduck/policies/alpha_walking.onnx \
    --vx 0.3 --duration 8 --out walk_forward.mp4 \
    --gif docs/assets/microduck/microduck_walk.gif
```

`render_video.py` steps the sim manually at the control frequency and captures
each frame with a tracking camera locked to the pelvis, so the duck stays
centered as it walks. `--vx`/`--vy`/`--vyaw` set the twist command, and any
weight that runs on the default scene (`alpha_stand`, `roulade`,
`alpha_sitstand`, `alpha_ground_pick`) drops straight in. The script builds
`Robot("microduck")`, so the four skills that need a different scene -
`roller`, `roller_crouch`, `ball_kick_left`, `ball_kick_right` - need the
scene named for them in [Skill scenes](#skill-scenes) below.


## Install

```bash
pip install "strands-robots[microduck]"
```

That pulls `onnxruntime` (runs the graph). Weights are not bundled — they ship
in Pollen's `microduck` repository under `policies/*.onnx`. A MuJoCo rollout
additionally needs `strands-robots[sim-mujoco]`.

## Walk in simulation

```python
from strands_robots import Robot
from strands_robots.policies.microduck import MicroduckPolicy

sim = Robot("microduck")
sim.reset()

policy = MicroduckPolicy(onnx_path="alpha_walking.onnx")
sim.run_policy(
    policy_object=policy,
    control_frequency=50,
    duration=8.0,
    policy_kwargs={"target_velocity": [0.3, 0.0, 0.0]},  # forward twist
)
```

See [`examples/microduck/microduck_walk_sim.py`](https://github.com/strands-labs/robots/blob/main/examples/microduck/microduck_walk_sim.py)
for the runnable script.

## Skill scenes

A shipped weight and the scene it was trained in are one pair. `Robot("microduck")`
resolves the entry's declared asset - flat ground, no props - which is what the
walking, standing, sitstand, roulade and ground-pick skills need. Four skills
need something the default scene does not contain, and Pollen ships the scene
for each one in the same asset directory:

| skill | scene | what the scene adds |
| --- | --- | --- |
| `alpha_walking`, `alpha_stand`, `alpha_sitstand`, `roulade`, `alpha_ground_pick` | `scene.xml` (the entry's declared asset) | nothing - flat ground |
| `roller`, `roller_crouch` | `scene_rollers.xml` | four passive ankle wheels, so the feet can roll |
| `ball_kick_left`, `ball_kick_right` | `scene_ball.xml` | a 70 mm, 15 g ball placed in front of the duck |

Reach a non-default scene by path. The registry entry names the fourteen-hinge
model deliberately - it is the layout the catalog documents - so the variants are
loaded as an explicit asset rather than resolved by name:

```python
from pathlib import Path

from strands_robots import Robot
from strands_robots.policies.microduck import MicroduckPolicy
from strands_robots.utils import get_search_paths

scene = next(
    candidate
    for root in get_search_paths()
    if (candidate := Path(root) / "microduck" / "scene_rollers.xml").exists()
)
sim = Robot("microduck", urdf_path=str(scene))
sim.reset()
sim.run_policy(policy_object=MicroduckPolicy(onnx_path="roller.onnx"), duration=8.0)
```

Running a skill on the default scene is not an error and reports success: a
roller policy writes the same fourteen control targets, and with no wheels under
the feet the duck simply stands; a ball-kick policy swings at a ball that is not
there. Nothing refuses it, so the scene is the caller's to choose.

### The ball scene carries the ball, not the kick geometry

`scene_ball.xml` places the prop, and where it sits is not where the kick policies
were trained to find it. The scene declares the ball 0.3 m straight ahead
(`ball.xml`: `pos="0.3 0 0.035"`). Pollen's training reset placed it 0.09 m ahead and
0.042 m to the side of the kicking foot, in the robot's yaw frame - 3.3x closer, and
offset laterally to the foot that swings.

Loading the scene is therefore necessary and not sufficient. Driven from the shipped
position, `ball_kick_left` completes and reports success while no robot body ever
touches the ball: across a four-second rollout the closest any robot geom comes to the
ball centre is 0.109 m, against a 0.035 m radius. The ball still travels forward on its
own - its geom sets a deliberately low rolling resistance - which is why the miss reads
as a weak kick rather than as a miss.

A caller who wants the trained geometry teleports the ball before the rollout, which is
what Pollen's runtime does at the moment a kick is triggered: write the ball free
joint's `qpos` to that offset rotated into the trunk's yaw frame, zero the joint's
`qvel`, and step. The file names the joint `ball_free`; `add_robot(name=...)` prefixes
every joint with the name the caller passed, so resolve the name rather than assuming
either spelling.

### Reading joint positions on the rollers scene

`scene_ball.xml` appends the ball's free joint after the robot's, so the robot's
`qpos` layout is byte-for-byte the default one. `scene_rollers.xml` does not:
it inserts two passive wheel joints after `left_ankle` and two more after
`right_ankle`, which moves nine of the fourteen actuated joints to a different
`qpos` index. A consumer that reads a flat `qpos[7:21]` slice therefore reads
different joints there - the two left wheels arrive where `neck_pitch` and
`head_pitch` sit on the default scene.

The actuator order is identical across all three scenes, so a policy writing
`ctrl` is unaffected, and `MicroduckPolicy` reads its observation by joint name
rather than by slice, so the provider is immune either way. Only a raw position
read has to care.

### The stance every weight was trained in

A weight and its scene are one pair; so are a weight and the stance it starts
from. Every shipped Pollen weight bakes that stance into its ONNX metadata as
`default_joint_pos`, and all nine declare the same fourteen values.
`MicroduckPolicy` reads it into `default_pose` and decodes every action relative
to it - `motor_target = default_pose + raw_action * action_scale` - so the stance
is not advice, it is the origin the network's output is measured from. The same
values ship as `strands_robots.policies.microduck.MICRODUCK_DEFAULT_POSE`, the
fallback the provider uses when a session carrying no metadata is injected.

The asset ships that stance too, as the `STAND` keyframe in `scene.xml` and
`scene_rollers.xml`. Name it at spawn and the robot starts there:

```python
sim = Robot("microduck", urdf_path=str(scene), keyframe="STAND")
```

A keyframe spawn is sticky across resets: `sim.reset()` restores the pose and the
actuator command that holds it, so every episode of a `run_policy` + `reset` loop
begins from the same stance.

Spawning without `keyframe` is not an error and reports success. The robot starts
at the zero configuration, 0.458 rad from the trained stance at the widest joint
- legs straight rather than crouched - while the policy still decodes relative to
the stance it expects. Nothing refuses it, so the first inference of the rollout
reads a pose no shipped weight was trained on.

Two details a caller meets:

- The keyframe is named `STAND`. The asset's own comment calls the current values
  "STAND2", because they supersede an earlier `STAND` that is commented out
  beside them; the live keyframe kept the name. `keyframe="STAND2"` is refused,
  and the refusal names the keyframes the model does declare.
- `scene_ball.xml` declares no keyframe at all, so the route above is unavailable
  on the one scene a ball kick needs. Seat the stance yourself there, reading it
  from `MICRODUCK_DEFAULT_POSE` rather than copying the numbers - the asset has
  already revised this pose once.

## The observation contract

The vector is a fixed float32 concatenation (measured off Pollen's reference
`infer_policy.py` and each ONNX's `observation_names` metadata):

| block | width | source |
| --- | --- | --- |
| `base_ang_vel` | 3 | IMU angular velocity |
| `projected_gravity` | 3 | world `-Z` rotated into the base frame from `base_quat` |
| `joint_pos` | 14 | current joint position − `DEFAULT_POSE`, contract order |
| `joint_vel` | 14 | joint velocity, contract order |
| `last_action` | 14 | the **previous raw** ONNX output (not the motor target) |
| `command` | C | unified command (`twist(3) + head_pose(4) + body_pose(6)`) |

Total width is `48 + C`: **61** for the shipped alpha policies (C = 13) and 51
for legacy twist-only policies (C = 3). The width is read from `command_names`,
never hardcoded, and unused command slots stay present and zero (the
dead-weight rule) so one observation layout serves every skill. Actions decode
as `motor_target = DEFAULT_POSE + action * action_scale`.

`action_scale` is the only path from the network's output to the joint targets,
so it must be a positive finite number. A scale of `0` would make every target
exactly `DEFAULT_POSE` — the network's decision discarded and the biped holding
its nominal stance while the rollout reports success — and a non-finite one
would make all fourteen targets `nan`. Both routes to the decode are held to
that domain: an explicit `action_scale=` and the value read from the ONNX
`action_scale` metadata.

## Commanding motion

The command vector defaults to all-zero (stand in place). Steer with the
well-known `target_velocity` kwarg (writes the twist slots) or replace it
wholesale with `command=`:

```python
await policy.get_actions(obs, "", target_velocity=[0.3, 0.0, 0.2])  # vx, vy, ω
```

`target_velocity` takes three components (`[vx, vy, omega]`) or two
(`[vx, vy]`, which leaves `omega` at its current value - the command vector
persists across ticks). Any other component count is refused rather than
truncated or partially written, and a `nan`/`inf` component is refused before it
reaches the command; `command=` must be `command_names`-wide and finite for the
same reasons.

What the twist slots MEAN, however, is a property of the loaded weights rather
than of this provider. Pollen's locomotion exports (`alpha_walking`,
`alpha_stand`, the `roller*` pair) read them as a velocity, which is exactly what
`target_velocity` writes. Other exports in the family read the same three slots
differently - `alpha_ground_pick`, for instance, reads them as a progress
encoding through a one-shot motion rather than as a velocity - and for those a
caller supplies the slots wholesale through `command=` and advances them itself.
`target_velocity` is a locomotion kwarg, not a universal one, and the ONNX
metadata does not distinguish the two: several exports declare the same
`command_names` (`twist`) while reading it under different conventions.

## Hot-swapping skills

`MicroduckPolicyBundle` holds several `MicroduckPolicy` instances warm and
delegates each tick to the active one, so a controller can switch skill
mid-rollout without rebuilding sessions:

```python
from strands_robots.policies.microduck import MicroduckPolicy, MicroduckPolicyBundle

bundle = MicroduckPolicyBundle(
    {
        "walk": MicroduckPolicy(onnx_path="alpha_walking.onnx"),
        "stand": MicroduckPolicy(onnx_path="alpha_stand.onnx"),
    },
    active="stand",
    switch_on_velocity=0.1,  # auto walk<->stand by |twist|
)
```

Select explicitly with `get_actions(..., select="walk")` or `bundle.switch(...)`.

An explicit selection is not undone by the gate: it arbitrates *between*
`move_key` and `idle_key`, and leaves any other skill alone until a gate key is
selected again. That matters most for `alpha_sitstand`, whose `twist[0]` is a
posture flag (`1` sit, `0` stand) rather than a velocity — both of its commands
have a magnitude the gate would otherwise read as a walk or an idle request, so
neither would have reached the skill that was asked for.

`switch_on_velocity` must be a positive finite number. The gate compares a
magnitude, so a threshold of `0` or below could never select the idle skill and
a non-finite one could never select the move skill. Omit it (the default) to
leave the gate off and switch only explicitly.

The velocity it is compared against is held to the same standard as the threshold
it is compared with. With the gate on, a `target_velocity` this tick cannot honor
is refused before the gate arbitrates, naming the bundle and the parameter — so a
refused tick leaves the active skill exactly as it was, rather than selecting the
idle skill from a `nan` magnitude and keeping that selection on every tick after.
The accepted values are the ones the active skill accepts: finite numeric
components, and the same two component counts documented above. An absent
`target_velocity` is still simply "no goal this tick" and leaves the selection
alone.

`move_key` and `idle_key` name the two skills that gate selects between, and each
must be one of the bundle's own keys whenever the gate is enabled. The gate reads
both every tick, so a key naming no held skill leaves it inert rather than
failing — a bundle keyed by the weight names it loads (`alpha_walking`,
`alpha_stand`) and left on the defaults `"walk"`/`"stand"` constructs, reports a
validated threshold, and then never switches. Key by the names the gate expects,
or pass `move_key=`/`idle_key=` to match the keys you used. With the gate off
neither is read, and neither is checked.

## Byte-compatibility

`MicroduckPolicy.infer_raw(obs_vector)` runs the graph on a raw observation with
no normalisation — exactly as Pollen's reference deployment does. The provider's
test suite pins that an identical 61-D observation yields an action byte-identical
(0.0 max abs delta) to a bare `onnxruntime` session, and that a real MuJoCo
rollout moves the joints.
