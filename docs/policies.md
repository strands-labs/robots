---
description: A policy is the brain. Hand Policy() a string or a function and it figures out the rest.
---

# Policies

{{drawing:v03_one_loop}}

Every loop is the same three beats: **look** (`observe`), **think** (`policy.act`), **move** (`act`).
A policy is whatever does the thinking. You pick one by how you write it:

| you write | you get |
|---|---|
| `lambda obs: {...}` | your own function, called every step |
| `"microduck?skill=velstand&vx=0.3"`, `"holosoma"` | a ready-made ONNX skill |
| `"lerobot/smolvla_base"` | a LeRobot model, running in your process |
| `"zenoh://gpu-box:7447/smolvla"` | a LeRobot model running on another machine |

Get it wrong and it tells you every form it understands, before anything loads.

Your own function returns a `{joint: value}` mapping to move, or `None` to **hold**: the body keeps its last
target that tick, counted in `RunResult.holds`, never a missed deadline — how a run waits on a slow policy
or network without booking a miss.

## The duck has skills

The microduck ships with ten skills from Pollen Robotics. Each is one string:

<div class="sr-grid2" markdown>
<figure><img class="sr-media" src="../assets/media/skill_walk.gif" alt="walk"><figcaption><code>skill=velstand&vx=0.3</code> walk</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/skill_turn.gif" alt="turn"><figcaption><code>skill=velstand&wz=1.2</code> turn on the spot</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/skill_roll.gif" alt="roll"><figcaption><code>skill=roulade</code> forward roll</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/skill_pick.gif" alt="pick"><figcaption><code>skill=alpha_ground_pick</code> reach for the ground</figcaption></figure>
</div>

Also in the box: `alpha_walking`, `alpha_stand`, `alpha_sitstand`, `ball_kick_left`, `ball_kick_right`, `roller`,
`roller_crouch`. Skills are small and sharp. Chain them and you get behaviour:

## Skills + an agent = a robot that listens

<!-- test: skip, needs AWS credentials for a Bedrock agent turn -->
```python
from strands import Agent
from strands_robots import Robot

duck = Robot("microduck")
Agent(tools=[duck.tool], system_prompt="Skills: velstand (vx, vy, wz), roulade, alpha_ground_pick.")(
    "walk forward a bit, turn left, do a roll, then chill")
```

<img class="sr-media" src="../assets/media/agent_duck.gif" alt="the agent drives the duck">

The model never touches a joint. It picks a skill, the skill moves the motors, and the loop keeps everything safe.
This split is the whole trick behind agentic robots: the language model plans, the learned policy acts.

## LeRobot, without the glue

Any LeRobot checkpoint works, in your process or on a GPU across the room:

```bash
# on the GPU box
lerobot-policy-server --config_path=smolvla.yaml
```

```python
# anywhere
arm.run(Policy("zenoh://gpu-box:7447/smolvla"), instruction="pick up the cube", seconds=10)
```

A server across the internet is the same line with the link it listens on: `Policy("zenoh://tls/host:7447/deployment")`
(the locator a lease hands you, `tls/host:7447`, written as given). Off the LAN the client waits up to 30 s for the first
answer, sends JPEG frames and speaks to the router as a client; name any of those yourself to override.
A deployment that knows what it was trained to do says so — `Policy("zenoh://…", tasks=[...])`, or the server's own
descriptor — and `run(instruction=)` with any other string is refused naming the trained ones, before a frame goes out;
`policy.tasks` lists them.

We don't wrap LeRobot, we speak it: LeRobot loads the model and runs its own client. strands-robots just lines up
joint names, cameras and units, and refuses (by name) anything that doesn't match, before the robot moves.
Need the server config? `server_config(robot.description, "lerobot/smolvla_base")` writes it for you.
Units are checked, not assumed: a checkpoint's units default to the robot's LeRobot convention (the so101 is
range mode), and the checkpoint's own saved min/max must fit them. An SO-101 ACT trained in degrees (a
`wrist_roll` reaching -155) read as range mode is refused naming the joint and the fix,
`Policy(repo, norm={"*": "degrees", "gripper": "range_0_100"})`.

We tried every LeRobot family on an L40S: **13 drive our robots** (ACT, SmolVLA, π0, π0.5, GR00T, XVLA, EVO1, LaWAM,
VLA-JEPA, FineART, MolmoAct2, DM05, and SmolVLA in RTC mode). The rest are turned away by LeRobot itself, politely
and before anything moves. The same model gives the exact same actions locally and served.

## A leader arm is a policy

{{drawing:d18_leader_source}}

Teleoperation is just a run whose policy is the leader. Hand a LeRobot `Teleoperator` (or a `Robot` you are
mirroring) straight to `run()`:

```python
# a LeRobot SO-101 leader on its own serial port drives the follower for a minute
arm.run(leader, hz=30, seconds=60)
```

`as_policy` wraps the leader in a `SourcePolicy`, so no glue loop and no unit guesswork: the leader's bus already declares each motor's norm mode, so the follower converts
every reading through its own calibration, exactly as LeRobot's `teleoperate` does. A degrees leader drives a
range-mode follower to within one encoder count. A source that declares no units is refused, naming the fix; a raw
`{"shoulder_pan.pos": ...}` dict handed to `act()` is refused too, because `act()` takes a `<joint>` target in
radians, not a LeRobot name in the policy's units.
