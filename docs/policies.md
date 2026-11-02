---
description: A policy turns an observation into an action. Policy() accepts a function, an ONNX skill name, a LeRobot checkpoint, a policy server address or a recorded episode.
---

# Policies

{{drawing:v03_one_loop}}

Every run is the same loop: `observe`, `policy.act`, `act`. The policy is the part that decides. `Policy()` selects
an implementation from what you pass it:

| Argument | Policy |
|---|---|
| `lambda obs: {...}` | your own function, called every step |
| `"microduck?skill=velstand&vx=0.3"`, `"holosoma"` | an ONNX skill preset |
| `"lerobot/smolvla_base"` | a LeRobot checkpoint, running in this process |
| `"zenoh://gpu-box:7447/smolvla"` | a LeRobot policy served by `lerobot-policy-server` on another machine |
| `"data/wave?episode=0"` | a recorded episode of a LeRobot dataset, replayed |

An argument that matches none of these raises an error that lists every accepted form, before anything is loaded.

A function returns a `{joint: value}` mapping, or `None` to **hold**: the body keeps its last target for that tick.
Holds are counted in `RunResult.holds` and are not deadline misses, so a run can wait for a slow policy or network
without recording a miss.

## ONNX skills

The microduck preset ships ten skills from Pollen Robotics. Each is selected by a query string:

<div class="sr-grid2" markdown>
<figure><img class="sr-media" src="../assets/media/skill_walk.gif" alt="walk"><figcaption><code>skill=velstand&vx=0.3</code> walk</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/skill_turn.gif" alt="turn"><figcaption><code>skill=velstand&wz=1.2</code> turn in place</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/skill_roll.gif" alt="roll"><figcaption><code>skill=roulade</code> forward roll</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/skill_pick.gif" alt="pick"><figcaption><code>skill=alpha_ground_pick</code> reach for the ground</figcaption></figure>
</div>

The other skills are `alpha_walking`, `alpha_stand`, `alpha_sitstand`, `ball_kick_left`, `ball_kick_right`, `roller`
and `roller_crouch`. `Policy("holosoma")` is a standing and walking policy for the Unitree G1.

## Skills driven by an agent

<!-- test: skip, needs AWS credentials for a Bedrock agent turn -->
```python
from strands import Agent
from strands_robots import Robot

duck = Robot("microduck")
Agent(tools=[duck.tool], system_prompt="Skills: velstand (vx, vy, wz), roulade, alpha_ground_pick.")(
    "walk forward a bit, turn left, do a roll, then stand still")
```

<img class="sr-media" src="../assets/media/agent_duck.gif" alt="the agent drives the duck">

The language model never sets joint positions. It selects a skill and its parameters through `duck.tool`; the skill
drives the motors and the run loop applies the safety chain. See [Agent tools](tools.md).

## LeRobot policies

Any LeRobot checkpoint can be loaded in-process or reached on a policy server. See
[Running a LeRobot policy](lerobot.md).

## Teleoperation with a leader arm

{{drawing:d18_leader_source}}

Teleoperation is a run whose policy is the leader. Pass a LeRobot `Teleoperator` (or a `Robot` to mirror) to `run()`:

```python
# a LeRobot SO-101 leader on its own serial port drives the follower for a minute
arm.run(leader, hz=30, seconds=60)
```

`as_policy` wraps the leader in a `SourcePolicy`. The leader's bus declares each motor's norm mode, so the follower
converts every reading through its own calibration, as LeRobot's `teleoperate` does. A degrees leader drives a
range-mode follower to within one encoder count. A source that declares no units is refused, naming the fix. A raw
`{"shoulder_pan.pos": ...}` dict passed to `act()` is refused too, because `act()` takes `<joint>` targets in radians,
not LeRobot names in the policy's units.
