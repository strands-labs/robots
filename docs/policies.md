---
description: A policy turns an observation into an action. Policy() takes a function, an ONNX skill, a LeRobot checkpoint, a policy server or a recorded episode.
---

# Policies

A policy decides what the robot does next. Each step, it gets an observation and returns joint targets.

{{drawing:v03_one_loop}}

```python
from strands_robots import Policy, Robot

arm = Robot("so101")
wave = Policy(lambda obs: {"shoulder_pan": 0.4})
arm.run(wave, seconds=2)
```

`Policy()` picks the right kind from what you pass:

| You pass | You get |
|---|---|
| `lambda obs: {...}` | Your function, called every step |
| `"microduck?skill=velstand&vx=0.3"`, `"holosoma"` | An ONNX skill |
| `"lerobot/smolvla_base"` | A LeRobot checkpoint in this process |
| `"zenoh://gpu-box:7447/smolvla"` | A LeRobot policy on another machine |
| `"data/wave?episode=0"` | A recorded episode, replayed |

A function can return `None` to keep the last target for that step.

## ONNX skills

The microduck has ten skills from Pollen Robotics. Pick one with a query string:

<div class="sr-grid2" markdown>
<figure><img class="sr-media" src="../assets/media/skill_walk.gif" alt="walk"><figcaption><code>skill=velstand&vx=0.3</code> walk</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/skill_turn.gif" alt="turn"><figcaption><code>skill=velstand&wz=1.2</code> turn</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/skill_roll.gif" alt="roll"><figcaption><code>skill=roulade</code> roll</figcaption></figure>
<figure><img class="sr-media" src="../assets/media/skill_pick.gif" alt="pick"><figcaption><code>skill=alpha_ground_pick</code> reach down</figcaption></figure>
</div>

The others are `alpha_walking`, `alpha_stand`, `alpha_sitstand`, `ball_kick_left`, `ball_kick_right`, `roller`
and `roller_crouch`. `Policy("holosoma")` makes a Unitree G1 stand and walk.

## Let an agent pick skills

<!-- test: skip, needs AWS credentials for a Bedrock agent turn -->
```python
from strands import Agent

duck = Robot("microduck")
Agent(tools=[duck.tool])("walk forward a bit, turn left, do a roll, then stand still")
```

The agent chooses a skill and its parameters. It never sets joint positions. See [Agent tools](tools.md).

## LeRobot

See [Running a LeRobot policy](lerobot.md).

## Teleoperate with a leader arm

{{drawing:d18_leader_source}}

Pass a LeRobot `Teleoperator`, or another `Robot`, to `run()`. The follower copies it:

```python
arm.run(leader, hz=30, seconds=60)
```

The library converts each reading through both arms' calibration.
