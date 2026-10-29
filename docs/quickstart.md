---
description: Five minutes, one duck. Then two robots, a recording, and an agent.
cta: true
---

# Quickstart

## Install

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[all]"
uv pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@53feaef37b67e8f4c23b6953e880ff535e39d6fe"  # LeRobot main
```

`import strands_robots` only needs numpy. Everything heavier loads the first time you use it, and tells you which
extra to install if it's missing.

## Make a duck walk

<div class="sr-pair" markdown>
<div markdown>

<!-- test: network -->
```python
from strands_robots import Policy, Robot

duck = Robot("microduck")
walk = Policy("microduck?vx=0.3")
duck.run(walk, seconds=10)
print(duck.pose.position)  # ~1.3 m
```

That's a real ONNX walking policy from Pollen Robotics, running in MuJoCo. Swap `vx` for `wz=1.0` and it turns.
Swap the skill and it does something else:

```python
roll = Policy("microduck?skill=roulade")
duck.run(roll, seconds=3)  # a forward roll
```

</div>
<div markdown>
<img class="sr-media" src="../assets/media/skill_walk.gif" alt="the microduck walking">
</div>
</div>

## Two robots, one world

```python
arm, g1 = Robot("so101"), Robot("g1", position=[1, 0, 0])
arm.sim.run({arm: lambda obs: {"shoulder_pan": 0.4}, g1: Policy("holosoma")}, seconds=5)
```

Both share the same physics and the same clock. A policy can be a whole neural network or a one-line function.

## Keep what happened

```python
arm.sim.camera("front", size=(320, 240), attach=arm)
arm.run(lambda obs: {"elbow_flex": 0.4}, seconds=3, record="datasets/so101-wave")
```

That folder is now a LeRobot v3 dataset. Run it again and you get a second episode.

## Let an agent drive

```python
from strands import Agent

Agent(tools=[arm.tool, arm.sim.tool])("look at the arm, then open its gripper")
```

You need Amazon Bedrock credentials in your environment.

## See it

```python
from strands_robots.dashboard import Dashboard

Dashboard(arm.sim).serve(port=8090)   # open http://127.0.0.1:8090
```

Leave it running and the [home page](index.md) of these docs shows it live.
