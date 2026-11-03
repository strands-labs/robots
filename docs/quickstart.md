---
description: Walk a simulated robot, put two robots in one world, record a dataset, and give a robot to an agent.
cta: true
---

# Quickstart

First install the library and LeRobot. See [Installation](install.md) for details.

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install "strands-robots[all] @ git+https://github.com/strands-labs/robots@main"
uv pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@main"
```

## Walk a robot

<div class="sr-pair" markdown>
<div markdown>

<!-- test: network -->
```python
from strands_robots import Policy, Robot

duck = Robot("microduck")
walk = Policy("microduck?vx=0.3")
duck.run(walk, seconds=10)
print(duck.pose.position)  # about 1.3 m forward
```

This loads the microduck and an ONNX walking policy from the Hugging Face Hub. `vx=0.3` asks for 0.3 m/s forward.
Use `wz=1.0` to turn, or pick another skill:

```python
duck.run(Policy("microduck?skill=roulade"), seconds=3)  # a forward roll
```

</div>
<div markdown>
<img class="sr-media" src="../assets/media/skill_walk.gif" alt="the microduck walking">
</div>
</div>

## Put two robots in one world

```python
arm, g1 = Robot("so101"), Robot("g1", position=[1, 0, 0])
arm.sim.run({arm: lambda obs: {"shoulder_pan": 0.4}, g1: Policy("holosoma")}, seconds=5)
```

Both robots share one physics engine and one clock. A policy can be a model or a plain function.

## Record a dataset

```python
arm.sim.camera("front", size=(320, 240), attach=arm)
arm.run(lambda obs: {"elbow_flex": 0.4}, seconds=3, record="datasets/so101-wave")
```

This writes a LeRobot v3 dataset. Each run adds one episode. See [Recording a dataset](recording.md).

## Give a robot to an agent

```python
from strands import Agent

Agent(tools=[arm.tool, arm.sim.tool])("look at the arm, then open its gripper")
```

The default model runs on Amazon Bedrock, so you need AWS credentials. See [Agent tools](tools.md).

## Open the dashboard

```python
from strands_robots.dashboard import Dashboard

Dashboard(arm.sim).serve(port=8090)  # http://127.0.0.1:8090
```

See [Dashboard](dashboard.md).
