---
description: Walk a simulated robot with an ONNX policy, put two robots in one world, record a dataset, and hand a robot to an agent.
cta: true
---

# Quickstart

This page assumes the [installation](install.md) is done:

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[all]"
uv pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@53feaef37b67e8f4c23b6953e880ff535e39d6fe"  # LeRobot main
```

## Run a walking policy

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

`Robot("microduck")` loads the microduck model into the default MuJoCo world. `Policy("microduck?vx=0.3")` loads
Pollen Robotics' ONNX walking policy from the Hugging Face Hub with a forward velocity command of 0.3 m/s. Use
`wz=1.0` to turn in place, or select another skill:

```python
roll = Policy("microduck?skill=roulade")
duck.run(roll, seconds=3)  # a forward roll
```

</div>
<div markdown>
<img class="sr-media" src="../assets/media/skill_walk.gif" alt="the microduck walking">
</div>
</div>

## Two robots in one world

```python
arm, g1 = Robot("so101"), Robot("g1", position=[1, 0, 0])
arm.sim.run({arm: lambda obs: {"shoulder_pan": 0.4}, g1: Policy("holosoma")}, seconds=5)
```

Both robots share the same physics and the same clock. A policy can be a neural network or a function that returns
joint targets.

## Record a dataset

```python
arm.sim.camera("front", size=(320, 240), attach=arm)
arm.run(lambda obs: {"elbow_flex": 0.4}, seconds=3, record="datasets/so101-wave")
```

`datasets/so101-wave` is now a LeRobot v3 dataset. Each run on the same directory appends one episode. See
[Recording a dataset](recording.md).

## Give a robot to an agent

```python
from strands import Agent

Agent(tools=[arm.tool, arm.sim.tool])("look at the arm, then open its gripper")
```

The default Strands model runs on Amazon Bedrock, so AWS credentials must be available in the environment. See
[Agent tools](tools.md).

## Open the dashboard

```python
from strands_robots.dashboard import Dashboard

Dashboard(arm.sim).serve(port=8090)   # http://127.0.0.1:8090
```

While it runs, the [home page](index.md) of a locally served copy of these docs shows the live dashboard. See
[Dashboard](dashboard.md).
