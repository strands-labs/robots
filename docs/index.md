---
hide: [navigation, toc]
template_class: sr-home
title: strands-robots
---

# strands-robots

<div class="sr-hero" markdown>
<div markdown>
<p class="sr-hero__title">Build robots<br>in <em>Python</em>.</p>
<p class="sr-hero__lead">strands-robots is a Python library for building robots. The same code drives a simulated robot and a real one. Any robot can be a tool for an agent.</p>
<div class="sr-hero__actions">
<a class="sr-btn sr-btn--primary" href="quickstart/">Quickstart</a>
<a class="sr-btn" href="install/">Installation</a>
<span class="sr-install">uv pip install "strands-robots[dashboard,mujoco] @ git+https://github.com/strands-labs/robots@dev"<button class="sr-copy" data-clipboard-text='uv pip install "strands-robots[dashboard,mujoco] @ git+https://github.com/strands-labs/robots@dev"'>copy</button></span>
</div>
</div>
<div class="sr-hero__stage" markdown>
<div class="sr-stage" id="hero-stage" data-live="http://127.0.0.1:8091/ http://127.0.0.1:8090/" data-state="probing">
<img src="assets/media/agent_duck.gif" alt="An agent drives the microduck: walk, turn, roll, stand">
<span class="sr-stage__badge" id="hero-badge">recorded</span>
</div>
<p class="sr-stage__start">Start a dashboard to see it live here: <span class="sr-install">python -m strands_robots.dashboard<button class="sr-copy" data-clipboard-text="python -m strands_robots.dashboard">copy</button></span></p>
</div>
</div>

## Core concepts

<div class="sr-pair" markdown>

```python title="Robots share a world"
from strands_robots import Robot

arm = Robot("so101")                  # an SO-101 arm
g1 = Robot("g1", position=[1, 0, 0])  # a Unitree G1 humanoid
assert arm.sim is g1.sim
```

```python title="Policies move them"
from strands_robots import Policy

g1.run(Policy("holosoma"), seconds=10)       # ONNX
arm.run(Policy("lerobot/smolvla_base"))      # LeRobot
arm.run(Policy("zenoh://gpu:7447/smolvla"))  # remote
```

```python title="Safety"
arm.manifest                   # controls and safe ranges
arm.estop()                    # stops now, from any thread
arm.resume(approved_by="you")  # a named operator clears it
```

```python title="Agents use them as tools"
from strands import Agent

Agent(tools=[arm.tool, arm.sim.tool])(
    "look at the scene, then open the gripper")
```

</div>

## What you get

| Area | Provides |
|---|---|
| Robots | 76 robots by name, or your own MJCF or URDF |
| Simulation | MuJoCo on CPU, Isaac Sim and Newton on GPU |
| Hardware | SO-100 and SO-101 arms, Unitree robots |
| Policies | Python functions, ONNX, LeRobot |
| Recording | LeRobot v3 datasets |
| Mesh | Robots on other machines, over Zenoh or AWS IoT |
| Dashboard | A 3D view, controls, cameras, chat and e-stops |

## Run it locally

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install "strands-robots[dashboard,mujoco] @ git+https://github.com/strands-labs/robots@dev"
python -m strands_robots.dashboard
```

Open `http://127.0.0.1:8090`. See [Dashboard](dashboard.md) for options.
