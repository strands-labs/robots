---
hide: [navigation, toc]
template_class: sr-home
title: strands-robots
---

# strands-robots

<div class="sr-hero" markdown>
<div markdown>
<p class="sr-hero__title">A world.<br>A robot.<br>A <em>brain</em>.</p>
<p class="sr-hero__lead">That's the whole library. A <code>Simulation</code> is the world. A <code>Robot</code> lives in it, or on your desk. A <code>Policy</code> is the brain that decides what it does next. Same code in sim and on a real arm, and your agent can drive all of it.</p>
<div class="sr-hero__actions">
<a class="sr-btn sr-btn--primary" href="quickstart/">Make a duck walk</a>
<a class="sr-btn" href="status/">Where v1 is</a>
<span class="sr-install">uv pip install -e ".[mujoco,onnx,mesh]"<button class="sr-copy" data-clipboard-text='uv pip install -e ".[mujoco,onnx,mesh]"'>copy</button></span>
</div>
</div>
<div class="sr-hero__stage" markdown>
<div class="sr-stage" id="hero-stage" data-live="http://127.0.0.1:8090/">
<img src="assets/media/agent_duck.gif" alt="An agent drives the microduck: walk, turn, roll, stand">
<span class="sr-stage__badge" id="hero-badge">recorded · run a dashboard on :8090 and this goes live</span>
</div>
</div>
</div>

## Hi. Here's the idea in four snippets.

<div class="sr-pair" markdown>

```python title="1 · a world with two robots in it"
from strands_robots import Robot

arm = Robot("so101")                  # a little arm
g1 = Robot("g1", position=[1, 0, 0])  # a humanoid, same room
assert arm.sim is g1.sim              # one world, shared physics
```

```python title="2 · give one a brain"
from strands_robots import Policy

g1.run(Policy("holosoma"), seconds=10)     # it stands up
arm.run(Policy("lerobot/smolvla_base"))    # any LeRobot model
arm.run(Policy("zenoh://gpu:7447/smolvla"))  # ...or one on a GPU box
```

```python title="3 · make it safe to share"
arm.manifest                 # what it is, what it accepts
arm.estop()                  # stops. always. from anywhere.
arm.resume(approved_by="you")
```

```python title="4 · let an agent drive"
from strands import Agent

Agent(tools=[arm.tool, arm.sim.tool])(
    "look around, then wave at the humanoid")
```

</div>

## What that unlocks

<div class="sr-pair" markdown>
<div markdown>

**🤖 Any robot, by name.** 76 in the registry, MuJoCo Menagerie and `robot_descriptions` models, or your own
MJCF/URDF. Same verbs for all; on a USB port the same `Robot` drives a real SO-101.

**🌍 Worlds an agent can build.** `sim.tool` lets an agent add robots, objects and cameras, step time and look. Describe
a scene and the agent can put it together.

**🧠 Any brain, here or there.** Your function, an ONNX skill, or any LeRobot checkpoint, in your process or on a GPU
across the room.

</div>
<div markdown>

**📼 Every episode is a dataset.** `record=` on any run writes LeRobot v3, in sim or on the real arm. Collect, train,
run, repeat.

**🕸️ Robots that find each other.** A mesh with zero setup: discover, command, stream, and one e-stop for the room.

**🛑 Safe enough to share.** A manifest, one safety chain, and an e-stop nothing can override.

</div>
</div>

## Talk to it

You ask in plain words. The agent picks the robot's skills, one tool call at a time, and the robot does them.
No glue code, no hand-written tool specs: the tool is generated from the robot itself.

<div class="sr-pair" markdown>
<div markdown>
<img class="sr-media" src="assets/media/agent_duck.gif" alt="Claude drives the microduck through walk, turn, roll and stand">
</div>
<div markdown>

```text
you    walk forward a bit, turn left, do a roll, then chill

agent  run  velstand vx=0.3        2 s
agent  run  velstand wz=0.8        2 s
agent  run  roulade                3 s
agent  run  velstand (stand still) 2 s
agent  Done! It walked, turned, rolled, and it's chilling.
```

A real turn on Claude Sonnet 4.5. The duck's skills are Pollen Robotics' open policies; the agent strings them
together. That's the agentic loop, small enough to read in one go. [How tools work →](tools.md)

</div>
</div>

## What it can do today

| | |
|---|---|
| 🧠 **Any policy, any robot** | your function, an ONNX skill or a LeRobot model; joints, cameras and units are matched for you, and a mismatch is refused before anything moves |
| 🤖 **Any LeRobot model, anywhere** | 13 policy families drive our robots: in your process, or served from a GPU over the network |
| 🌍 **One world, many robots** | an arm and a humanoid share physics; a laptop steps both in 85 µs |
| ⚡ **Big batches on a GPU** | 1,024 copies of the world at once: 1.75 million steps a second on one L40S |
| 🛑 **Stop means stop** | an e-stop halts a running robot in about 10 ms, and nothing can override it |
| 🕸️ **Robots find each other** | on a LAN, with no setup, in about 50 ms; over AWS IoT from anywhere |
| 📼 **Every run can be a dataset** | `record=` writes LeRobot v3, ready to train on |
| 🪶 **Small** | about 10,000 lines. The version it replaces was 258,000. |
