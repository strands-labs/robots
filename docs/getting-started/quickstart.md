---
description: Five minutes from install to a robot picking up a cube.
---

# Quickstart

```bash
uv pip install "strands-robots[sim-mujoco]"
```

```python
from strands_robots import Robot
import imageio.v3 as iio

sim = Robot("so100")                              # sim by default; CPU-only, no GPU
sim.step()

frame = sim.get_observation("so100")["default"]   # uint8 HxWx3
iio.imwrite("first_frame.png", frame)
```

> Headless box? `export MUJOCO_GL=osmesa` before importing. See [Troubleshooting](../troubleshooting.md).

## Add an object and run a policy

```python
sim.add_object(
    name="cube", shape="box", size=[0.025]*3,
    position=[0.3, 0.0, 0.025], color=[1, 0, 0, 1],
)

sim.run_policy(
    robot_name="so100",
    instruction="pick up the red cube",
    policy_provider="mock",   # no GPU; swap "groot" or "lerobot_local" with a real model
    duration=10.0,
)
```

## Drive with an agent

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100")
agent = Agent(tools=[robot])
agent("Add a red cube and pick it up using the mock policy")
```

## See also

- [Policy providers](../policies/overview.md) — GR00T, LeRobot Local, Cosmos 3.
- [Robot catalog](../robots/index.md) — all 68 robots.
- [Real hardware](../hardware/robot-control.md) — same code, `mode="real"`.
