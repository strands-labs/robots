---
description: robot.tool, sim.tool and mesh.tool. Generated from the robot, so they're never out of date.
---

# Agent tools

<!-- test: skip, needs AWS credentials for a Bedrock agent turn -->
```python
from strands import Agent
from strands_robots import Robot

arm = Robot("so101")
Agent(tools=[arm.tool, arm.sim.tool])("look at the arm, then move shoulder_pan to 0.5")
```

On a real turn, Claude looked first, then moved the joint: it landed at 0.469, the motor still settling toward 0.5.

| tool | what the agent can do |
|---|---|
| `robot.tool` | describe, observe, act, run a policy, stop, hold, estop |
| `sim.tool` | add robots, objects and cameras, step, reset, render a picture to look at |
| `mesh.tool` | the same, for every robot on the mesh |

Nobody writes these by hand. They're generated from the robot's manifest and method signatures, so a new skill or a
new joint shows up in the tool automatically.

What an agent **can't** do: approve its own moves on a real robot (the operator decides, every step of a run),
resume after an e-stop, calibrate (that turns the torque off), or plug in hardware through `sim.tool`. It *can* hit
e-stop. That one's always allowed.
See the duck taking orders on the [policies page](policies.md#skills-an-agent-a-robot-that-listens).
