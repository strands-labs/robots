---
description: robot.tool, sim.tool and mesh.tool give a Strands agent a robot, a world or a whole mesh.
---

# Agent tools

Each robot, world and mesh has a `.tool` you can give to a Strands agent.

<!-- test: skip, needs AWS credentials for a Bedrock agent turn -->
```python
from strands import Agent
from strands_robots import Robot

arm = Robot("so101")
Agent(tools=[arm.tool, arm.sim.tool])("look at the arm, then move shoulder_pan to 0.5")
```

| Tool | The agent can |
|---|---|
| `robot.tool` | Describe, observe, act, run a policy, stop, hold, e-stop |
| `sim.tool` | Add robots, objects and cameras, step, reset, render an image |
| `mesh.tool` | Do the same for every robot on the mesh |

The tools are built from the robot's manifest. A new joint or skill appears in the tool without extra code.
Each joint lists its unit and range, so the agent sees them.

Calls to one world run in the order the agent sent them.
[`agent_builds_a_scene.py`](https://github.com/strands-labs/robots/blob/main/notebooks/examples/agent_builds_a_scene.py)
shows an agent building a scene and looking at it.

## Give the agent your policies

```python
arm.tool(policies={"wave": my_policy})
```

The agent can then call `run(policy="wave", ...)`. Give a remote policy this way too. The agent cannot pass
a network address itself.

A robot reached over the mesh counts as a real robot. An agent's `run` through `mesh.tool` is refused unless you
build the tool with the operator's answer:

```python
from strands_robots.mesh.tool import use_mesh

tool = use_mesh(lab, approve=lambda plan: input(f"allow {plan}? [y/N] ") == "y")
```

A device that offers skills (`mesh.mount(robot, skills={...})`, see [Mesh networking](mesh.md)) is different: the
agent's `run` names one of them, `status` reports it, and the policy runs on the device.

## What an agent cannot do

| Action | Who does it |
|---|---|
| Approve a run on a real robot | The operator, once per plan |
| Choose `record`, `push` or `units` | You, in code |
| Resume after an e-stop | The operator |
| Calibrate (it turns torque off) | You |

An agent can always call e-stop. See [Let an agent pick skills](policies.md#let-an-agent-pick-skills).
