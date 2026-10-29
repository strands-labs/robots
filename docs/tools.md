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

One agent can hold both: [`agent_builds_a_scene.py`](https://github.com/strands-labs/robots/blob/dev/notebooks/examples/agent_builds_a_scene.py)
gives it `sim.tool` and `arm.tool`; it places three cubes and a camera, moves `shoulder_pan` and renders what it built.
Calls on one world run in the order the model sent them, even under strands' default executor, which starts every
call of one message at once: `camera c`, `act`, `render c` in one message renders from `c` after the arm moved. No
`SequentialToolExecutor()` needed.

Nobody writes these by hand. They're generated from the robot's manifest and method signatures, so a new skill or a
new joint shows up in the tool automatically. Each `act` joint carries its unit and range from the manifest, each
rpc reads its docstring's first paragraph and its arguments the rest. A `policy` argument lists the shipped presets
that drive that robot (`holosoma` for `unitree_g1`, `microduck` for `microduck`, none for `so101`) and every spec form
an agent can send, so its first `run` names a real policy.

A policy the developer built — a remote one included — is handed to the agent as a **handle**:
`arm.tool(policies=[ttt])` is the same tool with `run(policy="ttt", instruction=..., seconds=...)` resolving to that
object (named by `.name`, a remote policy's deployment, or the keys of a dict; with exactly one handle, `policy` may be
left out). The model never types a network address — `policy="zenoh://..."` from a model stays refused — and on a real
arm the run still goes through `approve=` once and the safety chain every step.

What an agent **can't** do: approve its own run on a real robot (the operator says yes once, to the whole plan),
resume after an e-stop, calibrate (that turns the torque off), or plug in hardware through `sim.tool`. It *can* hit
e-stop. That one's always allowed.
See the duck taking orders on the [policies page](policies.md#skills-an-agent-a-robot-that-listens).
