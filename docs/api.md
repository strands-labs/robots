---
description: The classes exported from strands_robots, and the subpackages that hold the rest of the API.
---

# API reference

```python
from strands_robots import Robot, Simulation, Policy, Body, Description, Observation, Action, JointState, Pose
```

These are the main classes. Import the rest from their subpackages:

| Subpackage | Holds |
|---|---|
| `strands_robots.mesh` | `Mesh`, `authority` |
| `strands_robots.robot.safety` | `RequestRefused` |
| `strands_robots.recording` | Dataset recording |
| `strands_robots.dashboard` | `Dashboard` |
| `strands_robots.policy` | `Perturb`, LeRobot policies |

## Robot

::: strands_robots.Robot

## Simulation

::: strands_robots.Simulation

## Policy

::: strands_robots.Policy

## Description

::: strands_robots.Description

## Observation, JointState, Pose

::: strands_robots.Observation

::: strands_robots.JointState

::: strands_robots.Pose
