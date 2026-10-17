---
description: 76 robots in the registry. Load one by name and it tells you everything about itself.
---

# Robots

{{drawing:d01_description}}

```python
from strands_robots import Description

d = Description.load("so101")
d.action_keys     # ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
d.limits["gripper"]
```

A `Description` is everything about a robot that doesn't depend on where it is: its joints, their limits, its
keyframes, how it's driven on real hardware. It's read straight from the robot's model file, which is downloaded once
and cached. The registry knows 76 robots (arms, hands, humanoids, quadrupeds, a duck). Ask for one it doesn't know and
it says so.

Facts about a robot are **data**, not code: its LeRobot units, its driver, its safe ranges all live in
`registry/robots.json`. Adding a robot means adding an entry, not editing Python.

| what flows around | |
|---|---|
| `Observation` | joints, base pose, camera images, time |
| `Action` | a number per joint, or `{joint: target}` for just the ones you care about |
| `JointState`, `Pose` | positions, velocities, efforts · position and orientation |

The robot speaks radians. A policy trained in LeRobot speaks something else (degrees, or -100..100). The unit
conversion is automatic: the policy says what it speaks, the robot converts.
