---
description: Load a robot model by name or path and read its joints, limits and keyframes. The data types a world, a robot and a policy pass to each other.
---

# Robot descriptions

A `Description` is a robot model plus what the library reads from it: joints, actuators, limits and keyframes.

{{drawing:d01_description}}

```python
from strands_robots import Description

d = Description.load("so101")
d.action_keys       # ('shoulder_pan', ..., 'gripper')
d.limits["gripper"]  # (lo, hi) in radians
d.keyframes, d.floating
d.describe()        # a JSON-safe summary
```

`Description.load` takes a registry name, an alias, a `robot_descriptions` name, or a path to an MJCF or URDF file.
The registry has 76 robots. Models are downloaded on first use into `$STRANDS_ROBOTS_CACHE`
(default `~/.cache/strands_robots`). A model hosted on GitHub is fetched at the commit its registry entry pins
(`asset.source.commit`) and must hash to the entry's `asset.source.sha256` (`description.assets.tree_digest`), or
nothing is cached. The joint limits come from these files, so they change only with a registry edit.

| Error | Cause |
|---|---|
| `KeyError` | The name is not in the registry. |
| `FileNotFoundError` | The robot is hardware only and has no simulation model. |
| `AssetMismatch` (an `OSError`) | A GitHub-hosted model is not the commit or content its registry entry pins. |

## Data types

| Type | Contents |
|---|---|
| `JointState` | `names`, `positions`, `velocities`, `efforts` |
| `Pose` | `position` in metres, `orientation` as (w, x, y, z) |
| `Observation` | `t`, `joints`, `pose`, `images`, `extras`; `to_lerobot()` converts it |
| `Action` | A vector in `action_keys` order, or a `{joint: target}` dict |
| `Calibration` | Bus calibration, in LeRobot's file format |
| `Requirements` | What a policy needs; `check(obs)` names what is missing |
| `RunResult` | What a run measured |

## Units

Joints use radians. `Units.for_robot(description, "lerobot")` converts to LeRobot's ranges. For an SO-10x
that is -100..100 for body joints and 0..100 for the gripper.
