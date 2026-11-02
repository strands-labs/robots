---
description: What a robot is, independent of any world, and the data that flows between a world, a robot and a policy.
---

# Robots and their data

{{drawing:d01_description}}

```python
from strands_robots import Description

d = Description.load("so101")       # a registry name, an alias, a robot_descriptions name or an MJCF/URDF path
d.action_keys   # ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
d.limits["gripper"], d.keyframes, d.floating
d.describe()    # JSON-safe summary
```

`Description.load(name)` resolves the name through the registry (`registry/robots.json`, 76 robots, and
`registry/urdf_robots.json`) to a model file, `model_path` (and its scene, `scene_path`). A `robot_descriptions` module is cloned on first use; an asset
with a `github` source is fetched once into `$STRANDS_ROBOTS_CACHE` (default `~/.cache/strands_robots`).
The joints, actuators, `action_keys`, limits and keyframes are read from the model itself: through MuJoCo when
the `mujoco` extra is installed, through a plain XML walk when it is not. A robot file that only compiles
inside its scene is read through the scene. A registry robot that ships no simulation model (hardware only)
refuses with `FileNotFoundError` naming it; a name that is not in the registry refuses with `KeyError`.

| type | what it holds |
|---|---|
| `JointState` | `names`, `positions`, `velocities`, `efforts` (rad, rad/s, N·m), read-only arrays |
| `Pose` | `position` (m), `orientation` (w, x, y, z), optional twist |
| `Observation` | `t`, `joints`, `pose`, `images`, `extras`; `to_lerobot()` is the one place LeRobot key names live |
| `Action` | a vector in `action_keys` order or a `{joint: target}` mapping |
| `Calibration` | per-joint bus calibration, `to_json` / `from_json` in LeRobot's file format |
| `Requirements` | what a policy needs (camera names and sizes, state length); `check(obs)` refuses naming the gap |
| `Physics`, `Contact`, `RunResult` | engine and step sizes, one contact, what a run measured |

Units: the robot speaks radians. `Units.for_robot(description, "lerobot")` maps them to the LeRobot norm
modes the registry names (`hardware.lerobot_norm`; an SO-10x: body joints -100..100, gripper 0..100).
