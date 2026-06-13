---
description: Compose non-trivial scenes — multiple robots, tables, obstacles, custom MJCF.
---

# World building

```python
from strands_robots import Robot

sim = Robot("so100")                              # one arm on flat ground plane
sim.add_robot(name="so100", position=[0.0, 0.5, 0.0])   # second arm

sim.add_object(name="table", shape="box", size=[0.5, 0.5, 0.02],
               position=[0.0, 0.0, 0.0], color=[0.5, 0.3, 0.1, 1.0], mass=20.0)

sim.add_camera(name="overhead", position=[0.0, 0.0, 1.5], target=[0.0, 0.0, 0.0])
```

## Strategies

| Need | Approach |
|------|----------|
| Add robots / objects incrementally | `add_robot` / `add_object` / `add_camera` |
| Replace entire world | `load_scene(scene_path=...)` |
| Procedural scene | loop over `add_object` |
| Raw MJCF tweak without recompile | `patch_scene_mjcf(ops)` |

## Procedural objects

```python
import random

sim = Robot("so100")

for i in range(5):
    sim.add_object(
        name=f"cube_{i}", shape="box", size=[0.025, 0.025, 0.025],
        position=[random.uniform(0.2, 0.5), random.uniform(-0.15, 0.15), 0.025],
        color=[random.random(), random.random(), random.random(), 1.0],
    )
```

## Cameras

Free cameras look from `position` toward `target` (`fov=60.0`, `width=640`, `height=480`). Robot-URDF cameras (wrist, etc.) are auto-discovered on `add_robot` — no `add_camera` needed.

## Multi-robot policies

```python
from strands_robots.policies import create_policy

sim.run_multi_policy(
    policies={"so100": create_policy("mock"), "panda": create_policy("mock")},
    instructions={"so100": "pick cube", "panda": "hold tray"},
    duration=10.0,
)
```

## See also

- [Simulation overview](overview.md)
- [Domain randomization](domain-randomization.md)
- [Simulation overview](../simulation/overview.md)
- [LIBERO benchmark](https://github.com/strands-labs/robots/tree/main/strands_robots/benchmarks/libero)
