---
description: A Simulation holds the scene, the robots in it and the clock. MuJoCo by default, Isaac Sim and Newton on a GPU.
---

# Simulation

A `Simulation` is a world. It holds the robots, objects, cameras and lights, and it owns the clock.

{{drawing:d02_one_world}}

```python
from strands_robots import Robot, Simulation

arm = Robot("so101")
sim = Simulation.default()  # the world arm joined
sim.add_object("box", shape="box", position=[0.3, 0, 0.1], size=[0.03] * 3)
sim.camera("wrist", attach="so101/gripper")
frame = sim.render()        # an HxWx3 image
print(sim.pose("box").position)
```

Robots made without `sim=` join `Simulation.default()`. It uses MuJoCo and needs no display.

## Scene methods

| Method | Does |
|---|---|
| `add_object(name, shape=, position=, size=)` | Add a box, sphere or other shape. `size` is half-sizes in metres. |
| `camera(name, position=, target=, attach=)` | Add a camera, or re-aim one with the same name |
| `light(name, position=, color=)` | Add a spot light. `directional=True` makes a sun. |
| `set_pose(name, position=)` | Teleport a robot or object |
| `remove(name)` | Remove a robot, object, camera or light |
| `pose(name)` | Where a body or link is now |
| `render()` | An image from the overview camera |
| `reset(keyframe=)` | Return every robot to a keyframe |
| `describe()` | The whole world as JSON-safe data |

`attach="so101"` puts a camera on the robot base. `attach="so101/gripper"` puts it on a link.
`describe()["robots"]["so101"]["links"]` lists the link names.

`describe()["robots"][id]["reach"]` tells you which way each joint moves the end effector. Read it before you
guess a sign.

## Time

`act()` on a simulated robot advances the whole world by one control step (20 ms by default). To step the world
yourself, use `manual_time()`:

```python
with sim.manual_time():
    arm.act({"shoulder_pan": 0.2})  # only writes the target
    sim.step(10)
```

## Many changes at once

Each scene change rebuilds the world. Group changes in `editing()` to rebuild once:

```python
with sim.editing():
    sim.add_object("cube", shape="box", position=[0.25, 0, 0.05], size=[0.015] * 3)
    sim.camera("front", position=[0.6, 0, 0.3], target=[0, 0, 0.1])
```

## Sim to real

{{drawing:d11_sim_to_real}}

Randomize physics and add noise so a policy does not overfit one world:

```python
from strands_robots.policy import Perturb

sim.randomize(target="cube", mass=(0.5, 2.0), friction=(0.7, 1.3), seed=0)
sim.randomize(camera=0.01, lighting=(0.6, 1.2))
arm.run(Perturb(my_policy, joint_noise=0.005, action_delay=2), seconds=2)
```

`Perturb` adds noise only to what the policy sees, and delays each action by `action_delay` steps.

## Floor and rendering

`ground=` sets the floor: the default checker, `"plain"`, or `False` for none. On a headless Linux machine the
library sets `MUJOCO_GL=egl` for you. If you import `mujoco` before `strands_robots`, set `MUJOCO_GL` yourself.

## Other backends

`Simulation("isaac")` and `Simulation(physics="newton")` run on a CUDA GPU with the same API. See
[GPU simulation](gpu.md).
