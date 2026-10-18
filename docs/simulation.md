---
description: One world, many robots. A Simulation owns time and the scene; a Backend steps the physics.
---

# One world

{{drawing:d02_one_world}}

```python
from strands_robots import Simulation

sim = Simulation.default()        # MuJoCo, headless; every Robot given no world joins this one
sim.robots                        # {"so101": ..., "unitree_g1": ...}: ids are unique, a second so101 is "so101_2"
sim.physics                       # Physics("mujoco", timestep=0.002, substeps=10): control_dt is 20 ms
sim.control_step()                # what act() costs a simulated robot: one control step for the whole world
with sim.manual_time():           # inside, act() only writes targets; the caller calls sim.step(n)
    ...
sim.set_pose("unitree_g1", position=[2, 0, 0.8])   # teleport; velocities zeroed, time does not advance
sim.add_object("box", shape="box", position=[0.3, 0, 0.1], size=[0.03, 0.03, 0.03])
sim.camera("wrist", position=[0.15, -0.25, 0.25], target=[0.2, 0, 0.05], attach="so101")
sim.render()                      # HxWx3 uint8 from the auto-framed overview camera that sees every robot
sim.pose("box")                   # where it is now: position (m), quaternion (w, x, y, z)
sim.remove("wrist")               # a robot, an object or a camera, by name
sim.describe()                    # robots, objects where they are now, cameras, physics, time; JSON-safe
```

A `Simulation` owns the `Scene` (plain data: robots, objects, cameras, gravity, timestep), the robots
in it and the time. Every scene change (`add`, `remove`, `add_object`, `camera`, `set_pose`,
`randomize`) bumps `revision`; motion does not. Adding a robot appends to the scene and rebuilds the
world, and every body that survives keeps its joint state. `sim.lock` is one re-entrant lock around
the world, so a reader and a stepping loop never see a half-built world.

`set_pose` moves a fixed base by moving its spawn frame in the scene and rebuilding, the path every
scene change takes: joint state, time and randomised physics carry over. A floating base is teleported in place and keeps its spawn frame, so `reset()` stands it back
on its keyframe at the spawn position.

Vectors are numbers, never strings: `position` 3, `orientation` a `(w, x, y, z)` quaternion, `color` RGBA in
0-1; a string or a wrong length is refused by name. `reset(keyframe=)` refuses a keyframe no robot has.
`randomize` takes a factor or a `(lo, hi)` range and returns the scales it drew. `sim.tool` offers all of these to
an agent, and `run({"a": spec, "b": spec})` drives robots in lock-step by id.

`Simulation()` refuses what it cannot be: an unknown backend, `physics=` other than `native`,
`substeps` below 1, `num_envs` above 1, an unknown keyword. Without the `mujoco` extra it raises an `ImportError` that
says `pip install 'strands-robots[mujoco]'`, not a bare `ModuleNotFoundError`.

{{drawing:d08_missing_extra}}

## The backend contract

`Backend` is an abstract class: `load`, `step`, `reset`, `time`, `joint_state`, `set_targets`,
`body_pose`, `set_pose`, `set_gains`, `render`, `contacts`, `physics` and `close`. `mujoco` is the
first backend (MuJoCo 3.3.4 or newer): one `MjSpec`, each robot attached under an `id/` prefix,
rebuilt on every scene change with time kept and joint state carried by joint name. `mujoco` is imported when the first world is made, not by
`import strands_robots`. Rendering uses EGL when no display is present (`MUJOCO_GL` is respected). Every render runs on one GL thread, whichever thread asks, so a `start()` after a `run()` renders from its background thread too. With `MUJOCO_GL=glfw` on macOS a render runs on the thread that asks: GLFW makes its window on the main thread only.

Measured headless: 2000 physics steps of so101 plus unitree_g1 take 85 us each (MuJoCo 3.15, one CPU core).
