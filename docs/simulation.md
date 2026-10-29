---
description: One world, many robots. A Simulation owns time and the scene; a Backend steps the physics.
---

# One world

{{drawing:d02_one_world}}

```python
from strands_robots import Robot, Simulation

arm = Robot("so101"); g1 = Robot("g1", position=[1, 0, 0])   # two bodies, no world given, so they join the default one
sim = Simulation.default()        # MuJoCo, headless; the world arm and g1 joined
sim.robots                        # {"so101": ..., "g1": ...}: an id is the name as written, unique: a second so101 is "so101_2"
sim.physics                       # Physics("mujoco", timestep=0.002, substeps=10): control_dt is 20 ms
sim.control_step()                # what act() costs a simulated robot: one control step for the whole world
with sim.manual_time():           # inside, act() only writes targets; the caller calls sim.step(n)
    ...
sim.set_pose("g1", position=[2, 0, 0.8])   # teleport; velocities zeroed, time does not advance
sim.add_object("box", shape="box", position=[0.3, 0, 0.1], size=[0.03, 0.03, 0.03])
sim.camera("wrist", position=[0.035, 0, -0.02], target=[-0.2, 0, -0.09], attach="so101/gripper")  # link frame
sim.render()                      # HxWx3 uint8 from the overview camera, fitted to every robot's geometry
sim.pose("box")                   # where it is now: position (m), quaternion (w, x, y, z)
sim.pose("so101/gripper")         # a link too (any name in describe()["robots"]["so101"]["links"]): the end effector
sim.remove("wrist")               # a robot, an object or a camera, by name
sim.describe()                    # robots, objects where they are now, cameras, physics, time; JSON-safe
```

A `Simulation` owns the `Scene` (plain data: robots, objects, cameras, gravity, timestep), the robots
in it and the time. Every scene change (`add`, `remove`, `add_object`, `camera`, `set_pose`,
`randomize`) bumps `revision`; motion does not. Adding a robot appends to the scene and rebuilds the
world, and every body that survives keeps its joint state. `sim.lock` is one re-entrant lock around
the world, so a reader and a stepping loop never see a half-built world. It is fair: a release hands it to
the oldest waiter, so a reader looping on `with sim.lock:` never starves an `act()`.

`set_pose` moves a fixed base by moving its spawn frame in the scene and rebuilding, the path every
scene change takes: joint state, time and randomised physics carry over. A floating base is teleported in place and keeps its spawn frame, so `reset()` stands it back
on its keyframe at the spawn position.

`attach="so101"` puts a camera on the robot's base; `attach="so101/gripper"` on one of its links (the
wrist view a VLA dataset expects). `describe()["robots"]["so101"]["links"]` lists the link names.
Vectors are numbers, never strings: `position` 3, `orientation` a `(w, x, y, z)` quaternion, `color` RGBA in
0-1; a string or a wrong length is refused by name. `reset(keyframe=)` refuses a keyframe no robot has.
`randomize` takes a factor or a `(lo, hi)` range and returns what it drew: the world factors, `scales[<target>]` per targeted body, `camera_jitter` per camera. `sim.tool` offers all of these to
an agent, and `run({"a": spec, "b": spec})` drives robots in lock-step by id.

`Simulation()` refuses what it cannot be: an unknown backend, `physics=` other than `native` or `newton`,
`substeps` below 1, `num_envs` above 1 without `physics="newton"`, an unknown keyword. Without the `mujoco` extra it raises an `ImportError` that
says `pip install 'strands-robots[mujoco]'`, not a bare `ModuleNotFoundError`.

{{drawing:d08_missing_extra}}

A world's floor and light are sized for a camera, not the eye. `ground=` chooses the floor: the default
checker texture gives a policy edges and colour instead of a white blob on white, `"plain"` is a flat grey
plane and `False` is none; the default lighting (headlight diffuse 0.5, ambient 0.15) leaves `light_scale`
headroom in both directions. An unknown `ground=` is refused by name.

{{drawing:d14_world_a_camera_sees}}

## The backend contract

`Backend` is an abstract class: `load`, `step`, `reset`, `time`, `joint_state`, `set_targets`,
`body_pose`, `set_pose`, `set_gains`, `render`, `contacts`, `physics` and `close`. `mujoco` is the
first backend (MuJoCo 3.3.4 or newer): one `MjSpec`, each robot attached under an `id/` prefix,
rebuilt on every scene change with time kept and joint state carried by joint name. `mujoco` is imported when the first world is made, not by
`import strands_robots`. Rendering uses EGL when no display is present: `import strands_robots` sets `MUJOCO_GL=egl` (or `osmesa`) on a headless Linux host before anything imports `mujoco`, which reads it only once. An explicit `MUJOCO_GL` is respected; a display or macOS keeps the native context. If `mujoco` was imported first, `render()` raises and names the variable to set before that import. Every render runs on one GL thread, whichever thread asks, so a `start()` after a `run()` renders from its background thread too. Its contexts are freed before EGL shuts down at exit, so a script that rendered ends quietly; an `EGL_NOT_INITIALIZED` traceback after the last line of a script is a teardown bug to report, not a failed render.

Measured headless: 2000 physics steps of so101 plus unitree_g1 take 85 us each (MuJoCo 3.15, one CPU core).

## Sim to real

{{drawing:d11_sim_to_real}}

```python
with sim.editing():                          # five verbs, one rebuild when the block exits
    sim.add_object("cube", shape="box", position=[0.25, 0, 0.05], size=[0.015] * 3)
    sim.camera("wrist", attach="so101/gripper")
sim.randomize(target="cube", mass=(0.5, 2.0), friction=(0.7, 1.3), seed=0)  # this body only
sim.randomize(camera=0.01, lighting=(0.6, 1.2))   # every camera within 1 cm per axis, the whole light
arm.run(Perturb(policy, joint_noise=0.005, image_noise=0.02, action_delay=2), seconds=10)
```

`randomize` without `target` scales every body, the robot's links included. `Perturb` (from
`strands_robots.policy`) wraps any policy: only the policy sees the noisy joints and pixels, and each action
lands `action_delay` control steps late (the robot holds its measured pose until the first one arrives).
Inside `editing()` the scene changes but the world is rebuilt once, on exit; a failed rebuild restores
the scene as it was. `step`, `render` and `set_pose` flush pending edits first.

`describe()["physics"]["deterministic"]` says whether the same scene and actions replay bit-exactly.
Measured on one NVIDIA Blackwell GPU (CUDA 13), so101 plus a cube, 200 control steps:

| physics | deterministic | run to run |
|---|---|---|
| `native` (MuJoCo) | true | bit-exact |
| `newton`, 1 env | true | bit-exact |
| `newton`, 1024 envs | false | env 0 differs by 3e-8 |

MuJoCo and Newton are different integrators: over the same 4 s the joints agree to 1.5e-7 rad and the
cube to 4.7 mm.

## Memory: what a robot costs

A MuJoCo world keeps the compiled model and its data, not the `MjSpec` it was built from (the spec holds every
mesh a second time), and robots of one model share its meshes in that compiled model (`MjSpec.dedup_assets`,
MuJoCo 3.17+). `python scripts/profile_robot_rss.py 8` prints resident memory as robots join one world;
measured on an aarch64 host with MuJoCo 3.17:

| robots `so101` | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| keeping the spec | 648 MB | 955 MB | 1623 MB | 3027 MB |
| one mesh set per robot | 623 MB | 679 MB | 881 MB | 1185 MB |
| one mesh set per model, now | 611 MB | 646 MB | 682 MB | 729 MB |

An added so101 costs about 17 MB: the compiled model stays at 62 MB however many so101 join, since their
meshes and mesh collision trees are stored once. The Python heap stays near 5 MB throughout: the growth is
MuJoCo's C memory, which `tracemalloc` does not see.

## GPU worlds: the first scene is slow

`physics="newton"` compiles warp kernels for each new scene shape and caches them (`WARP_CACHE_PATH`,
default `~/.cache/warp`); a load slower than 5 s logs the compile time. Build a scene inside
`with sim.editing():` so it compiles once. Measured on one NVIDIA Blackwell GPU (CUDA 13):

| step | cold cache | warm cache |
|---|---|---|
| `Simulation(physics="newton")` | 10.2 s | 1.3 s |
| `Robot("so101")` | 42.8 s | 0.63 s |
| `add_object("cube")` | 22.9 s | 0.55 s |
| `step(1)`, steady | 0.41 ms | 0.41 ms |

`Simulation("isaac")` never accepts NVIDIA's Isaac Sim EULA for you: it raises until you set
`OMNI_KIT_ACCEPT_EULA=YES` after reading the
[license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html).
The first scene change after a fresh Isaac install warms Kit's caches and can take minutes; the app logs a warning
when it starts.
