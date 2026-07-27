---
description: Compose non-trivial scenes - multiple robots, tables, obstacles, custom MJCF.
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

## Setup entry points

`Robot("so100")` is the one-step way to get a ready-to-drive engine: it builds
the world and adds the named robot for you. Constructing a backend directly -
`create_simulation("mujoco")` or `Simulation()` - gives an **empty** engine; you
then call `create_world()` and `add_robot("so100")` yourself.

`robot_name` therefore belongs to `Robot(...)` and `add_robot(...)`, never to a
backend constructor. Passing it to the constructor
(`Simulation(robot_name="so100")`) is rejected with a `TypeError` rather than
silently ignored, so the mistake is caught up front instead of surfacing later
as an unrelated `No world` error.

## Strategies

| Need | Approach |
|------|----------|
| Add robots / objects incrementally | `add_robot` / `add_object` / `add_camera` |
| Replace entire world | `load_scene(scene_path=...)` |
| Procedural scene | loop over `add_object` |
| Raw MJCF tweak without recompile | `patch_scene_mjcf(ops)` |

## Spawn pose (keyframes)

By default a robot spawns at the all-zero joint configuration. Many MuJoCo
Menagerie models ship a canonical ready pose in a MJCF `<keyframe>` (panda,
ur5e, fr3, kuka `home`; aloha `neutral_pose`; quadrupeds/humanoids a standing
`home`). Pass `keyframe=` to spawn in that pose instead - important when a
policy was trained from the home pose, since the zero configuration is
out-of-distribution:

```python
sim.add_robot(name="panda", data_config="panda", keyframe="home")  # or keyframe=0
```

The pose is applied to the robot's joints by name and is restored by `reset()`,
so a keyframe spawn is sticky across episodes. It also survives later
`add_robot` calls, so incrementally building a multi-robot scene keeps every
already-spawned arm at its home pose rather than collapsing it to zero. An
unknown keyframe name/index
is an error that lists the model's available keyframes. `keyframe=None` (the
default) keeps the zero-pose spawn. (MuJoCo backend; the Newton backend rejects
`keyframe=` as not-yet-supported.)

## Rough terrain

By default `create_world()` lays down a flat ground plane. A locomotion
policy is only interesting on ground it can trip on, so pass a `terrain=`
kind to lay down a deterministic heightfield instead - a floating-base robot
then settles onto and walks over it. Four kinds ship:

- `terrain="rough"` - smoothed value-noise bumps (robustness to uneven ground).
- `terrain="stairs"` - a flight of discrete step plateaus rising along +x
  (foot placement + climbing).
- `terrain="pyramid"` - concentric square step plateaus rising toward the centre
  from every direction (an omnidirectional climb).
- `terrain="slope"` - a constant-grade inclined ramp rising along +x
  (a continuous uphill pitch).

```python
sim.create_world(terrain="rough")        # bumpy heightfield ground
sim.add_robot("unitree_go2", keyframe="home")
```

The field spans the same +/-5 m footprint as the flat plane (the reachable
workspace is unchanged), its surface ranges from 0 up to ~8 cm on a solid
base slab (flush with `z=0` at its lowest point, so a robot never falls
below the nominal floor), and it is regenerated identically on every
`reset()` (deterministic given the terrain kind), so a benchmark that
evaluates a policy on rough ground is reproducible. `terrain` only applies
when `ground_plane=True` (the default, which is the master floor switch);
an unknown kind is rejected with an error listing the supported kinds. It
is the ground-generation primitive a terrain *curriculum* (progressive
difficulty across resets) builds on. (MuJoCo backend; the Newton backend
rejects `terrain=` as not-yet-supported.)

That curriculum knob is `difficulty`, which scales the terrain's peak
elevation (the metre height its normalized `[0, 1]` field maps to) without
changing the terrain *kind*:

```python
sim.create_world(terrain="rough", difficulty=0.3)  # gentle bumps (early stage)
# ... later, harder stages ...
sim.create_world(terrain="rough", difficulty=1.0)  # full ~8 cm bumps (default)
sim.create_world(terrain="rough", difficulty=2.0)  # exaggerated ~16 cm bumps
```

`difficulty=1.0` (the default) is the full-height terrain, byte-identical to
omitting it; `<1` is gentler, `>1` harsher. It must be a finite value `> 0`,
and it only applies with a `terrain` - setting `difficulty != 1.0` on a flat
world (no `terrain`) is rejected with an error rather than silently having no
effect. A locomotion curriculum ramps `difficulty` across resets to grow the
terrain the policy must handle.

A floating-base robot added to a terrain world (or reset in one) spawns
SEATED on the local terrain surface: its base is raised by the heightfield
height beneath its `(x, y)` so its feet rest on the ground, rather than at
the flat-ground keyframe height (which would leave them buried below a raised
heightfield). A flat ground plane and a fixed-base arm (no free joint) are
unaffected.

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

## Object size

`size` is the **full extent in meters** along each local axis - not MuJoCo's
native half-extent. It is halved when the geom is compiled, so
`size=[0.05, 0.05, 0.05]` is a 5 cm cube.

Pass every component the shape consumes; a partial vector is rejected rather
than completed from a default, because a completed vector compiles a
differently-sized object while `add_object` reports success:

| Shape | Components consumed |
|-------|---------------------|
| `box` / `ellipsoid` | `[x, y, z]` - all three full edge lengths / diameters |
| `cylinder` / `capsule` | `[diameter, unused, full height]` - three (index 1 is ignored) |
| `sphere` | `[diameter]` - one is enough |
| `plane` | `[x]` or `[x, y]` visual half-widths (`y` mirrors `x` when omitted) |
| `mesh` | none - the asset's own units define the extent |

At most 3 components are accepted; omit `size` entirely for the 5 cm default.

```python
sim.add_object("crate", shape="box", size=[0.5])
# status=error: box needs 3 'size' component(s) [x, y, z] full edge lengths,
#               got 1 (size=[0.5]). ...
sim.add_object("crate", shape="box", size=[0.5, 0.5, 0.5])   # 50 cm crate
```

## Object mass

`mass` (kg) applies to dynamic objects and must be a finite number greater than
zero - the same domain `set_body_properties(mass=...)` enforces when it writes
the same body. A mass outside it is rejected up front, naming the parameter,
instead of surfacing as a recompile failure:

```python
sim.add_object("crate", shape="box", mass=0)
# status=error: add_object: 'mass' must be a finite number > 0, got 0.0
sim.add_object("crate", shape="box", mass=1e-16)
# status=error: add_object: 'mass' must be >= MuJoCo's mjMINVAL (1e-15 kg) ...
```

This matters beyond the one object: a body's mass divides every force acting on
it, and the solver keeps a single state vector, so an infinite mass turns the
whole world's `qpos`/`qvel` to `nan` on the next step - every other body
included. `is_static=True` needs no mass (MuJoCo derives it from the geom's
density), so `mass` is ignored there.

Whatever the reason for a rejection - mass, `size`, an unsupported `shape`, an
unloadable mesh - the scene is rolled back to its previous compilable state and
the object name stays reusable, so a corrected retry under the same name works
and one bad add never bricks later scene edits.

## Mesh objects

Beyond primitives, `add_object` can inject a triangle-mesh asset (STL/OBJ) into
the live scene at runtime. Pass `shape="mesh"` with a `mesh_path` to the asset
file; the extent is defined by the mesh's own units, so `size` is ignored.

```python
sim.add_object(name="bracket", shape="mesh", mesh_path="/abs/path/bracket.stl",
               position=[0.3, 0.0, 0.1])
```

`mesh_path` is required for `shape="mesh"` - a mesh without a path is rejected
with an actionable error rather than an opaque recompile failure. If the mesh
file cannot be loaded the add is rejected and the scene is rolled back to its
previous compilable state (including the mesh asset), so the object name stays
reusable and one bad add never bricks later scene edits.

## Materials and textures

By default an object renders with a flat `color` (rgba) - a glossy, obviously
synthetic primitive. Pass `material=` to `add_object` to attach a real MuJoCo
material so the surface can be matte or carry a texture. This narrows the
sim-to-real visual gap for VLM/VLA policies trained on real footage. The
`color` (rgba) still applies and tints a textured or solid material.

```python
# Matte (non-plastic) surface: kill specular highlight + shininess.
sim.add_object("apple", shape="sphere", size=[0.04, 0, 0], color=[0.8, 0.1, 0.1, 1],
               material={"specular": 0, "shininess": 0, "reflectance": 0})

# Image texture from disk (absolute path), tiled 2x2 across the surface.
sim.add_object("table", shape="box", size=[0.5, 0.5, 0.02], is_static=True,
               material={"texture": "/abs/path/wood.png", "texrepeat": [2, 2],
                         "specular": 0, "shininess": 0})

# Procedural builtin texture (no image file needed).
sim.add_object("floor_tile", shape="box", size=[0.3, 0.3, 0.01], is_static=True,
               material={"builtin": "checker", "rgb1": [0.2, 0.3, 0.4],
                         "rgb2": [0.1, 0.2, 0.3], "texdim": 512})
```

`material` is a dict; all keys are optional:

| Key | Type | Meaning |
|-----|------|---------|
| `reflectance` / `specular` / `shininess` | float 0..1 | Surface response. `specular=0, shininess=0` = matte; the defaults read as glossy plastic. |
| `texrepeat` | `[u, v]` | Texture tiling across the surface. |
| `texture` | str | Absolute path to an image file (PNG/etc.) used as the RGB texture. |
| `builtin` | `"checker" \| "gradient" \| "flat"` | Procedural texture, coloured by `rgb1` / `rgb2` and sized `texdim` (default 512) per side. |

Specify **either** `texture` **or** `builtin`, not both. An invalid texture
path, an unknown `builtin` name, or specifying both fails loudly with a
`ValueError` (returned as a `status=error` dict through the agent tool) - there
is no silent fallback to the flat-plastic default.

Only the keys in the table above are accepted. A key outside it (a typo such as
`rgb_1`, or a field borrowed from another renderer such as `roughness`), an
empty `material={}`, or `rgb1`/`rgb2`/`texdim` without `builtin` is rejected the
same way - the alternative is an object that compiles with MuJoCo's glossy
defaults while `add_object` reports success:

```python
sim.add_object("cube", material={"builtin": "checker", "rgb_1": [1, 0, 0]})
# status=error: unknown material key(s): 'rgb_1' (did you mean 'rgb1'?).
#               Accepted keys: builtin, reflectance, rgb1, rgb2, shininess, ...
```

For natural surfaces prefer
an **image texture**; the `checker` builtin reads as a literal checkerboard.
Materials are currently supported by the MuJoCo backend; the Newton backend
rejects a non-`None` `material` rather than silently ignoring it.

## Cameras

Free cameras look from `position` toward `target` (`fov=60.0`, `width=640`, `height=480`). Robot-URDF cameras (wrist, etc.) are auto-discovered on `add_robot` - no `add_camera` needed.

To mount a camera ON a moving body (a realistic wrist/gripper view that rides with the arm), pass `parent_body`. Body names are namespaced `<robot>/<body>`; discover the exact mount point with `list_bodies` instead of guessing:

```python
bodies = sim.list_bodies(robot_name="so101")["content"][1]["json"]
mount = bodies["gripper_body"]          # e.g. "so101/gripper" -- the wrist mount
sim.add_camera(name="wrist", parent_body=mount,
               position=[0.0, 0.0, 0.05], target=[0.0, 0.0, 0.1])  # local frame
```

`list_bodies()` (no `robot_name`) lists every body in the world; with `robot_name` it scopes to that robot and also returns `gripper_body`, the best-guess end-effector mount.

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
