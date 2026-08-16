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
so a keyframe spawn is sticky across episodes. An
unknown keyframe name/index
is an error that lists the model's available keyframes. `keyframe=None` (the
default) keeps the zero-pose spawn. (MuJoCo backend; the Newton backend rejects
`keyframe=` as not-yet-supported.)

### Adding a robot does not disturb the scene it joins

Only the robot being added is placed at a defined configuration - its keyframe,
or the zero pose. Everything already in the world is left exactly as it was: an
arm keeps the pose it is in (whether that is its keyframe pose or wherever a
policy or `send_action` has driven it) *and* the actuator setpoints holding it
there, objects stay where they settled or were carried to, latched `apply_force`
wrenches persist, and the clock keeps counting.

So a scene can be composed in any order, and a robot can be added mid-session
without invalidating what has already happened in it:

```python
sim.run_policy(robot_name="panda", ...)   # arm ends up somewhere useful
sim.add_robot(name="helper", data_config="so101", position=[0.0, -0.6, 0.0])
# 'panda' is still where the rollout left it; 'helper' starts at its zero pose
```

To return the *whole* world to its initial state - every robot, every object and
the clock - call `reset()`, which is what that method is for.

## Declared physics options

A robot MJCF may declare the solver settings its contacts and actuators were
tuned for. `add_robot` carries them onto the scene, because MuJoCo's `<option>`
is model-global and does not survive the spec attach:

```python
sim.create_world()
sim.add_robot(name="panda")          # model declares integrator="implicitfast"
sim.mj_model.opt.integrator          # -> mjINT_IMPLICITFAST
```

This matters for manipulation. Under the default Euler integrator a Panda's
position servos diverge enough that a top-down grasp pushes the object away and
squeezes through it; `so100`, `so101`, `aloha`, `shadow_hand` and `robotiq_2f85`
likewise declare `cone="elliptic" impratio="10"` so their grippers can hold load.

Precedence, highest first:

| Source | Wins for |
| --- | --- |
| `create_world(timestep=, gravity=)` | `timestep`, `gravity` - always |
| Your own scene MJCF (`replace_scene_mjcf`) | any field it sets |
| First robot attached that declares the field | everything else |

A model-global field holds one value, so if a second robot declares a different
value for a field already set, the existing value is kept and the discarded
request is logged with the field, both values and the robot name. Add that robot
first, or declare the value in your own scene MJCF, to make it win.

Vector environment fields (`wind`, `magnetic`, contact overrides) and the flag
bitfields describe the world rather than the robot and are never adopted.

Adoption is committed only once the robot is actually in the scene, so an
`add_robot` that reports an error leaves the world's solver settings exactly as
they were - and leaves the field free for the next robot that declares it.

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
omitting it; `<1` is gentler, `>1` harsher. It must be a finite *number*
`> 0` - the same positive-real domain every other continuous knob accepts, so
`0`, a negative value, `nan`/`inf`, a `bool` (`True` is not a scale, even
though it is an `int` subclass) and a string (including a numeric one like
`"0.5"`) are all refused with a structured error naming the parameter. Every
backend reports through that one domain, so a scale one `create_world` refuses
cannot be honored by another. It only applies with a `terrain` - setting
`difficulty != 1.0` on a flat world (no `terrain`) is rejected with an error
rather than silently having no effect. A locomotion curriculum ramps `difficulty` across resets to grow the
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

Every component must also be a finite number, and **that** part of the domain is
shared with the Newton and Isaac backends' `add_object` - word for word, not just
verdict for verdict - so an extent one backend refuses is refused by all three
with the same message. A `nan`/`inf`, boolean, `None` or otherwise non-numeric
component is rejected by name rather than reaching the solver, a NumPy array is
accepted and normalized to plain floats, and a value that is not a vector at all
is refused instead of raising from whatever first tries to iterate it:

```python
sim.add_object("crate", shape="box", size=[float("nan"), 0.1, 0.1])
# status=error: add_object: 'size' must contain finite numbers (no nan/inf),
#               got [nan, 0.1, 0.1]
sim.add_object("crate", shape="box", size=0.5)
# status=error: add_object: 'size' must be a list/tuple of numbers, got 0.5
sim.add_object("crate", shape="box", size=np.array([0.5, 0.5, 0.5]))   # accepted
```

An **empty** `size` is a component count, not an omission, so it is rejected
rather than quietly taking the default extent - omit `size` (or pass `None`) to
ask for the default. Here the three backends agree on the verdict but not on the
wording, because MuJoCo reaches an empty vector through the per-shape count above
and so names the count the shape needs:

```python
sim.add_object("crate", shape="box", size=[])
# status=error: box needs 3 'size' component(s) [x, y, z] full edge lengths,
#               got 0 (size=[]). ...
```

The per-shape counts in the table above remain MuJoCo's alone. Newton and Isaac
accept a short `size` (Isaac documents completing the missing trailing components
from defaults), and neither bounds a component to be positive, so a vector this
backend refuses on either of those axes may still be accepted there. Converging
the three is tracked in
[#1858](https://github.com/strands-labs/robots/issues/1858).

## Object mass

`mass` (kg) applies to dynamic objects and must be a finite number greater than
zero - the same domain `set_body_properties(mass=...)` enforces when it writes
the same body, and the same one the Newton and Isaac backends' `add_object`
applies, so a mass one backend refuses is refused by all three. A mass outside it
is rejected up front, naming the parameter, instead of surfacing as a recompile
failure:

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
density), so `mass` is ignored there - and not validated, on any backend, since
nothing reads it. The Newton backend additionally documents `mass=0` as an
alternative spelling of `is_static=True` and keeps accepting it; MuJoCo and Isaac
refuse a zero mass and name that flag as the remedy.

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

## Surgical MJCF edits

`patch_scene_mjcf(ops)` applies a list of structured ops to the live spec and
recompiles once, preserving joint state for untouched joints. Each op accepts
only the keys it reads:

| Op | Keys |
|----|------|
| `add_body` | `parent` (default `"world"`), `name` (required), `pos`, `quat` |
| `add_geom` | `body` (required), `type` (default `"box"`), `size`, `rgba`, `name`, `pos`, `quat` |
| `add_site` | `body` (default `"world"`), `name` (required), `pos`, `size`, `rgba` |
| `set_body_pos` | `name` (required), `pos` |
| `set_body_quat` | `name` (required), `quat` |
| `delete_body` | `name` (required) |

Any other key is rejected. Every field above has a fallback default (`pos` the
origin, `quat` identity, `type` `"box"`, `parent` the worldbody), so a key the op
does not read is not inert - it would leave that default in place while the patch
reports success:

```python
sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "position": [0.4, 0, 0.9]}])
# status=error: set_body_pos: unknown op key(s): 'position' (did you mean 'pos'?).
#               Accepted keys: name, op, pos.
```

Every numeric field an op writes is held to the domain the scene-construction
calls apply to the same buffer:

| field | accepted |
| --- | --- |
| `pos` | exactly 3 finite components |
| `quat` | exactly 4 finite components |
| `rgba` | 3 (RGB, completed with an opaque alpha) or 4 finite components |
| `size` | finite components, in the count the geom's shape consumes |

MuJoCo bakes a `nan`/`inf` component into the model without complaint, so an
unchecked one reports success and only surfaces later as a poisoned physics
state. A wrong component count is reported by the library rather than left to
MuJoCo, which for the two attribute-assigning ops (`set_body_pos`,
`set_body_quat`) dumps a C++ overload table naming neither the op nor the field:

```python
sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": [float("nan"), 0, 0.3]}])
# status=error: set_body_pos: 'pos' must contain finite numbers (no nan/inf),
#               got [nan, 0, 0.3]

sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "pos": [0.4, 0.9]}])
# status=error: set_body_pos: 'pos' must be a 3-element vector, got 2 ([0.4, 0.9])
```

A three-component `rgba` is the same RGB `add_object(color=...)` accepts, so the
two surfaces that write `geom_rgba` agree on what a colour is:

```python
sim.patch_scene_mjcf([{"op": "add_geom", "body": "rig", "type": "box",
                       "size": [0.1, 0.1, 0.1], "rgba": [0.9, 0.3, 0.1]}])
# status=success - stored as [0.9, 0.3, 0.1, 1.0]
```

The batch is atomic: if any op is rejected the world is rolled back to its
pre-patch state, so a bad key or a non-finite component never leaves a
half-applied scene. Use
`replace_scene_mjcf(xml)` for MJCF elements this vocabulary does not cover.

## Cameras

Free cameras look from `position` toward `target` (`fov=60.0`, `width=640`, `height=480`). Robot-URDF cameras (wrist, etc.) are auto-discovered on `add_robot` - no `add_camera` needed.

A discovered camera is registered under its short MJCF name (`wrist`), and the
compiled model also carries it namespaced (`so101/wrist`); `render` takes the
namespaced form, `get_observation` keys on the short one. The short name is
first-come across robots: when a second robot declares a camera whose short name
is already taken, that camera is registered under its namespaced name instead
(logged, naming both), so two arms that both declare `wrist` give you `wrist` and
`arm2/wrist` rather than one of them shadowing the other. Each camera belongs to
exactly one robot, which is what makes `remove_robot` take that robot's cameras
with it and leave every other robot's alone.

To mount a camera ON a moving body (a realistic wrist/gripper view that rides with the arm), pass `parent_body`. Body names are namespaced `<robot>/<body>`; discover the exact mount point with `list_bodies` instead of guessing:

```python
bodies = sim.list_bodies(robot_name="so101")["content"][1]["json"]
mount = bodies["gripper_body"]          # e.g. "so101/gripper" -- the wrist mount
sim.add_camera(name="wrist", parent_body=mount,
               position=[0.0, 0.0, 0.05], target=[0.0, 0.0, 0.1])  # local frame
```

`list_bodies()` (no `robot_name`) lists every body in the world; with `robot_name` it scopes to that robot and also returns `gripper_body`, the best-guess end-effector mount.

A mounted camera survives `remove_robot`, which rebuilds the whole scene: it is
re-mounted on its body once every surviving robot is re-attached, keeping its
local pose and its tracking. Removing the robot the camera is mounted ON leaves
it with no mount point, so that camera is dropped (with a warning naming it)
rather than blocking the removal.

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
