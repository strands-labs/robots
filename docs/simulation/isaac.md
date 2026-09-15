# Isaac Sim Backend (GPU)

The Isaac Sim backend runs the simulation on
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) (PhysX physics + RTX
path-traced rendering on the GPU; note that **physics currently solves on the
CPU** - see [PhysX runs on the CPU](#physx-runs-on-the-cpu-and-what-that-costs)). It is a **built-in, in-tree** backend that lives at
`strands_robots.simulation.isaac`, a peer of the `mujoco` and `newton` backends.
It implements the same `SimEngine` contract as the MuJoCo backend, so the
`Robot()` / `Simulation` / policy APIs are identical - only the physics and
rendering runs on the GPU through Isaac Sim.

`strands-robots` has **no hard dependency** on Isaac Sim: the `sim-isaac` extra
provides the pip-installable helpers, and `create_simulation("isaac")` resolves
the **built-in** backend, exactly like `create_simulation("mujoco")`. The Isaac
Sim runtime itself (~30 GB) is provisioned separately - via its own pip wheels
on Python 3.12, or out-of-band (see below).

## When to use it

- You have an NVIDIA RTX GPU (Ubuntu 22.04+, CUDA 12+) and want photoreal,
  path-traced observations for sim2real visuals or paper-grade frames.
- You want USD-native scenes (real CAD assets, Nucleus, IsaacLab compatibility).
- You want RTX metric depth alongside RGB, for depth-aware compositing
  (`get_frame` returns an `(H, W) float32` buffer in metres).
- You want to clone a scene into many parallel environments - see
  [Fleet replication](#fleet-replication-isaaclab-style), and note that the
  per-environment action/observation API is not implemented, so this is not yet
  fleet RL.

On macOS / Apple Silicon or CPU-only hosts, install the lightweight default
[`strands-robots`](https://github.com/strands-labs/robots) and use the MuJoCo
backend instead - it runs everywhere and the agent contract is identical. Isaac
Sim is a ~30 GB install and requires an NVIDIA GPU.

## Install

Install the Isaac Sim runtime first, then the `sim-isaac` extra:

```bash
# Step 1 - install Isaac Sim 6.0 (Python 3.12) via one of:
#   - NGC Docker (the only route verified to RENDER; see below):
#       docker pull nvcr.io/nvidia/isaac-sim:6.0.1
#   - Omniverse Launcher -> Isaac Sim 6.0, OR
#   - Isaac Lab: git clone IsaacLab && ./isaaclab.sh -i, OR
#   - pip wheels (physics only - no RTX frames; see caveats below):
#       pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url https://pypi.nvidia.com

# Step 2 - install the sim-isaac extra (helpers for the built-in backend):
pip install 'strands-robots[sim-isaac]'
```

Use a full `major.minor.patch` docker tag. NVIDIA publishes no `major.minor` tag
for this image, so `nvcr.io/nvidia/isaac-sim:6.0` and `:latest` both fail with
`no such manifest`; `6.0.1`, `6.0.0`, `5.0.0` and `4.5.0` exist. `6.0.1` is the
tag this backend is verified against.

The `sim-isaac` extra lives in **`strands-robots`** (a peer of `sim-mujoco` and
`sim-newton`). Backend discovery is lazy, so MuJoCo-only users never pay the
Isaac Sim import cost - and that laziness is why `create_simulation("isaac")`
**succeeds** with no Isaac Sim installed rather than refusing: nothing imports
the runtime until you build a world. The failure arrives at `create_world()`, as
the structured error every `SimEngine` method returns, naming each install route:

```python
sim = create_simulation("isaac")     # succeeds - resolves the in-tree backend
sim.create_world()
# {"status": "error", "content": [{"text":
#   "Isaac Sim import failed: omni.isaac.kit.SimulationApp / isaacsim.SimulationApp
#    not available. Isaac Sim must be installed first - via pip ..., Omniverse
#    Launcher, Isaac Lab ..., or Docker (nvcr.io/nvidia/isaac-sim:6.0.1)."}]}
```

To check up front instead, ask - this is the eager check, and it needs no world:

```python
from strands_robots.simulation.isaac import IsaacSimulation
ok, reason = IsaacSimulation.is_available()   # (False, "<every install route>")
```

Note the contrast with the Newton backend, which *does* raise `ImportError` from
`create_simulation("newton")` when `warp` is absent. The two differ deliberately:
Newton imports its runtime to construct, Isaac does not.

### Installing Isaac Sim via pip - caveats

> **The pip route runs physics but produces no RTX pixels.** Measured on an AWS
> `g5.2xlarge` (A10G, an RT-core GPU): a pip-wheel install boots `SimulationApp`,
> steps physics and reports success, and every RTX camera read comes back empty -
> while the **same script under `nvcr.io/nvidia/isaac-sim:6.0.1` on the same
> instance returns real frames**. That isolates it to the install route rather
> than to the GPU, the driver, or this backend: it reproduces in *pure Isaac Sim*
> with no `strands-robots` code in the process.
>
> Nothing raises, which is what makes it expensive. `render()` degrades to a blank
> frame by contract, so a rollout recording video writes an all-black MP4 and
> reports success; `get_frame()` raises, since it refuses degraded output. If you
> need camera observations, dataset video, or the hybrid compositor, **use the
> container**. Use pip only for physics, joint state and proprioceptive rollouts.

Since the cp312 wheels shipped for Isaac Sim 6.0.x, the runtime itself is
pip-installable on Python 3.12. The `extscache` extra is **required** - the
bare `isaacsim[all]` metapackage omits the `isaacsim-extscache-*` packages, and
`SimulationApp` aborts resolving its extension graph without them. The pip
install also degrades an existing dev environment in ways pip only *warns*
about, so run this exact sequence:

```bash
# 1. Install the Isaac Sim wheels (NVIDIA index required):
pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url https://pypi.nvidia.com

# 2. Repair the coverage downgrade (see below):
pip install 'coverage>=7.6.1'

# 3. Accept the EULA for non-interactive first import:
export OMNI_KIT_ACCEPT_EULA=YES
```

Known collateral (observed with isaacsim 6.0.0.1 and 6.0.1.0):

- **`coverage` downgrade breaks any numba-backed import with a red-herring
  error.** `isaacsim-kernel` pins `coverage==7.4.4`, silently downgrading modern
  coverage. numba's tracer probe then fails, and the first visible symptom is far
  from the cause: an unrelated import dies with `module 'coverage.types' has no
  attribute 'Tracer'`. Verified remedy: `pip install 'coverage>=7.6.1'` after the
  isaacsim install. The reverse pip conflict warning (`isaacsim-kernel requires
  coverage==7.4.4`) is cosmetic: coverage is test tooling for the kit, not a
  runtime dependency.
- **torch stack bump vs lerobot pins.** The isaacsim install upgrades
  `torch`/`torchvision` (and numpy/scipy/pyarrow), leaving pip conflict
  warnings against lerobot's `torchvision` pin. Expect those warnings; they do
  not by themselves indicate breakage. Validated combination as of 2026-07-31:
  isaacsim 6.0.x with torch 2.11 / torchvision 0.26.0 alongside lerobot 0.5.1 -
  GR00T-on-MuJoCo re-verified green post-install. The environment is outside
  lerobot's declared support, so re-verify your own policy path after
  installing.
- **EULA prompt on first import.** Any non-interactive first import fails with
  `Do you accept the EULA? ... EOF when reading a line` unless
  `OMNI_KIT_ACCEPT_EULA=YES` is set.
- **Exit code 134 after successful work.** Isaac Sim has a known atexit
  segfault that makes otherwise-clean scripts exit 134 *after* completing
  successfully. Scripts that boot SimulationApp should guard with
  `os._exit(...)` after SimulationApp teardown.

## Usage

```python
from strands_robots.simulation import create_simulation

# Kwargs flow into IsaacConfig. "isaac" resolves as a built-in backend.
sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)
sim.create_world()
sim.add_robot("so100")                          # resolves the same description MuJoCo loads
sim.add_object(name="cube", shape="cuboid",
               position=[0.4, 0.0, 0.05], scale=[0.05, 0.05, 0.05])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.0, 0.0, 0.1])
sim.reset()                                      # see "Adding a dynamic body" below
sim.step(120)
frame = sim.render(camera_name="front")          # RGB + depth
sim.destroy()
```

`Robot("so100", backend="isaac", ...)` routes through the same factory, so the
backend selection is identical whether you go through `Robot()` or
`create_simulation()`.

`scale=` above is an accepted alias for `add_object(size=...)`, and it is the only
extra keyword that method reads. Any other keyword is refused by name rather than
dropped -- the same contract `IsaacConfig` applies to `create_simulation` kwargs,
and the same verdict the MuJoCo and Newton backends give (they declare the same
`add_object` parameters and no `**kwargs`, so an unknown keyword is a `TypeError`
there):

```python
sim.add_object(name="cube", heigth=0.3)
# {"status": "error", "content": [{"text":
#   "Unknown parameter(s) ['heigth'] for action 'add_object'. Valid: [...]"}]}
```

## Configuration (`IsaacConfig`)

Keyword arguments to `create_simulation("isaac", ...)` (or
`Robot(..., backend="isaac", ...)`) construct an `IsaacConfig`. Unknown keys are
rejected eagerly. The commonly used fields:

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `num_envs` | `int` | `1` | Default environment count for `replicate()`, which is what actually clones them. Setting it alone creates nothing. A positive integer - the same domain `replicate(num_envs=...)` takes. |
| `device` | `str` | `"cuda:0"` | CUDA device (`cuda:N`). Must be a CUDA device. **Selects the CUDA device for RTX rendering; PhysX itself currently solves on the CPU** - see below. |
| `headless` | `bool` | `True` | Run without a GUI (required for cloud/CI). |
| `physics_dt` | `float` | `1/120` | Physics timestep (seconds). Positive and finite - the domain `create_world()` applies to the effective dt, and the one the legacy `IsaacSimulation(default_timestep=...)` shortcut that writes this field takes as well. |
| `rendering_dt` | `float` | `1/30` | Rendering timestep (seconds). |
| `render_mode` | `str` | `"headless"` | `"headless"`, `"rtx_realtime"` (raster), or `"rtx_pathtracing"` (photoreal). |
| `gravity` | `tuple` | `(0, 0, -9.81)` | Gravity vector (Z-up). Three finite components, Z-aligned - the same domain `create_world(gravity=...)` takes. |
| `ground_plane` | `bool` | `True` | Add a ground plane on `create_world()`. |
| `stage_path` | `str` | `"/World"` | USD prim-path prefix every created prim is addressed under. Must be absolute, with at least one component, every component a prim name (`[A-Za-z_][A-Za-z0-9_]*`). |
| `nucleus_url` | `str \| None` | `None` | Override Omniverse Nucleus URL (env-resolvable). |
| `camera_width` / `camera_height` | `int` | `640` / `480` | Default camera resolution, for every `add_camera` / render call that states none of its own. Positive integers - the same pixel floor those `width` / `height` arguments take. |
| `verbose` | `bool` | `False` | Verbose Isaac Sim / Kit logging. |

### Environment variables

The Isaac backend reads three `STRANDS_ISAAC_*` variables (resolved when
`IsaacConfig` is constructed). `STRANDS_ISAAC_NUCLEUS_URL` is read only when
`nucleus_url` is not passed, so there the kwarg wins; the two switches override
their field whenever they are set. Which of those two directions the switches
*should* have is [#2062](https://github.com/strands-labs/robots/issues/2062).

Both switches accept four symmetric pairs, case-insensitively and ignoring
surrounding whitespace:

| on | off |
|----|-----|
| `1` | `0` |
| `true` | `false` |
| `yes` | `no` |
| `on` | `off` |

Unset -- or set to an empty value, which is what an undefined `${{ vars.* }}`
interpolation in a GitHub Actions `env:` block produces -- leaves the field
alone. Any other spelling raises `ValueError` naming both vocabularies, rather
than falling through to the off side: `STRANDS_ISAAC_HEADLESS=enabled` used to
open a window.

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_ISAAC_NUCLEUS_URL` | Override the Omniverse Nucleus server URL when `nucleus_url` is not passed | unset (Isaac defaults) |
| `STRANDS_ISAAC_HEADLESS` | On forces `headless`; off forces windowed | unset (uses `headless` kwarg) |
| `STRANDS_ISAAC_RTX_PATHTRACING` | On forces `render_mode="rtx_pathtracing"`; off leaves `render_mode` alone | unset |

### PhysX runs on the CPU, and what that costs

Rendering is on the GPU. **Physics is not.** `create_world` builds `World` without
passing `device`, and `World`'s own default resolves to `"cpu"`, so PhysX solves on
the CPU regardless of what `device` says.

`get_state()` reports both, and on current hardware they disagree - which is the
honest signal:

```python
sim.get_state()["content"][0]["json"]
# {..., "device": "cpu", "device_requested": "cuda:0", ...}
```

Read `device` when you want to know where physics is actually running.
`device_requested` is what the config asked for.

The cost is real: measured on an AWS `g5.2xlarge` (A10G) under Isaac Sim 6.0.1 -
40 cuboids, 300 `step()` calls after a 30-step warmup - **9.9 steps/s on the CPU
against 112.3 steps/s** with the GPU pipeline enabled.

It is not enabled because enabling it breaks `add_robot`:

```
device="cuda:0"  ->  gpu_pipeline=True  ->  add_robot() fails with
    CUDA error: an illegal memory access was encountered
    (omni.physx.tensors GpuArticulationView.cpp:631)
```

PhysX's GPU pipeline **pre-sizes its tensor buffers at `world.reset()`**, and
`create_world` resets immediately - sizing them for a stage with a ground plane and
*zero* articulations. The first `add_robot` initializes an articulation into a view
with no room for it, and the illegal access poisons the CUDA context, so the next
`add_object` fails too. Adding while the sim is stopped fails differently
(`'NoneType' object has no attribute 'create_articulation'`).

Using the GPU pipeline requires Isaac Lab's pattern - build the entire scene, then
reset once - which is incompatible with this backend's incremental contract, where
an agent calls `create_world` and then `add_robot` one tool call at a time. If your
workload is physics-throughput-bound and your scene is known up front, the MuJoCo
backend is currently faster for that shape; use this backend for RTX observations
and USD scenes.

### Adding a dynamic body needs a `reset()` before the next `step()`

`add_object` with `is_static=False` (the default) adds a body PhysX's tensor
simulation view does not cover, and that view is what every articulation read goes
through. Measured on an A10G, adding a dynamic cuboid takes a Franka's
`get_observation` from 9 joint keys to **0**; removing one is worse - PhysX logs
`the physics.tensors simulationView was invalidated` and the next joint read *hangs*.

So `step()` refuses until a `reset()` rebuilds the view:

```python
sim.add_object(name="cube", shape="cuboid", position=[0.4, 0.0, 0.5])
sim.step(60)     # {"status": "error", ...} - the view does not cover the cube
sim.reset()      # rebuilds it
sim.step(60)     # runs
```

Note `reset()` also returns robots to their default pose, so build the scene and
reset **before** posing anything. An in-place rebuild that avoids the teleport was
tried and does not work reliably: `SimulationManager.initialize_physics()` +
`world.play()` restores the view only when the timeline was stopped, and reports
success without restoring it when the timeline is already playing.

A **static** body needs none of this - `add_object(is_static=True)` and removing a
static body both leave the view intact (measured: 9 joint keys → 9), as do
`add_camera`, `move_object`, `add_robot` and `remove_robot`.

## Capabilities and parity

`IsaacSimulation` exposes the same `SimEngine` shape as the MuJoCo backend:

- **World & lifecycle** - `create_world`, `destroy`, `reset`, `step`,
  `get_state`, `cleanup`.
- **Robots** - `add_robot`, `remove_robot`, `list_robots`, `robot_joint_names`,
  `send_action`, `get_observation`. `add_robot` takes an explicit `usd_path=`,
  `urdf_path=` or `mjcf_path=`, and with none of them resolves the robot name (or
  `data_config=`) through the **same** registry resolver the MuJoCo backend uses,
  so both backends load one description file for one name. An MJCF is converted
  to USD once via Isaac's own `isaacsim.asset.importer.mjcf` extension and cached
  under `$STRANDS_BASE_DIR/asset_cache/usd_robots/`, content-addressed over the
  description *and every file its directory holds* (an MJCF `<include>`s bodies
  and references a `meshdir`, so keying on the named file alone would serve a
  stale conversion). The cached USD is then referenced by the native USD path, so
  a name, an MJCF, a URDF and a USD all converge on one loader.

  A robot the registry cannot resolve is refused with the same three-way
  diagnosis the MuJoCo backend gives - a typo (with close matches), a
  hardware-only registry entry, or an asset that is simply not downloaded - since
  both backends now share that message rather than each spelling its own.

  There was previously a fourth route here: a "procedural builder" for `so100`,
  `panda` and `unitree_g1` that needed no asset files. It has been deleted. It
  reported success while creating no prims at all, left the articulation
  unwired and `get_observation()` empty for the whole lifecycle, and reported
  joint names that disagreed with this backend's own parity claim - `so100` as
  `shoulder_pan`/`shoulder_lift`/... against MuJoCo's `Rotation`/`Pitch`/..., and
  `panda` as 7 joints against MuJoCo's 9.

  `add_robot(..., fix_base=False)` gives a URDF robot a **floating base**, which
  is what a humanoid or quadruped needs; the default `True` welds the root, as
  every URDF import here used to do unconditionally. It is a parameter rather
  than something read from the file because URDF cannot answer it - the universal
  convention for a mobile robot is a root link with no parent joint, which is
  byte-identical to how a bolted-down arm declares its base - which is why
  Isaac's own importer takes the flag and why MuJoCo and Newton, reading MJCF's
  `<freejoint>`, have no equivalent parameter. It applies to `urdf_path` only: a
  USD asset carries its own articulation root, so `fix_base=False` there is
  refused rather than ignored.

  A floating-base robot reports the four `base_*` observation entries the
  `SimEngine.get_observation` schema requires (`base_pos`, `base_quat`,
  `base_lin_vel`, `base_ang_vel`), matching MuJoCo and Newton; a fixed-base arm
  reports none of them, as the schema specifies. Note that a `reset()` does not
  preserve a floating base's spawn height - see `add_robot`'s docstring for the
  measurement and the reason.
- **Objects** - `add_object` (`cuboid` / `sphere` / `cylinder` / `capsule` /
  `mesh`, dynamic or static), `remove_object`. A `shape="mesh"` add takes a
  `mesh_path` to an STL/OBJ/MSH asset (converted to USD once and cached under
  `$STRANDS_BASE_DIR/asset_cache/usd_meshes/`, content-addressed) or to a
  USD file (referenced directly). The asset defines the extent - `size` is
  ignored for a mesh, the MuJoCo read of that parameter (the Newton backend
  consumes it as a scale instead; see
  [#2300](https://github.com/strands-labs/robots/issues/2300)) - and
  collision uses the mesh's **convex hull**, also the MuJoCo contract, with
  the same caveat for concave assets: the hull fills every cavity. A missing
  file, an unconvertible format, or an asset declaring a vertex coordinate
  that is not finite is refused up front, never realized as a
  fallback primitive.
- **Cameras & rendering** - `add_camera` (look-at, FOV), `render` (RGB + depth).
  World-fixed only: `parent_body` (a body-mounted wrist camera, supported on
  mujoco/newton) is refused here with an error naming those backends, because
  camera prims are parented to the stage camera scope rather than to an
  articulation link.
- **Loaders** - `load_urdf` / `load_mjcf` / `load_usd` resolve to a
  `ProceduralRobot` dataclass. These are the **description-introspection** API -
  parse a robot file into a joints/bodies report, for tooling and for the
  cross-backend parity tests - not the load path: `add_robot` builds
  articulations through Isaac's own importers and does not call them. Both XML
  loaders report each link's pose in its parent's frame. `load_mjcf` reads the rotation from whichever of MJCF's five
  spellings the body uses - `quat`, `euler`, `axisangle`, `xyaxes` or `zaxis` -
  under the model-global `<compiler angle>` and `<compiler eulerseq>`. The
  reported orientation is always a unit quaternion, the one MuJoCo's compiler
  stores: a non-unit spelling such as `quat="1 -1 0 0"` (the idiomatic quarter
  turn) is reported as the quarter turn it means, not as the components as
  written, which applied as a rotation would scale the frame by `|q|^2` as well
  as turning it. `load_urdf` reads both halves from the `<origin>` of the joint
  that reaches the link, since URDF places a link on that joint rather than on
  the `<link>` element: `xyz` into `position` and `rpy` - fixed-axis
  roll-pitch-yaw, always radians - into `orientation`. A root link, reached by no
  joint, keeps the identity pose. A joint's axis comes from `<axis xyz>`, and
  each format's own default applies when the element states no vector the
  parser can read: a URDF joint that omits the optional `<axis>` acts about
  **+X**, an MJCF `<joint>` that omits `axis` about **+Z**. Both are valid
  axes, so a joint read under the other format's default would be reported
  acting in the perpendicular plane with the load still reporting success. A
  default applies only where the format declares one, which is why the two
  formats answer an omitted `type` differently: MJCF documents `hinge` as the
  default for a `<joint>`, so an MJCF joint with no `type` is read as a hinge,
  while URDF requires `type` on every `<joint>`, so a URDF joint that omits it
  is refused by name - by both readers, `load_urdf` and `urdf_joint_names`.
  Reading it as `fixed` welded a joint the file never described and returned a
  robot with fewer actuated DOFs than the file declares, indistinguishable from
  a deliberate `type="fixed"`. Both
  of MJCF's spellings of a free joint are read - the dedicated `<freejoint>`
  element and `<joint type="free">`, which MuJoCo compiles to the same joint -
  so a floating base is reported rather than absent. `<freejoint>` is how every
  shipped quadruped and humanoid states its base, and it resolves no default
  class, because MJCF has no `<default><freejoint>` block: a `<default><joint>`
  class reaches the `type="free"` spelling only, exactly as MuJoCo applies it.
  Either spelling is reported with `joint_type="fixed"`, since `JointDef` has no
  6-DOF spelling, so a floating base is visible in `joints` without being
  counted as an actuated DOF by `num_joints`.

Because the joint-name and observation contract matches the MuJoCo backend,
policies and observation mappings transfer unchanged between backends. That holds
for a shared reason rather than by coincidence: for a robot named rather than
pathed, both backends resolve the same description file through the same
resolver, so the joint names are read from one asset. Measured on Isaac Sim
6.0.1, `panda` reports `joint1`..`joint7` plus `finger_joint1`/`finger_joint2`
(9 of 9 matching MuJoCo) and `so100` reports `Rotation`, `Pitch`, `Elbow`,
`Wrist_Pitch`, `Wrist_Roll`, `Jaw` (6 of 6). Passing an explicit `usd_path=` opts
out of that guarantee, since nothing then ties your asset to the one MuJoCo would
have loaded.

Mesh-bearing scenes get the same treatment for their *visuals*: `load_scene`
renders each scene object with its real mesh (bowls, plates - the assets a
pixel-conditioned policy was trained on) while keeping the validated
collision-AABB box as the invisible physics proxy, so switching backends does
not also switch what the cameras see. That box covers both MJCF spellings of a
capsule or cylinder - `pos` plus `size="radius half-length"`, and `fromto` plus
`size="radius"`, where the two endpoints carry the placement and the axis
extent - so a `fromto` bar is proxied by its full length at its midpoint rather
than by a ball of its radius at the body origin. An object whose mesh cannot be resolved
keeps a visible box proxy, and the `load_scene` report then carries an
explicit caveat that pixel-conditioned policy scores on that scene are not
comparable across backends; when every object renders its mesh, the caveat
disappears. A mesh asset that is declared but missing on disk fails the scene
load loudly - never a silent box - and so does one declaring a vertex
coordinate that is not finite: `min`/`max` order a NaN as neither smaller nor
larger than anything, so the collision AABB measured from such an asset is the
box of the vertices that *are* finite, numerically indistinguishable from a
mesh that declared only those, while an infinite coordinate makes the proxy
unbounded. MuJoCo refuses the same asset (`vertex coordinate N is not
finite`), so the scene loader does too rather than sizing a proxy around it.

The accepted *input* domain matches too, so a call one backend refuses is
refused by all three. For the setup methods that means the pose vectors, an
object's `color` and `mass`, the camera `fov` and the pixel dimensions - and the
entity `name`: `add_robot`,
`add_object` and `add_camera` each require a non-empty string containing no NUL.
That matters more here than on MuJoCo because the name is interpolated into the
USD prim path (`{stage_path}/Robots/{name}`), so an unaddressable name does not
just produce an entity you cannot look up - `add_robot("")` resolved to
`/World/Robots/`, the *container* scope for every robot, and `remove_robot`
prunes its cleanup registry by that prefix. Unlike the MuJoCo backend there is
no "derive a label from the model" short form: `name` is also the key the registry
resolution falls back to when no `data_config=` is given, so `None` / `""` are
refused rather than replaced with a generated label.

`stage_path` is the other half of that same path and carries the same floor, so
a path this backend records is one it can address whichever component the caller
got wrong. It must be an absolute USD prim path with at least one component,
every component a prim name (`[A-Za-z_][A-Za-z0-9_]*`). A non-`str` was
previously interpolated as its rendered text (`stage_path=None` recorded
`None/Robots/arm`); a relative prefix (`"World"`) recorded a path that
`get_body_state` cannot take, because it distinguishes an absolute prim path
from a `<robot>/<link>` pair by the leading `/`; a trailing or doubled separator
(`"/World/"`) left an empty component; and a component outside USD's identifier
alphabet (`"/My World"`) is transcoded by USD, so the prim does not land at the
path recorded for it. The identifier rule applies to the prefix only - `name` is
shared with backends whose entity names are not USD identifiers.

The one deliberate difference in that list is `mass=0`. The Newton backend
documents it as an alternative spelling of `is_static=True` and honours it, so it
stays accepted there; this backend documents no such spelling, so a zero mass is
refused with `is_static=True` named as the remedy - the MuJoCo contract these
docs otherwise mirror. A static object's mass is read by nobody on any backend,
so it is not validated there.

Looking an entity *up* is the other half of that contract, and it answers rather
than refuses: a name only *addresses* an entity here, so a name that cannot be a
registry key is honestly absent. `remove_robot`, `remove_object`,
`remove_camera`, `send_action`, `move_object`, `get_body_state` and the rest
report it with the unknown-entity message they already had, `robot_joint_names`
and `get_observation` keep answering empty, and `get_frame` /
`get_camera_params` raise the `KeyError` their contract names. Previously the
membership test itself raised `TypeError: unhashable type` for a list or dict
name, so the miss escaped the envelope those methods document as their only
failure channel - reachable with no entities registered at all.

`render` is the one lookup that cannot answer with a frame, so it reports the
same verdict as its raw sibling. Given a `camera_name` that *names* a camera the
scene does not carry it returns `{"status": "error"}` with
`Camera '<name>' not found. Available: [...]` - the message `get_frame` raises
for the identical name, and the one the MuJoCo and Newton `render` give. It used
to report `status="success"` with an all-black frame instead, tagged
`Rendered (no camera)`, plus `pixel_mean` `0.0` as a measurement and the missing
name in the `camera` field; because that envelope carries the PNG block the
shared frame extractor reads, a rollout recording a mistyped camera wrote an
all-black video and reported success. A `camera_name` that names *no* camera -
`None`, `""`, `"default"` (the signature default) or `"free"` - still gets that
blank frame: Isaac has no free camera to fall back to, unlike the two backends
whose render entry points resolve those tokens to one, so for them it is a
degradation rather than a mistake. Registering a camera under one of those names
is accepted here and renders normally, since nothing on this backend routes them.

## Fleet replication (IsaacLab-style)

`replicate()` clones the scene you have built into a grid of parallel
environments using Isaac Sim's own `isaacsim.core.cloner.GridCloner`. The scene
already on the stage is environment 0, so `num_envs` counts it - `replicate(64)`
produces the source plus 63 clones under `{stage_path}/envs/env_1 .. env_63`:

```python
sim = create_simulation("isaac", num_envs=64, headless=True,
                        render_mode="headless")
sim.create_world()
sim.add_robot(name="panda", usd_path="/path/to/franka.usda")

result = sim.replicate(64, spacing=1.5)   # or replicate() to use config.num_envs
# {"status": "success", ... "Cloned the scene into 64 environments (63 clones of
#  1 source prim(s) plus the source as env_0, 1197 prims) in 1840ms at 1.50m
#  spacing on cuda:0. NOTE: get_observation/send_action address env_0 only ..."}

sim.destroy()
```

Physics is replicated across every environment and inter-environment collisions
are filtered, so the clones do not push each other around. The result payload
reports what was actually built - `clones_created`, `prims_created`,
`build_time_ms`, `physics_replicated`, `collisions_filtered` - rather than echoing
the count you passed in.

**What is not implemented: a per-environment observation or action API.**
`get_observation` and `send_action` address environment 0's robot, which is the
only one carrying an `Articulation` handle. The clones advance under physics and
are what a renderer and a domain-randomisation pass see, but they cannot be driven
or read individually; that needs an articulation view across environments, which
this backend does not build. The success message says so on every call, so a
`num_envs: 64` in the payload is not mistaken for 64 drivable robots.

Two boundaries worth knowing. `replicate(1)` is an accepted no-op - the scene is
already one environment - and deliberately does *not* mark the simulation
replicated, so `add_robot` keeps working; `replicate(n)` for `n > 1` does mark it,
and `add_robot` is refused from then on because the clones were built from the
scene as it stood. And `num_envs` / `spacing` are validated on the shared numeric
domains, so a negative count or a NaN spacing is refused rather than reported as a
fleet.

Until recently this method was a stub: it reported
`"Replicated to N environments. Build time: 0ms."` while calling no cloner at all,
left the stage untouched, made `get_state()` report the fabricated count, and set
the flag that refuses `add_robot` - so the no-op also locked you out of the scene.

## Where to go next

The Isaac backend was originally prototyped in the `strands-robots-sim`
project, which still hosts a MkDocs site with additional architecture notes and
troubleshooting. It is kept here as background reference; the backend itself now
ships in-tree in `strands-robots`:

- Background docs: <https://strands-labs.github.io/robots-sim/>
- Backend reference: <https://strands-labs.github.io/robots-sim/backends/isaac/>
- Source (historical): <https://github.com/strands-labs/robots-sim>
