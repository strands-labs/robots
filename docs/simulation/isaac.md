# Isaac Sim Backend (GPU)

The Isaac Sim backend runs the simulation on
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) (PhysX GPU physics +
RTX path-traced rendering). It is a **built-in, in-tree** backend that lives at
`strands_robots.simulation.isaac`, a peer of the `mujoco` and `newton` backends.
It implements the same `SimEngine` contract as the MuJoCo backend, so the
`Robot()` / `Simulation` / policy APIs are identical - only the physics and
rendering run on the GPU through Isaac Sim.

`strands-robots` has **no hard dependency** on Isaac Sim: the `sim-isaac` extra
provides the pip-installable helpers, and `create_simulation("isaac")` resolves
the **built-in** backend, exactly like `create_simulation("mujoco")`. The Isaac
Sim runtime itself (~30 GB) is provisioned separately - via its own pip wheels
on Python 3.12, or out-of-band (see below).

## When to use it

- You have an NVIDIA RTX GPU (Ubuntu 22.04+, CUDA 12+) and want photoreal,
  path-traced observations for sim2real visuals or paper-grade frames.
- You want USD-native scenes (real CAD assets, Nucleus, IsaacLab compatibility).
- You want Replicator synthetic data - ground-truth depth, segmentation, and
  bounding boxes alongside RGB.
- You want fleet RL on PhysX GPU with 1024+ parallel environments.

On macOS / Apple Silicon or CPU-only hosts, install the lightweight default
[`strands-robots`](https://github.com/strands-labs/robots) and use the MuJoCo
backend instead - it runs everywhere and the agent contract is identical. Isaac
Sim is a ~30 GB install and requires an NVIDIA GPU.

## Install

Install the Isaac Sim runtime first, then the `sim-isaac` extra:

```bash
# Step 1 - install Isaac Sim 6.0 (Python 3.12) via one of:
#   - pip wheels (see caveats below):
#       pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url https://pypi.nvidia.com
#   - Omniverse Launcher -> Isaac Sim 6.0, OR
#   - Isaac Lab: git clone IsaacLab && ./isaaclab.sh -i, OR
#   - NGC Docker: docker pull nvcr.io/nvidia/isaac-sim:6.0

# Step 2 - install the sim-isaac extra (helpers for the built-in backend):
pip install 'strands-robots[sim-isaac]'
```

The `sim-isaac` extra lives in **`strands-robots`** (a peer of `sim-mujoco` and
`sim-newton`). Requesting `create_simulation("isaac")` without the extra
installed raises a `ValueError` whose message carries the exact install hint
(`pip install 'strands-robots[sim-isaac]'`). Backend discovery is lazy, so
MuJoCo-only users never pay the Isaac Sim import cost.

### Installing Isaac Sim via pip - caveats

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

- **`coverage` downgrade breaks robosuite/LIBERO with a red-herring error.**
  `isaacsim-kernel` pins `coverage==7.4.4`, silently downgrading modern
  coverage. numba's tracer probe then fails, and the first visible symptom is
  far from the cause - robosuite's OSC controller import dies inside the LIBERO
  adapter with `module 'coverage.types' has no attribute 'Tracer'`. Verified
  remedy: `pip install 'coverage>=7.6.1'` after the isaacsim install. The reverse
  pip conflict warning (`isaacsim-kernel requires coverage==7.4.4`) is cosmetic:
  coverage is test tooling for the kit, not a runtime dependency.
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
  successfully. The drivers in this repo guard with `os._exit(...)` after
  SimulationApp teardown (see `examples/libero/run_isaac.py`); user scripts
  that boot SimulationApp should do the same.

## Usage

```python
from strands_robots.simulation import create_simulation

# Kwargs flow into IsaacConfig. "isaac" resolves as a built-in backend.
sim = create_simulation("isaac", render_mode="rtx_realtime", headless=True)
sim.create_world()
sim.add_robot("so100")                          # procedural; no asset files needed
sim.add_object(name="cube", shape="cuboid",
               position=[0.4, 0.0, 0.05], scale=[0.05, 0.05, 0.05])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.0, 0.0, 0.1])
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
| `num_envs` | `int` | `1` | Parallel environments. Set to `1024`+ for fleet RL. |
| `device` | `str` | `"cuda:0"` | CUDA device (`cuda:N`). Must be a CUDA device. |
| `headless` | `bool` | `True` | Run without a GUI (required for cloud/CI). |
| `physics_dt` | `float` | `1/120` | Physics timestep (seconds). |
| `rendering_dt` | `float` | `1/30` | Rendering timestep (seconds). |
| `render_mode` | `str` | `"headless"` | `"headless"`, `"rtx_realtime"` (raster), or `"rtx_pathtracing"` (photoreal). |
| `gravity` | `tuple` | `(0, 0, -9.81)` | Gravity vector (Z-up). |
| `ground_plane` | `bool` | `True` | Add a ground plane on `create_world()`. |
| `stage_path` | `str` | `"/World"` | USD stage path prefix. |
| `nucleus_url` | `str \| None` | `None` | Override Omniverse Nucleus URL (env-resolvable). |
| `camera_width` / `camera_height` | `int` | `640` / `480` | Default camera resolution. |
| `enable_rtx_sensors` | `bool` | `True` | Enable RTX-accelerated camera / LiDAR sensors. |
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

## Capabilities and parity

`IsaacSimulation` exposes the same `SimEngine` shape as the MuJoCo backend:

- **World & lifecycle** - `create_world`, `destroy`, `reset`, `step`,
  `get_state`, `cleanup`.
- **Robots** - `add_robot` (procedural builders, or USD via `usd_path=`, or
  URDF), `remove_robot`, `list_robots`, `robot_joint_names`, `send_action`,
  `get_observation`.
- **Objects** - `add_object` (`cuboid` / `sphere` / `cylinder` / `capsule`,
  dynamic or static), `remove_object`.
- **Cameras & rendering** - `add_camera` (look-at, FOV), `render` (RGB + depth).
  World-fixed only: `parent_body` (a body-mounted wrist camera, supported on
  mujoco/newton) is refused here with an error naming those backends, because
  camera prims are parented to the stage camera scope rather than to an
  articulation link.
- **Loaders** - `load_urdf` / `load_mjcf` / `load_usd` resolve to a
  `ProceduralRobot` dataclass.

Because the joint-name and observation contract matches the MuJoCo backend,
policies and observation mappings transfer unchanged between backends.

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
no "derive a label from the model" short form: `name` is also the procedural
lookup key, so `None` / `""` are refused rather than replaced with a generated
label.

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

## Fleet (IsaacLab-style) preview

```python
sim = create_simulation("isaac", num_envs=1024, headless=True,
                        render_mode="headless")
sim.create_world()
sim.add_robot(name="panda", usd_path="/path/to/franka.usda")
# ... RL training loop ...
sim.destroy()
```

## Where to go next

The Isaac backend was originally prototyped in the `strands-robots-sim`
project, which still hosts a MkDocs site with additional architecture notes and
troubleshooting. It is kept here as background reference; the backend itself now
ships in-tree in `strands-robots`:

- Background docs: <https://strands-labs.github.io/robots-sim/>
- Backend reference: <https://strands-labs.github.io/robots-sim/backends/isaac/>
- Source (historical): <https://github.com/strands-labs/robots-sim>
