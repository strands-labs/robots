# Isaac Sim Backend (GPU)

The Isaac Sim backend runs the simulation on
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) (PhysX GPU physics +
RTX path-traced rendering). It ships **out-of-tree** in the sibling package
[`strands-robots-sim`](https://github.com/strands-labs/robots-sim) and registers
an `IsaacSimulation` under the `strands_robots.backends` entry-point group. It
implements the same `SimEngine` contract as the built-in MuJoCo backend, so the
`Robot()` / `Simulation` / policy APIs are identical - only the physics and
rendering run on the GPU through Isaac Sim.

`strands-robots` has **no hard dependency** on Isaac Sim: install the plugin and
`create_simulation("isaac")` discovers it through entry points, exactly like
`create_simulation("mujoco")`.

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

Isaac Sim itself is **not on PyPI** - install it first, then the Python plugin:

```bash
# Step 1 - install Isaac Sim 6.0 (Python 3.12) via one of:
#   - Omniverse Launcher -> Isaac Sim 6.0, OR
#   - Isaac Lab: git clone IsaacLab && ./isaaclab.sh -i, OR
#   - NGC Docker: docker pull nvcr.io/nvidia/isaac-sim:6.0

# Step 2 - install the Python plugin (pulls in strands-robots transitively):
pip install 'strands-robots-sim[isaac]'
```

The `[isaac]` extra lives in `strands-robots-sim`, not in `strands-robots`.
Requesting `create_simulation("isaac")` without the plugin installed raises a
`ValueError` whose message carries the exact install hint
(`pip install 'strands-robots-sim[isaac]'`). Backend discovery is lazy, so
MuJoCo-only users never pay the Isaac Sim import cost.

## Usage

```python
from strands_robots.simulation import create_simulation

# Kwargs flow into IsaacConfig. "isaac" resolves via the
# strands_robots.backends entry point.
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

The plugin reads three `STRANDS_ISAAC_*` variables (resolved when `IsaacConfig`
is constructed). An explicit kwarg always wins over the env var.

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_ISAAC_NUCLEUS_URL` | Override the Omniverse Nucleus server URL when `nucleus_url` is not passed | unset (Isaac defaults) |
| `STRANDS_ISAAC_HEADLESS` | Truthy (`1`/`true`/`yes`) forces `headless`; falsy forces windowed | unset (uses `headless` kwarg) |
| `STRANDS_ISAAC_RTX_PATHTRACING` | Truthy forces `render_mode="rtx_pathtracing"` | unset |

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
- **Loaders** - `load_urdf` / `load_mjcf` / `load_usd` resolve to a
  `ProceduralRobot` dataclass.

Because the joint-name and observation contract matches the MuJoCo backend,
policies and observation mappings transfer unchanged between backends.

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

The plugin ships its own MkDocs site with a Quickstart, architecture, backend
reference, and troubleshooting:

- Plugin docs: <https://strands-labs.github.io/robots-sim/>
- Backend reference: <https://strands-labs.github.io/robots-sim/backends/isaac/>
- Source: <https://github.com/strands-labs/robots-sim>
