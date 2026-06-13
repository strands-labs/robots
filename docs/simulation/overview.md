---
description: The Simulation AgentTool — every action grouped by category, with parameters.
---

# Simulation overview

```python
from strands_robots import Robot
sim = Robot("so100")   # preferred factory; 60+ actions as an AgentTool
```

For walkthroughs see [Simulation overview](../simulation/overview.md).

## World

| Action | Key params | Notes |
|--------|-----------|-------|
| `create_world` | `timestep=0.002`, `gravity=[0,0,-9.81]`, `ground_plane=True` | Implicit on `Robot()` |
| `load_scene` | `scene_path` | Replace world with MJCF |
| `reset` | — | State to t=0, keep model |
| `get_state` | — | Sim time, joint positions, object poses |
| `destroy` | — | Tear down model, data, executor |
| `export_xml` | — | Serialise model to MJCF string |

## Scene-MJCF

| Action | Notes |
|--------|-------|
| `replace_scene_mjcf(xml)` | Swap entire world XML |
| `patch_scene_mjcf(ops)` | Incremental patches, no full recompile |
| `raycast(origin, direction, ...)` | Single ray–mesh intersection |
| `multi_raycast(rays, ...)` | Batch ray–mesh intersections |

## Robots

| Action | Key params |
|--------|-----------|
| `add_robot` | `robot_name`, `position=[0,0,0]`, `data_config=None`, `urdf_path=None` |
| `remove_robot` | `name` |
| `list_robots` | — |
| `get_robot_state` | `name` → joint positions, velocities, torques |

## Objects

| Action | Key params |
|--------|-----------|
| `add_object` | `name`, `shape="box"\|"sphere"\|"cylinder"\|"plane"\|"mesh"`, `size`, `position=[x,y,z]`, `color=[r,g,b,a]`, `orientation=[w,x,y,z]`, `mass=0.1`, `is_static=False`, `mesh_path=None` — `plane` requires `is_static=True` |
| `remove_object` | `name` |
| `move_object` | `name`, `position`, `orientation` (NOT `pos`/`quat`) |
| `list_objects` | — |

## Cameras

| Action | Key params |
|--------|-----------|
| `add_camera` | `name`, `position`, `target`, `fov=60.0`, `width=640`, `height=480` — no `attach_to`/`fovy`/`lookat` |
| `remove_camera` | `name` |

Robot-URDF cameras are auto-discovered on `add_robot`.

## Rendering

| Action | Notes |
|--------|-------|
| `render(camera_name="default", width=None, height=None)` | PNG in `content[...]["image"]["source"]["bytes"]`; no `frame` key |
| `render_depth(camera_name="default", width=None, height=None)` | Depth float32 in content; no `depth` key |
| `open_viewer` / `close_viewer` | Interactive MuJoCo passive viewer |

!!! note "Get a numpy frame"
    `sim.get_observation(robot_name)[camera_name]` → `np.uint8 (H, W, 3)`

## Physics

| Action | Key params |
|--------|-----------|
| `step` | `n_steps=1` (max 100 000/call) |
| `set_gravity` | `gravity=[x,y,z]` |
| `set_timestep` | `timestep` |
| `get_contacts` / `get_contact_forces` | — |
| `apply_force` | `body_name`, `force`, `torque`, `point` |
| `get_jacobian` | `body_name` *or* `site_name` *or* `geom_name` |
| `get_mass_matrix` | — |
| `inverse_dynamics` | — |
| `forward_kinematics` | `body_name` (optional) |
| `save_state` / `load_state` | snapshot/restore full physics |
| `get_energy` | — |
| `get_sensor_data` | `sensor_name` (optional) |

## Policy

| Action | Key params |
|--------|-----------|
| `run_policy` | `robot_name` (required), `policy_provider="mock"`, `policy_config={}`, `policy_object=None`, `instruction=""`, `duration=10.0`, `control_frequency=50.0`, `action_horizon=8`, `n_steps=None` |
| `start_policy` | same args, async/non-blocking |
| `stop_policy` | `robot_name` (optional, defaults to `""`) |
| `list_policies_running` | — |
| `run_multi_policy` | `policies={robot: Policy}`, `instructions`, `duration`, `n_steps` |
| `eval_policy` | `robot_name` (required), `n_episodes=1`, `max_steps=300`, `success_fn=None` |
| `replay_episode` | `repo_id`, `robot_name=None`, `episode=0` |

## Recording

| Action | Notes |
|--------|-------|
| `start_recording(repo_id, task="", fps=30, ...)` | LeRobot v3 (parquet+MP4); requires `[lerobot]` extra |
| `stop_recording(output_path=None)` | Finalise episode |
| `get_recording_status` | Episode, frame count, output dir |
| `start_cameras_recording(...)` | Plain MP4 via imageio-ffmpeg; `[sim-mujoco]` only, no lerobot |
| `stop_cameras_recording` / `get_cameras_recording_status` | — |

## Randomize

| Action | Key params |
|--------|-----------|
| `randomize` | `randomize_colors=True`, `randomize_lighting=True`, `randomize_physics=False`, `randomize_positions=False`, `position_noise=0.02`, `color_range=(0.1,1.0)`, `friction_range=(0.5,1.5)`, `mass_range=(0.5,2.0)`, `seed=None` |

Destructive — writes into model arrays. Recompile scene to undo.

## Registry

| Action | Notes |
|--------|-------|
| `list_urdfs` | Loaded URDFs/MJCFs in current world |
| `register_urdf(name, path)` | Register additional asset |
| `get_features(robot_name=None)` | Observation/action feature schema for recording |

## See also

- [World building](world-building.md) — composing scenes.
- [Domain randomization](domain-randomization.md) — `randomize` distributions.
- [Architecture](../architecture.md)
