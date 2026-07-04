---
description: The Simulation AgentTool - every action grouped by category, with parameters.
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
| `reset` | - | State to t=0, keep model |
| `get_state` | - | Sim time, joint positions, object poses |
| `destroy` | - | Tear down model, data, executor |
| `export_xml` | - | Serialise model to MJCF string |

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
| `list_robots` | - |
| `get_robot_state` | `name` → joint positions, velocities, torques |

## Objects

| Action | Key params |
|--------|-----------|
| `add_object` | `name`, `shape="box"\|"sphere"\|"cylinder"\|"plane"\|"mesh"`, `size`, `position=[x,y,z]`, `color=[r,g,b,a]`, `orientation=[w,x,y,z]`, `mass=0.1`, `is_static=False`, `mesh_path=None` - `plane` requires `is_static=True` |
| `remove_object` | `name` |
| `move_object` | `name`, `position`, `orientation` (NOT `pos`/`quat`) |
| `list_objects` | - |

## Cameras

| Action | Key params |
|--------|-----------|
| `add_camera` | `name`, `position`, `target`, `fov=60.0`, `width=640`, `height=480` - no `attach_to`/`fovy`/`lookat` |
| `remove_camera` | `name` |
| `list_cameras` | - renderable camera names, `"default"` first, incl. model + user cameras |

Robot-URDF cameras are auto-discovered on `add_robot`.

`sim.list_cameras()` returns every name `render` / `start_recording` accepts -
the built-in `"default"` free view first, then all model-defined and
`add_camera` cameras. It equals `sim.describe()["cameras"]` and matches the
Newton backend, so a rollout rig can be enumerated instead of guessed.

!!! tip "Discover the scene-construction surface"
    `add_robot`, `add_object`, `remove_object`, `add_camera`,
    `remove_camera`, and `list_cameras` are all listed in
    `sim.describe()["methods"]`, so an agent can learn how to build a scene
    (robot, manipulanda, camera rig) before a rollout from one `describe()`
    call instead of guessing method names.

## Rendering

| Action | Notes |
|--------|-------|
| `render(camera_name="default", width=None, height=None)` | PNG in `content[...]["image"]["source"]["bytes"]`; no `frame` key |
| `render_depth(camera_name="default", width=None, height=None)` | Viewable grayscale depth PNG `image` block (near=bright, far=dark) + metric `depth_min`/`depth_max` (meters) in the `json` block |
| `render_all(cameras=None, width=None, height=None)` | One `image` block per camera (multi-view snapshot) |
| `open_viewer` / `close_viewer` | Interactive MuJoCo passive viewer |

!!! note "Get a numpy frame"
    `sim.get_observation(robot_name)[camera_name]` → `np.uint8 (H, W, 3)`

!!! tip "Discover the render surface"
    `render`, `render_depth`, and `render_all` are all listed in
    `sim.describe()["methods"]`, so an agent can enumerate the full rendering
    surface in one call instead of guessing method names.

## Physics

| Action | Key params |
|--------|-----------|
| `step` | `n_steps=1` (max 100 000/call) |
| `set_gravity` | `gravity=[x,y,z]` |
| `set_timestep` | `timestep` |
| `get_contacts` / `get_contact_forces` | - |
| `apply_force` | `body_name`, `force`, `torque`, `point` |
| `get_jacobian` | `body_name` *or* `site_name` *or* `geom_name` |
| `get_mass_matrix` | - |
| `inverse_dynamics` | - (compensation torques to hold the current `qpos`/`qvel`) |
| `forward_kinematics` | `body_name` (optional) |
| `save_state` / `load_state` | snapshot/restore full physics |
| `get_energy` | - |
| `get_sensor_data` | `sensor_name` (optional) |

## Actions

`send_action(action, robot_name=None, n_substeps=1)` writes actuator/joint targets and advances physics. `action` accepts either form:

| Form | Binding |
|------|---------|
| `{joint_or_actuator_name: value}` mapping | applied by name; unresolved keys are reported in an `unresolved_keys` JSON block so a caller can self-correct (no silent drop) |
| ordered numeric vector (`list` / `tuple` / 1-D `numpy` array) | bound positionally to `robot_action_keys(robot_name)` (the robot's actuator keys) in declaration order - the same convention `replay_episode` uses |

A vector lets a policy's raw action chunk drive the arm directly without first zipping it into a dict. It binds to `robot_action_keys` (not `robot_joint_names`) because those are the keys `send_action` resolves and the ordering the `LeRobotDataset` recorder writes the `action` column in; the two coincide unless a robot has passive/mimic joints or a tendon gripper. The vector length must match the robot's actuator count exactly; a mismatch (or a non-numeric / scalar / string `action`) returns a structured `status="error"` dict naming the actuator count and order, rather than crashing or silently truncating commands. Use a mapping to target a subset of actuators.

## Policy

| Action | Key params |
|--------|-----------|
| `run_policy` | `robot_name` (required), `policy_provider="mock"`, `policy_config={}`, `policy_object=None`, `instruction=""`, `duration=10.0`, `control_frequency=50.0`, `action_horizon=8`, `n_steps=None`, `seed=None`, `async_rtc=None`, `rtc_inference_timeout_s=None` |
| `start_policy` | same args, async/non-blocking |
| `stop_policy` | `robot_name` (optional, defaults to `""`) |
| `list_policies_running` | - |
| `run_multi_policy` | `policies={robot: Policy}`, `instructions`, `duration`, `n_steps` |
| `eval_policy` | `robot_name` (optional; auto-resolves the sole robot like `run_policy`), `n_episodes=1`, `max_steps=300`, `success_fn=None`, `async_rtc=False`, `rtc_inference_timeout_s=None`, `video=None` |

When a policy is run via `run_policy` / `eval_policy` / `run_multi_policy`, the simulation configures the policy's output keys with the robot's *action keys* via `set_robot_state_keys(robot_action_keys(robot_name))`. `robot_action_keys` returns the actuator short-names that `send_action` resolves - which are not always the robot's joints. Robots with passive / mimic finger joints (no driving actuator) or a tendon-driven gripper (an actuator with no matching joint name) have an actuator set distinct from their joint set, so keying a policy by `robot_joint_names` would emit keys that resolve to nothing and leave those DOFs unmoved. The default `robot_action_keys` mirrors `robot_joint_names` for backends whose actuators match their joints.

The step horizon is given either as `duration` (seconds) or as `n_steps` (`duration = n_steps / control_frequency`; `n_steps` wins when both are set, and the legacy `max_steps` is an alias for `n_steps`). A non-positive `n_steps` or `control_frequency` is rejected up front with a structured `status="error"` dict naming the bad parameter - `start_policy` validates synchronously before the background rollout starts, so a malformed horizon never returns a false "started" success. `eval_policy` likewise rejects a non-positive `n_episodes`, `max_steps`, or `control_frequency` at the entry point (before `create_policy`), so a typo cannot produce a "successful" evaluation over zero or negative episodes.

Pass `seed=` to `run_policy` / `start_policy` for a reproducible single rollout: it reseeds Python / NumPy / torch / cuDNN and forwards `policy.reset(seed=...)`, so a stochastic policy (VLA action-chunk sampling, diffusion noise) produces the same trajectory on re-run of the same scene. Without a seed the rollout draws from the process-global RNG and can differ run to run. `eval_policy` already seeds per episode via the same mechanism.

### Async-RTC chunk pipeline (latency masking)

`async_rtc` overlaps policy inference with action execution: while the current action chunk drains, the *next* `get_actions` runs on a single background worker (using a fresh mid-chunk observation) and is atomically swapped in when the current chunk runs out. A policy whose inference latency is at most one chunk's execution time then pays (almost) zero visible stall at the chunk seam - the same way an async real-time controller hides inference latency on real hardware.

```
async_rtc=True (inference <= chunk execution):

chunk N exec   |####============|
prefetch N+1            |~~~~~~~|              <- fires at ~50% of chunk N
chunk N+1 exec                  |####========|   <- ready at the seam: HIT, no stall

async_rtc=False (synchronous chunk-then-drain):

chunk N exec   |####|
infer N+1            |~~~~~~~|                 <- the loop stalls here every seam
chunk N+1 exec               |####|
```

**Auto-enable rule.** `async_rtc=None` (the default) resolves the flag from `policy.is_chunk_emitting()`: chunk-emitting VLA / flow-matching policies (pi0, pi0.5, pi0-FAST, SmolVLA, MolmoAct2) get the overlap automatically, while single-step policies (MockPolicy, classical planners) stay on the synchronous loop, where overlap would gain nothing. An explicit `async_rtc=True` / `async_rtc=False` always wins over the auto-resolution. `Policy.is_chunk_emitting()` defaults to `execution_horizon > 1`; `LerobotLocalPolicy` additionally reports `True` for an RTC model or a checkpoint that must be driven via `predict_action_chunk` (MolmoAct2). See [LeRobot Local -> RTC](../policies/lerobot-local.md#synchronous-vs-async-chunk-execution-in-sim).

**Hardening.** If a prefetched chunk arrives empty, the runner degrades to one synchronous re-query before erroring (a transient hiccup does not kill an otherwise-healthy rollout). When a prefetch blocks at the seam (inference slower than chunk execution) the runner logs a starvation warning so you can shorten the chunk or fire the prefetch earlier. Set `rtc_inference_timeout_s` to bound a stuck inference: the swap then returns a structured `status="error"` result (carrying the telemetry below) instead of waiting for every remaining chunk - bounded by the single in-flight inference the executor joins on shutdown (Python cannot forcibly kill a running worker thread).

**Telemetry.** Every `run_policy` result `{"json": {...}}` block carries six RTC fields so latency masking is provable from the payload, not the logs:

| Field | Meaning |
|-------|---------|
| `rtc_async_enabled` | Whether the overlap pipeline ran (the resolved `async_rtc`) |
| `rtc_chunks_acquired` | Chunks the rollout acquired (cold start + swaps + re-queries) |
| `rtc_prefetch_hits` | Seams where the next chunk was already computed (stall hidden) |
| `rtc_prefetch_blocks` | Seams where the runner had to wait for inference (seam starved) |
| `rtc_avg_inference_ms` | Mean `get_actions` wall time across the rollout |
| `rtc_max_inference_ms` | Slowest `get_actions` wall time |

A healthy masked rollout shows `rtc_prefetch_hits` near the chunk count and `rtc_prefetch_blocks == 0`; persistent blocks mean inference is slower than chunk execution and the seam cannot be fully hidden.

**Async-RTC in `eval_policy` (opt-in).** The success-rate eval path (`eval_policy` / `evaluate(success_fn=...)`) accepts the same `async_rtc` and `rtc_inference_timeout_s`, but defaults to `async_rtc=False`. The synchronous eval pauses the world during inference, so the success-rate is bit-stable and reproducible (the policy always sees the seam observation). Setting `async_rtc=True` evaluates a chunk-emitting policy under the realistic control latency it faces in deployment: the prefetch feeds the policy a slightly staler (mid-chunk) observation at the seam, so the measured success-rate can shift - that is the point, it measures robustness to inference latency. Either way the eval `{"json": {...}}` payload now carries the same six `rtc_*` fields (inference timing is reported even on the synchronous path). `async_rtc=True` is rejected on the benchmark/spec path (`evaluate_benchmark` / `evaluate(spec=...)`), which stays synchronous for bit-stable reproducibility; use `run_policy(async_rtc=...)` for benchmark-style wall-clock latency masking.

`run_policy` returns a `{"json": {...}}` content block alongside the human-readable `text`, mirroring `eval_policy`. The json block carries the rollout facts as typed fields - `robot_name`, `policy`, `instruction`, `n_steps`, `elapsed_s`, `stopped_early`, `action_errors`, `video_path` (`None` when no MP4 was written), `video_frames`, `sim_time_s` (when the backend reports it) and the six `rtc_*` async-RTC telemetry fields above - so an agent can read the outcome programmatically (did it move? how many steps? was inference masked?) without regex-parsing the prose.

`eval_policy` accepts the same `video={...}` recording config as `run_policy` (`path` enables it, plus `fps` / `camera` / `width` / `height`), but writes **one MP4 per episode** with `_ep{i}` inserted into the filename (`eval.mp4` -> `eval_ep0.mp4`, `eval_ep1.mp4`, ...), so a multi-episode evaluation can be *watched* to see why episodes fail rather than only read as an aggregate `success_rate`. The written files are listed in the result json `video_paths`; the output path is validated and the camera probed up-front, so a bad camera fails the eval immediately instead of after N episodes of empty MP4s. Recording is scoped to the `success_fn` path; passing `video` with a benchmark `spec` (`evaluate_benchmark`) is rejected - use `run_policy(video=...)` for a benchmark rollout video.
| `replay_episode` | `repo_id`, `robot_name=None`, `episode=0` |

## Recording

| Action | Notes |
|--------|-------|
| `start_recording(repo_id, task="", fps=30, ...)` | LeRobot v3 (parquet+MP4); requires `[lerobot]` extra |
| `save_episode()` | Flush the current rollout as one episode; call once per `run_policy` to record N episodes instead of one merged episode |
| `stop_recording(output_path=None)` | Finalise dataset (flushes any trailing rollout) |
| `get_recording_status` | Episode, frame count, output dir |
| `start_cameras_recording(...)` | Plain MP4 via imageio-ffmpeg; `[sim-mujoco]` only, no lerobot |
| `stop_cameras_recording` / `get_cameras_recording_status` | - |

## Randomize

| Action | Key params |
|--------|-----------|
| `randomize` | `randomize_colors=True`, `randomize_lighting=True`, `randomize_physics=False`, `randomize_positions=False`, `position_noise=0.02`, `color_range=(0.1,1.0)`, `friction_range=(0.5,1.5)`, `mass_range=(0.5,2.0)`, `seed=None` |

Destructive - writes into model arrays. Recompile scene to undo.

## Registry

| Action | Notes |
|--------|-------|
| `list_urdfs` | Loaded URDFs/MJCFs in current world |
| `register_urdf(name, path)` | Register additional asset |
| `get_features(robot_name=None)` | Observation/action feature schema for recording |

## See also

- [World building](world-building.md) - composing scenes.
- [Domain randomization](domain-randomization.md) - `randomize` distributions.
- [Architecture](../architecture.md)
