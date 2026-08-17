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
| `export_xml` | `output_path` | Serialise live scene to MJCF; reloadable via `load_scene` (assets referenced by absolute path) |

## Scene-MJCF

| Action | Notes |
|--------|-------|
| `replace_scene_mjcf(xml)` | Swap entire world XML |
| `patch_scene_mjcf(ops)` | Incremental patches, no full recompile |
| `raycast(origin, direction, ...)` | Single ray–mesh intersection |
| `multi_raycast(origin, directions, ...)` | Batch ray–mesh intersections from one origin; all-or-nothing, a direction it cannot cast refuses the batch |

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
| `get_world_point(camera_name="default", pixels=[[u, v], ...])` | Ground picked pixels to metric world coordinates via the depth buffer; `point` is the median over the valid samples, `points` aligns with the input pixels |
| `open_viewer` / `close_viewer` | Interactive MuJoCo passive viewer |

!!! note "Get a numpy frame"
    `sim.get_observation(robot_name)[camera_name]` → `np.uint8 (H, W, 3)`

!!! tip "Discover the render surface"
    `render`, `render_depth`, `render_all`, and `get_world_point` are all
    listed in `sim.describe()["methods"]`, so an agent can enumerate the full
    rendering surface in one call instead of guessing method names.

!!! note "Camera intrinsics follow the renderer"
    `sim.get_camera_params(camera_name)` returns the pinhole `K` of the frame
    the renderer actually draws. A camera declaring a physical sensor (MJCF
    `sensorsize` / `focal` / `principal` / `resolution`) has its `K` read from
    the view frustum MuJoCo computes for that camera, so non-square pixels
    (`fx != fy`) and an off-center principal point are honored - including the
    vertical principal-point convention, which MuJoCo changed in 3.6.0. Every
    other camera falls back to `fovy`: square pixels, principal point at the
    image center.

## Physics

| Action | Key params |
|--------|-----------|
| `step` | `n_steps=1` (MuJoCo: max 100 000/call; Isaac and Newton have no ceiling). Non-negative whole number; `0` is an accepted no-op. Errors if the world is destroyed mid-run, naming the steps completed |
| `send_action` | `n_substeps=1` - **positive** whole number, no per-call ceiling (see Actions) |
| `set_gravity` | `gravity=[x,y,z]` or a scalar z-component |
| `set_timestep` | `timestep` |
| `get_contacts` / `get_contact_forces` | - . `get_contacts` lists every geom pair inside the detection range (`margin` + `gap`) and marks each one `active` - MuJoCo hands only the pairs inside `margin` to the solver, so a pair between the two thresholds is a proximity report carrying no force. Contact predicates count only `active` pairs; `get_contact_forces` gives the load a touching pair carries |
| `apply_force` | `body_name`, `force`, `torque`, `point` - latched on that body and re-applied every step until the next `apply_force` for it, so several bodies can hold wrenches at once (`force=[0,0,0]` stops one, `reset()` stops all) |
| `get_jacobian` | `body_name` *or* `site_name` *or* `geom_name` |
| `get_mass_matrix` | - |
| `inverse_dynamics` | - (compensation torques to hold the current `qpos`/`qvel`) |
| `forward_kinematics` | `body_name` (optional) |
| `save_state` / `load_state` | `name` - snapshot/restore full physics. A checkpoint is valid only for the model it was taken against: any scene mutation that swaps the compiled model (`add_object`, `add_robot`, `add_camera`, `remove_camera`, `remove_robot`, `patch_scene_mjcf`, `replace_scene_mjcf`) invalidates it, and `load_state` then returns a structured error instead of writing a state vector whose indices now mean something else. Save a fresh checkpoint after mutating the scene |
| `set_joint_positions` | `positions` (dict or ordered list), `robot_name` (optional), `hold` (optional) - write `qpos` directly + run FK (teleport / set an initial pose, bypassing actuators). Kinematic only: a joint held by a position servo is pulled back toward the setpoint that servo already holds by the next `step`, and the success text names those joints. `hold=True` moves the matching position-servo setpoints with the pose so it survives stepping; a joint driven by a torque or velocity actuator is left alone, since its `ctrl` is not a pose, as is a joint a tendon couples to one `ctrl` (every stock gripper, and the `stretch3` telescoping arm), whose `ctrl` is in tendon units and drives several joints at once |
| `set_joint_velocities` | `velocities` (dict or ordered list), `robot_name` (optional) - write `qvel` directly (set an initial dynamic state) |
| `get_energy` | - |
| `get_sensor_data` | `sensor_name` (optional) |

!!! tip "Discovering joint names"
    The dict form of `set_joint_positions` / `set_joint_velocities` keys by
    joint name, and a name the model cannot resolve is refused rather than
    skipped (the write is all-or-nothing), so the refusal has to say where the
    real names come from. From an agent that is `get_robot_state`, which reports
    every joint of one robot by name with its position and velocity - the
    joint-side counterpart of `list_bodies` for body names. `robot_joint_names`
    returns the same ordering as a plain list, but it is a Python-only
    capability: it is not in the tool schema's `action` enum, so an agent that
    calls it is refused. Reach for it from Python, and for `get_robot_state`
    from a tool call.

!!! note "Numeric domain of the state writers"
    `set_joint_positions`, `set_joint_velocities` and the `apply_force`
    vectors take finite real numbers - a python or NumPy scalar - and refuse a
    boolean. `float(True)` is `1.0`, so a `True` would be written as 1 radian,
    1 rad/s or 1 N and the call would report success; `nan` / `inf` are refused
    because `mj_forward` propagates a `nan` across the whole kinematic state
    and an `inf` velocity blows up the integrator. Each write is
    all-or-nothing, so a refused value leaves `qpos` / `qvel` and every latched
    wrench untouched. This is the same domain the scene-construction vectors
    (`add_object`, `add_camera`) and [`send_action`](#actions) enforce - one
    library, one answer to "is this a usable number".

!!! note "The same domain applies to the world-configuration parameters"
    `set_gravity` / `create_world(gravity=...)`, `set_timestep` /
    `create_world(timestep=...)`, the `mass` on `set_body_properties` and
    `add_object`, the `randomize` ranges and the `set_obs_noise` magnitudes all
    refuse a boolean for the same reason, as do the vectors `raycast`,
    `multi_raycast` and `set_geom_properties` take (a ray origin and direction, a
    geom size and friction, an rgba colour).

    Passing one is not a near miss. `set_gravity(True)` would have configured a
    gravity of **+1 m/s^2, pointing up**, and `set_timestep(True)` a 1-second
    integration step - each reported as `status="success"`. The check is on the
    type, not the value: `1`, `1.0` and `numpy` scalars remain accepted
    everywhere, so `set_timestep(1.0)` is still a legal (if unusual) request.

    Both spellings are refused - a python `bool` and a `numpy.bool_`. The second
    matters more in practice, because it is what a comparison such as
    `gripper > 0.5` produces, and because `numpy.bool_` is not a `bool` subclass
    an `isinstance(x, bool)` guard silently misses it.

!!! note "Component count of a vector parameter"
    Every vector parameter (`position`, `target`, `origin`, `force`, `torque`,
    `point`, `gravity`, `direction`, `orientation`, `color`, `get_world_point`'s
    `pixels`, and `send_action`'s ordered-vector form) is checked for its
    component count before it is read, and a value that carries no readable
    count is refused with a structured error like any other. That includes a
    0-d NumPy array or torch tensor - `np.mean(...)`, `np.array(0.5)`, a
    squeezed observation slice - which *declares* `__len__` and then raises
    from it, so it is reported as "not a vector of N numbers" rather than
    escaping as a bare `len() of unsized object`. Correctly sized NumPy arrays
    are accepted throughout, so an observation slice can be passed straight
    through.

!!! tip "Discover the sim-state surface"
    `get_state` plus the checkpoint (`save_state` / `load_state`) and
    direct pose-setting (`set_joint_positions` / `set_joint_velocities`)
    methods are all listed in `sim.describe()["methods"]`, so an agent can
    learn how to snapshot/restore the world and set a deterministic initial
    condition from one `describe()` call - no method-name guessing.

## Actions

`send_action(action, robot_name=None, n_substeps=1)` writes actuator/joint targets and advances physics. `action` accepts either form:

| Form | Binding |
|------|---------|
| `{joint_or_actuator_name: value}` mapping | applied by name; unresolved keys are reported in an `unresolved_keys` JSON block so a caller can self-correct (no silent drop) |
| ordered numeric vector (`list` / `tuple` / 1-D `numpy` array) | bound positionally to `robot_action_keys(robot_name)` (the robot's actuator keys) in declaration order - the same convention `replay_episode` uses |

A vector lets a policy's raw action chunk drive the arm directly without first zipping it into a dict. It binds to `robot_action_keys` (not `robot_joint_names`) because those are the keys `send_action` resolves and the ordering the `LeRobotDataset` recorder writes the `action` column in; the two coincide unless a robot has passive/mimic joints, a tendon gripper, or a floating base on the Newton backend (whose 6-DoF free joint is a joint but not a commandable scalar, so it is absent from the action keys). The vector length must match the robot's actuator count exactly; a mismatch (or a non-numeric / scalar / string `action`) returns a structured `status="error"` dict naming the actuator count and order, rather than crashing or silently truncating commands. Use a mapping to target a subset of actuators.

`n_substeps` is the number of physics steps the written targets are held for. It must be a **positive** whole number: a NumPy or integral-float count (`np.int64(3)`, a `3.0` read from a config) is honored and coerced, and a fractional, zero, negative, non-finite, boolean or non-numeric count returns a structured `status="error"` dict. Nothing is written when it does - a refusal arriving after the write would leave the robot commanded and the world un-advanced. The floor is `1` rather than `step`'s `0` because of that write: to advance without commanding, use `step(n)`, whose `0` is an accepted no-op. It is also the floor both producers of this count already enforce (`PolicyRunner`'s `control_substeps` and the RL env's `n_substeps`).

Each action *value* must be a finite number, and must not be a boolean. `nan` / `inf` are refused because they are not clamped into the actuator's range - MuJoCo discards the step and resets every robot in the scene while reporting success. A `bool` (or `numpy.bool_`) is refused because `float(True)` is `1.0`, and each drive reads 1.0 in its own units: a 1-radian target on a joint-position drive, a full-travel command on a normalized or tendon drive (a `[0, 255]` tendon gripper reads it as fully open), and an out-of-range value that is silently clamped where `ctrlrange` excludes 1 - so the same `True` commands a different pose on every actuator. Send the command in the actuator's own units; for a binary gripper, its endpoint value rather than a flag. This is the domain the teleop wire validator already enforces on an input frame, and `InputReceiver` applies those frames through `send_action`.

## Policy

| Action | Key params |
|--------|-----------|
| `run_policy` | `robot_name` (required), `policy_provider="mock"`, `policy_config={}`, `policy_object=None`, `instruction=""`, `duration=10.0`, `control_frequency=50.0`, `action_horizon=8`, `n_steps=None`, `seed=None`, `async_rtc=None`, `rtc_inference_timeout_s=None` |
| `start_policy` | same args, async/non-blocking |
| `stop_policy` | `robot_name` (optional, defaults to `""`) |
| `list_policies_running` | - |
| `run_multi_policy` | `policies={robot: Policy}`, `instructions`, `duration`, `n_steps` |
| `eval_policy` | `robot_name` (optional; auto-resolves the sole robot like `run_policy`), `n_episodes=1`, `max_steps=300`, `success_fn=None`, `async_rtc=False`, `rtc_inference_timeout_s=None`, `video=None` |

When a policy is run via `run_policy` / `eval_policy` / `run_multi_policy`, the simulation configures the policy's output keys with the robot's *action keys* via `set_robot_state_keys(robot_action_keys(robot_name))`. `robot_action_keys` returns the actuator short-names that `send_action` resolves - which are not always the robot's joints. Robots with passive / mimic finger joints (no driving actuator) or a tendon-driven gripper (an actuator with no matching joint name) have an actuator set distinct from their joint set, so keying a policy by `robot_joint_names` would emit keys that resolve to nothing and leave those DOFs unmoved. The list can also be *narrower* than the joint names rather than differently spelled: on the Newton backend a floating base's 6-DoF free joint is a joint with no scalar target to write, so it is excluded from the action keys (its pose is read as the structured `base_pos` / `base_quat` / `base_lin_vel` / `base_ang_vel` signals instead) and `send_action` refuses it as a command key. The default `robot_action_keys` mirrors `robot_joint_names` for backends whose actuators match their joints; do not assume the two have the same width.

The step horizon is given either as `duration` (seconds) or as `n_steps` (`duration = n_steps / control_frequency`; `n_steps` wins when both are set, and the legacy `max_steps` is an alias for `n_steps`). A non-positive `n_steps` or `control_frequency` is rejected up front with a structured `status="error"` dict naming the bad parameter - `start_policy` validates synchronously before the background rollout starts, so a malformed horizon never returns a false "started" success. `eval_policy` likewise rejects a non-positive `n_episodes`, `max_steps`, or `control_frequency` at the entry point (before `create_policy`), so a typo cannot produce a "successful" evaluation over zero or negative episodes. The same entry-point check covers the two provider keyword bags: `policy_config` (splatted into `create_policy`) and `policy_kwargs` (splatted into `policy.get_actions`) must be dicts, so a `policy_config="host=127.0.0.1"` string returns a structured error naming the parameter instead of a bare `TypeError` from the splat - and, on the `start_policy` path, instead of a false "started" for a rollout that never produced an action. The pair resolves the same way one layer down, on `PolicyRunner.run` - the surface those entry points delegate to, which is documented as drivable directly - and both knobs carry their domain there too, raising `ValueError` rather than returning an error dict because a direct caller has no envelope to read a refusal from. `n_steps` is judged whenever it is *given*, which is the condition the entry point's own resolver judges it on; unvalidated, a step count outside the domain did not fail but handed the horizon to the other knob, so `n_steps=0` ran `duration`'s `10.0`s default - 500 control steps and 500 applied actions for a caller who asked for zero. `duration` is judged only when no step count was given, because that is the only case in which it sets the horizon; unvalidated, `0` and a negative value returned `status="success"` with zero steps and `stopped_reason="budget"` - the field a caller reads to decide whether to retry, asserting a horizon was exhausted when there was none.

`action_horizon` (how many actions are consumed from each policy chunk before it is re-queried) is validated the same way at every entry point, so a horizon the rollout cannot run - `0`, a negative value, a float, `nan` - is a structured error rather than a value silently clamped to 1. `run_multi_policy` additionally accepts per-robot mappings (`instructions={robot: text}`, `action_horizon={robot: horizon}`): a key must name a robot driven by that call (i.e. a key of `policies`), because an unmatched key cannot be applied to anything - a robot omitted from a mapping keeps its documented default. The same domain applies one layer down, on `PolicyRunner.run` / `PolicyRunner.evaluate` - the surfaces those entry points delegate to, which are documented as drivable directly. There it raises `ValueError` rather than returning an error dict, matching the sibling `control_substeps` and `control_frequency` guards of the same signature, because a direct caller has no envelope to read a refusal from; unvalidated, the value was clamped to 1 inside the first chunk query or leaked a bare `int()` conversion error naming neither the parameter nor the method. The two bounds of `PolicyRunner.evaluate`'s own episode loop - `n_episodes` and, on the legacy `success_fn` path, `max_steps` - carry that same domain for a stronger reason: a horizon outside it degrades a rollout, while a loop bound outside it removes the evaluation and still reports one. `n_episodes=0` returned `status="success"` over zero episodes and `max_steps=0` over episodes of zero length, both with `success_rate: 0.0` and `success_measured: true` - the flag that exists so a `0.0` cannot be read as a measurement - and with no action ever applied; `max_steps=inf` never terminated at all, since `while steps < max_steps` has no false case. `max_steps` is checked only when it is the horizon actually read, because a `spec=` call takes its horizon off the benchmark (validated at that read) and never reads the parameter.

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

**Hardening.** If a prefetched chunk arrives empty, the runner degrades to one synchronous re-query before erroring (a transient hiccup does not kill an otherwise-healthy rollout). When a prefetch blocks at the seam (inference slower than chunk execution) the runner logs a starvation warning so you can shorten the chunk or fire the prefetch earlier. Set `rtc_inference_timeout_s` to bound a stuck inference: the swap then returns a structured `status="error"` result (carrying the telemetry below) instead of waiting for every remaining chunk - bounded by the single in-flight inference the executor joins on shutdown (Python cannot forcibly kill a running worker thread). That deadline must be a positive finite number of seconds, or `None` (the default) to wait without one - `0`, a negative value and `nan` all make the wait give up before any inference can answer, and `inf` overflows the platform's timestamp arithmetic, so each is refused at the call naming the parameter rather than reported one rollout later as a stuck policy.

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

`run_policy` returns a `{"json": {...}}` content block alongside the human-readable `text`, mirroring `eval_policy`. The json block carries the rollout facts as typed fields - `robot_name`, `policy`, `instruction`, `n_steps`, `elapsed_s`, `stopped_early`, `action_errors`, `video_path` (`None` when no MP4 was written), `video_frames`, `sim_time_s` (when the backend reports it) and the six `rtc_*` async-RTC telemetry fields above - so an agent can read the outcome programmatically (did it move? how many steps? was inference masked?) without regex-parsing the prose. The `status` reflects whether the robot *moved*, not merely whether every key resolved: a run where **no** step resolved any key (the robot never moved) returns `status="error"`, while a run where some keys resolve every step - e.g. a policy trained on a superset embodiment that emits one extra key the robot lacks - is operational and returns `status="success"` with a non-fatal `N/M action steps had unresolved keys` note and a `partial_action_failure_rate`.

`eval_policy` accepts the same `video={...}` recording config as `run_policy` (`path` enables it, plus `fps` / `camera` / `width` / `height` - an unknown key or a non-positive size is a caller error, never silently ignored), but writes **one MP4 per episode** with `_ep{i}` inserted into the filename (`eval.mp4` -> `eval_ep0.mp4`, `eval_ep1.mp4`, ...), so a multi-episode evaluation can be *watched* to see why episodes fail rather than only read as an aggregate `success_rate`. The written files are listed in the result json `video_paths`; the output path is validated and the camera probed up-front, so a bad camera fails the eval immediately instead of after N episodes of empty MP4s. `evaluate_benchmark` accepts the same `video={...}` config and records one MP4 per episode too, so a benchmark evaluation can be watched to see why episodes fail. Frames are captured synchronously on the eval thread (render is read-only over `mjData`), so recording does not perturb the bit-stable benchmark rollout.
| `replay_episode` | `repo_id`, `robot_name=None`, `episode=0` |

!!! tip "Discover the benchmark scoring surface"
    `evaluate_benchmark`, `list_benchmarks`, `register_benchmark_from_file`,
    and `register_builtin_benchmarks` are listed in `sim.describe()["methods"]`, so an agent that can run a
    policy from one `describe()` call can also discover how to score it
    against a success/failure/dense_reward benchmark - and author a new
    benchmark spec at runtime - without guessing the method names.

**Built-in benchmarks.** `sim.register_builtin_benchmarks()` (or the module
function `strands_robots.simulation.register_builtin_benchmarks()`) registers
the benchmarks shipped with the library so they appear in `list_benchmarks()`
and run via `evaluate_benchmark(...)` without hand-authoring a spec. It ships
`go2_walk_forward` - a canonical velocity-tracking locomotion task for the
Unitree Go2: succeed by walking the base past `x = 2 m` (`base_beyond_x`), fail
on a topple (`base_tipped`) or a height collapse (`base_below_z`), and shape on
a dense `base_velocity_tracking` (exp-kernel twist tracking) + `base_height` +
`base_orientation` reward. Registration is opt-in (mirrors the on-demand LIBERO
suite), so importing the library mutates no registry. `builtin_benchmark_specs()`
returns the spec dicts to copy/fork as a starting point for your own task.

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
| `get_features(robot_name=None)` | Joint / actuator / camera / robot names of the scene (scoped to one robot with `robot_name`) - the source of truth for the action keys a policy must emit, and the feature schema used for recording |

!!! tip "Discover the expected action keys"
    `get_features` is listed in `sim.describe()["methods"]`, so an agent can
    find it from one `describe()` call. When a policy's emitted action keys
    resolve to no actuator, `run_policy` fails fast with an error that names
    `get_features(robot_name=...)` as the way to inspect the keys the robot
    actually expects - the recommended method and the discovery surface agree.

## See also

- [World building](world-building.md) - composing scenes.
- [Domain randomization](domain-randomization.md) - `randomize` distributions.
- [Architecture](../architecture.md)
