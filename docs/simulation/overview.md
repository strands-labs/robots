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

!!! tip "Releasing a world you did not destroy"
    `destroy` / `cleanup` (or the context manager) release the world, the
    renderers, the executor, the ROS 2 bridge, attached teleoperated devices and
    the mesh peer. A script that never calls them is covered at process exit:
    the MuJoCo backend releases every still-live simulation from an `atexit`
    hook, which runs while the import system is intact. The `__del__` safety net
    alone cannot - by the time a finalizer runs at exit CPython has already set
    the teardown path's module globals to `None`, so the first step raises and
    nothing is released. Prefer explicit release anyway: it is the only form
    that gives you the result, and it frees the GPU/GL resources at the point
    you stop needing them rather than at exit.

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
| `list_robots` | - each robot's asset, joint count, and **live** base position, read from the physics rather than from the `add_robot` request, so a robot that walked (or whose model's root pose offset the request) reports where it is |
| `get_robot_state` | `name` → joint positions, velocities, torques |

!!! note "The frame `move_to` drives"
    `move_to` and `get_robot_state`'s `end_effector` line follow the frame
    `discover_ee_frame` finds: a tool-point **site** first (`tcp`, `gripper`,
    `attachment_site`, …), else a hand/wrist **body**. A vendored model that
    ships no site lands on the wrist, so a registry entry may declare the tool
    point the model lacks and the backend adds that site before the attach:

    ```json
    "tool_frame": {"body": "Fixed_Jaw", "pos": [0.0, -0.0995, 0.001], "site": "tcp"}
    ```

    `body` is the model's own body name, `pos` is meters in that body's frame,
    `site` defaults to `tcp`. The shipped `so100` entry carries one (its
    Menagerie model has zero sites; the point sits between the jaw tips like
    the SO-101's own `gripper` site). A malformed block, or one naming a body
    the model lacks, refuses `add_robot` with the reason - never a silent
    fall-back to the wrist. The overlay `user_robots.json` may declare one for
    your own robot.

## Objects

| Action | Key params |
|--------|-----------|
| `add_object` | `name`, `shape="box"\|"sphere"\|"cylinder"\|"plane"\|"mesh"`, `size`, `position=[x,y,z]`, `color=[r,g,b,a]`, `orientation=[w,x,y,z]`, `mass=0.1`, `is_static=None`, `mesh_path=None` - omitted lets the shape decide: `plane` is made static and refuses an explicit `is_static=False`, every other shape is dynamic |
| `remove_object` | `name` |
| `move_object` | `name`, `position`, `orientation` (NOT `pos`/`quat`) |
| `list_objects` | - each object's shape, mass, and **live** position, read from the physics rather than from the `add_object`/`move_object` request, so a settled or pushed object reports where it is |

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

!!! note "Where `render(output_path=...)` may write"
    `output_path` is model-supplied, so it is confined to a render sandbox: a
    bare filename (`"frame.png"`) lands there and an absolute path outside it
    is refused. The sandbox is `~/.strands_robots/renders` by default,
    `STRANDS_ROBOTS_RENDER_ROOT` process-wide, or - per Simulation, without an
    environment variable - `Robot("so101", render_dir="./shots")` /
    `Simulation(render_dir=...)`. Set `render_dir` to the directory you want the
    files in and the agent can hand them to you there.

!!! tip "Discover the render surface"
    `render`, `render_depth`, `render_all`, and `get_world_point` are all
    listed in `sim.describe()["methods"]`, so an agent can enumerate the full
    rendering surface in one call instead of guessing method names.

!!! note "Frame reads are serialised against physics"
    `render`, `render_depth` and `get_frame` copy mjData into the renderer under
    the simulation lock and hand back an independent buffer, so a frame captured
    while a policy worker, the `step()` loop or the camera recorder is advancing
    physics is a consistent snapshot rather than a torn one. Only the PNG
    encoding runs unlocked. This holds for a direct Python call and for a call
    from your own thread, not just through the tool surface.

!!! note "State reads are serialised against physics too"
    `get_robot_state` reads every joint's `qpos`/`qvel`, a floating base's pose
    and twist, and the world position of the frame `move_to` drives in one
    critical section, so the answer is a configuration the robot was actually in
    rather than a splice of two physics steps - joint angles that never
    coexisted, or joints from one step beside an `end_effector` position from
    another (the field whose documented use is to offset a `move_to` target from
    it). `get_observation`, `get_body_state` and the joint writers serialise the
    same way.

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
| `get_jacobian` | `body_name` *or* `site_name` *or* `geom_name`. Columns are DOFs of the whole compiled model, so the width is not the robot's joint count: a free or ball joint owns several consecutive columns, and a scene holding two robots reports one width spanning both. The `json` block's `dof_joint_names` names the joint owning each column - pair `dq` with that, not with `robot_joint_names`, or one robot's Jacobian reads as another's |
| `get_mass_matrix` | - . The reported `diagonal` is DOF-indexed on the same terms, and `dof_joint_names` names each entry's joint |
| `inverse_dynamics` | - (compensation torques to hold the current `qpos`/`qvel`) |
| `forward_kinematics` | `body_name` (optional) - refreshes every body's pose from the current `qpos`, then filters to one body when named. Like `get_body_state` / `get_jacobian` / `apply_force` / `set_body_properties`, the name may be bare (`gripper`) or namespaced (`arm0/gripper`): `add_robot` namespaces every compiled body, and the bare form is retried under each robot's namespace |
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

    Some assets name joints by servo id or CAD term - the SO-101's are
    `1`..`6`, the SO-100's `Rotation`..`Jaw`. For those, the registry entry
    carries `joint_labels` (the `shoulder_pan` .. `gripper` the same arm's
    driver and datasets use): `get_robot_state` prints `1 (shoulder_pan)`, and
    the joint writers accept the label as a key - bare, `<robot>/<label>`, in
    any case - beside the asset name. A refused key lists the labels too.

!!! tip "The unit of a joint value"
    A joint value is in that joint's own MuJoCo unit: radians for a hinge,
    metres for a slide - never degrees. It matters most when mirroring a real
    arm onto its sim twin, because the driver on the other side reports the
    other unit (`drivers/feetech` reads an SO-arm in degrees) and the same
    number in the wrong unit is a pose an order of magnitude away. So
    `set_joint_positions` names the unit when it refuses a value the joint's
    range does not contain, and when converting that value *would* land inside
    the range it says which conversion:
    `shoulder_pan=-96.2 outside [-1.92, 1.92] rad (radians, not degrees:
    -96.2 deg = -1.679 rad)`.

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

!!! note "Finite is not enough for a value that lands in `qpos` or `qvel`"
    `mj_step` checks `qpos`, `qvel` and `qacc` against `mjMAXVAL` (1e10) before
    it integrates, and past that ceiling MuJoCo calls the simulation unstable
    and resets *every* joint and object to its initial state, reporting it only
    on stderr. So `set_joint_positions`, `move_object` and `add_object` hold a
    caller value to that ceiling, as `set_joint_velocities` does, and refuse
    before writing rather than reporting success over a world the next `step`
    wipes. A joint's range already bounds a limited joint; the ceiling is what
    bounds one that declares no range (a floating base, a continuous hinge) and
    a dynamic object's freejoint pose, whose quaternion is not renormalized on
    write. It is MuJoCo's own limit, so `mjMAXVAL` exactly is still writable. A
    static object is welded with no freejoint and owns no `qpos` entry, so a
    far-away static body is stable and is not held to the ceiling.

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

`stop_policy` is honored at **any** point after `start_policy` returns, including before the rollout's first frame and while it is still queued behind a busy executor. The rollout is claimed by the launching thread rather than by the background worker, so a stop can only ever land on a rollout that is already marked as running: it reports `Stopped on '<robot>'` and the rollout takes no further frames. Its verdict is derived from the same in-flight population `list_policies_running` reads, so the two never report opposite facts about the same robot at the same instant. That population counts a rollout in either launch shape - one submitted by `start_policy` and one being driven right now by the blocking `run_policy`, which registers no future - so a blocking rollout is reported as running, is named by `list_policies_running`, and is halted by a stop that carries no `robot_name` (the shape the mesh e-stop fanout broadcasts); `Was not running on '<robot>'` is reserved for the genuinely idempotent case, where nothing is in flight at all. Every stop path goes through one seam - `SimRobot.request_policy_stop` - which is what makes the halt durable: `stop_policy`, `remove_robot`, teardown, and the Device Connect `stop` / `emergencyStop` handlers cannot drift to different answers about whether a rollout was actually halted.
Scene mutations read that same in-flight population. `add_robot`, `remove_robot`, `add_object`, `remove_object`, `move_object`, `add_camera`, `remove_camera`, `load_scene`, `set_gravity`, `set_timestep` and `reset` refuse while a rollout is driving, because swapping the compiled model or writing the physics arrays under a live rollout is a segfault rather than a race the lock can serialize. The refusal names the robots in flight and the `stop_policy` remedy. Because the population is the one `list_policies_running` reports, the gate and the report cannot disagree about the same instant: a rollout that is *reported* as running is a rollout whose scene is *protected*, in either launch shape - a blocking `run_policy` registers no future, and was previously invisible to the gate while being named by every reporting surface. The rollout's own driving thread is exempt, so a multi-episode rollout still resets between its own episodes (`PolicyRunner` calls `reset()` at each episode boundary); the gate refuses the mutation arriving from *another* thread, which is the hazard.


`list_policies_running` answers on every backend, from that same in-flight population, so the pair above holds wherever a rollout can run rather than only where the population was first read: a MuJoCo, Newton or Isaac engine names the robots it is driving, and a peer polled over the mesh reports the same names at the same instant. It is a `SimEngine` verb reading one seam (`_rollouts_in_flight`, which each backend answers from the rollout claim its own hooks raise and lower), not a per-backend reimplementation, which is what would let the two drift. It is discoverable on every backend too: `describe()["methods"]` names it beside `stop_policy`, read from a single shared entry rather than written once per backend, so a caller enumerating the surface finds the verb wherever it answers. A backend narrows that surface only by hiding a verb it does not implement -- Newton hides `get_contacts` and `load_scene`, both of which raise -- so a name absent from `describe()` means the verb does not work there, not that it was never advertised.

A backend that reports no population at all is refused rather than reported as idle. "No policies running" is an affirmative claim about every robot in the world, and a backend that cannot enumerate its rollouts has no evidence for it - the same reason the state topic omits its `active` flag instead of publishing `false` there. The refusal names the backend and the seam to override, mirroring `stop_policy`, which declines rather than reporting a halt it cannot stand behind. This is reachable today on an Isaac engine before `create_world`: there is no world whose rollouts could be enumerated.

The step horizon is given either as `duration` (seconds) or as `n_steps` (`duration = n_steps / control_frequency`; `n_steps` wins when both are set, and the legacy `max_steps` is an alias for `n_steps`). A non-positive `n_steps` or `control_frequency` is rejected up front with a structured `status="error"` dict naming the bad parameter - `start_policy` validates synchronously before the background rollout starts, so a malformed horizon never returns a false "started" success. `eval_policy` likewise rejects a non-positive `n_episodes`, `max_steps`, or `control_frequency` at the entry point (before `create_policy`), so a typo cannot produce a "successful" evaluation over zero or negative episodes. The same entry-point check covers the two provider keyword bags: `policy_config` (splatted into `create_policy`) and `policy_kwargs` (splatted into `policy.get_actions`) must be dicts, so a `policy_config="host=127.0.0.1"` string returns a structured error naming the parameter instead of a bare `TypeError` from the splat - and, on the `start_policy` path, instead of a false "started" for a rollout that never produced an action. The policy configuration itself is judged there in full, not just the provider name: the provider's own class-level `preflight` hook - the check that refuses a camera the model's declared image inputs cannot be routed from, or an action-chunk count the consumer cannot execute - runs before the submit, so `start_policy` returns exactly the refusal `run_policy` returns for the same request. Nothing reads the worker's result (the future is tracked only to answer whether the robot is busy, and is pruned once done), so a refusal produced there is discarded and `list_policies_running` then reads the same as a rollout that completed. A pre-built `policy_object` skips the check on both surfaces - the provider is unused in that case. Its own shape is checked there instead, for the same reason the bags are: `policy_object` must be a `Policy` instance, so a provider name, a config dict, or the policy class passed in place of an instance returns a structured error naming the parameter rather than a bare `AttributeError` from the first method the runner reaches for - and, on `start_policy`, rather than a false "started" for a rollout that applies no action. The refusal precedes the robot claim, so a rejected call leaves the robot startable. The pair resolves the same way one layer down, on `PolicyRunner.run` - the surface those entry points delegate to, which is documented as drivable directly - and both knobs carry their domain there too, raising `ValueError` rather than returning an error dict because a direct caller has no envelope to read a refusal from. `n_steps` is judged whenever it is *given*, which is the condition the entry point's own resolver judges it on; unvalidated, a step count outside the domain did not fail but handed the horizon to the other knob, so `n_steps=0` ran `duration`'s `10.0`s default - 500 control steps and 500 applied actions for a caller who asked for zero. `duration` is judged only when no step count was given, because that is the only case in which it sets the horizon; unvalidated, `0` and a negative value returned `status="success"` with zero steps and `stopped_reason="budget"` - the field a caller reads to decide whether to retry, asserting a horizon was exhausted when there was none.

`action_horizon` (how many actions are consumed from each policy chunk before it is re-queried) is validated the same way at every entry point, so a horizon the rollout cannot run - `0`, a negative value, a float, `nan` - is a structured error rather than a value silently clamped to 1. `run_multi_policy` additionally accepts per-robot mappings (`instructions={robot: text}`, `action_horizon={robot: horizon}`): a key must name a robot driven by that call (i.e. a key of `policies`), because an unmatched key cannot be applied to anything - a robot omitted from a mapping keeps its documented default. The same domain applies one layer down, on `PolicyRunner.run` / `PolicyRunner.evaluate` - the surfaces those entry points delegate to, which are documented as drivable directly. There it raises `ValueError` rather than returning an error dict, matching the sibling `control_substeps` and `control_frequency` guards of the same signature, because a direct caller has no envelope to read a refusal from; unvalidated, the value was clamped to 1 inside the first chunk query or leaked a bare `int()` conversion error naming neither the parameter nor the method. The two bounds of `PolicyRunner.evaluate`'s own episode loop - `n_episodes` and, on the legacy `success_fn` path, `max_steps` - carry that same domain for a stronger reason: a horizon outside it degrades a rollout, while a loop bound outside it removes the evaluation and still reports one. `n_episodes=0` returned `status="success"` over zero episodes and `max_steps=0` over episodes of zero length, both with `success_rate: 0.0` and `success_measured: true` - the flag that exists so a `0.0` cannot be read as a measurement - and with no action ever applied; `max_steps=inf` never terminated at all, since `while steps < max_steps` has no false case. `max_steps` is checked only when it is the horizon actually read, because a `spec=` call takes its horizon off the benchmark (validated at that read) and never reads the parameter.

The policy can remove the same thing the loop bounds can, and that case is reported the same way. Both eval routes tolerate a policy call that returns an empty action chunk - they advance one physics step so a degenerate policy cannot hang the episode - and that per-step tolerance is right, since a policy may legitimately stall for a step. What it does not decide is the aggregate: when *every* call of an evaluation comes back empty, `send_action` is never reached, and `success_rate` / `avg_reward` / `pass_hat_k` then describe the scene's initial state rather than the policy, under `status="success"` and indistinguishable in every published field from a policy that was exercised and scored zero. `run_policy` already refuses an empty chunk on the first occurrence and `replay` refuses the aggregate of the same condition for a recorded episode whose frames carry no action, so the two eval routes were the only consumers of that condition without a verdict on it. Both now report `actions_applied` beside `steps_advanced` in the result json and per episode - an advanced step is not a commanded action, so one number cannot carry both facts - and refuse the evaluation with `status="error"` when `actions_applied` is zero. A *partial* shortfall is left as a count rather than refused: some empty calls are real policy behaviour, and refusing them would contradict the per-step tolerance above. `actions_applied` counts actions that **commanded** the robot rather than calls made to `send_action`: an action dict with no keys reaches `send_action` like any other and the backend accepts it, but it names no actuator, so applying it commands nothing. A chunk of those actions is not empty, so the empty-chunk tolerance above does not describe it - counting the calls made an evaluation of nothing but those actions indistinguishable, in every published field, from one that commanded every joint (both rollout surfaces now report the tally as `actions_applied` and refuse the aggregate; `run_policy`'s per-actuator `action_resolution_rate` does not cover it, because that map is keyed on the robot's actuators, so a robot declaring none contributes an empty map and a `partial_action_failure_rate` of `0.0` - what a robot with no resolution problem looks like too). A policy reaches that case whenever a decode yields a row with no joint values, as `CuroboPolicy._next_chunk` does when the planner's trajectory rows carry no joint position - the key list it zips against is then empty too, so every waypoint decodes to an action naming no key.

The four **posture** flags in the same signature are held to a domain of their own. `fast_mode`, `reset_between`, `wbc_install_torque_control` and `async_rtc` each select one of two branches rather than scale a quantity - pace the loop at `control_frequency` or run it unpaced, reset the scene between episodes or carry the end state over, install the WBC torque shim for the call or leave the actuators alone, overlap inference with actuation or drain each chunk first - so there is nothing to clamp and no partial effect, and a value that is not a boolean is refused rather than read by truthiness. Every non-empty string is truthy, so `"false"`, `"no"`, `"off"` and `"0"` would select the posture the word asks to skip, while `0`, `""` and `[]` would take the other branch without being a declared spelling of it; read that way, `run_policy(fast_mode="false")` ran unpaced, `run_policy(n_episodes=2, reset_between=0)` started episode two from wherever episode one left the arm, and `run_policy(async_rtc="false")` reported `rtc_async_enabled=True` beside the background inference thread the caller had declined - each with `status="success"`. The domain is the shared `boolean_flag_error` one that the recording postures and the mesh wire schema already use (the wire schema refuses this same `fast_mode` field unless it is a `bool`), bound to the tool-error envelope through `SimEngine._validate_posture_flags` and checked ahead of robot resolution, so a refused call builds no policy and touches no scene. `run_policy` checks all four; `async_rtc=None` is its documented "resolve from the policy" spelling and is checked only when a value is supplied, while `eval_policy` declares `async_rtc` as a plain `bool` and refuses `None` with everything else. MuJoCo's `start_policy` checks `fast_mode` before the submit, for the reason its numeric knobs already do: a refusal produced on the worker is discarded with the future and the caller reads "started". The `run_policy` agent tool checks its own `fast_mode` before it starts the recording it was asked to make, so the facade's refusal cannot arrive after the dataset at `dataset_root` has been replaced with an empty one. Unlike the numeric knobs above, the check sits at the facades only: `PolicyRunner.run` takes these flags as the facades hand them and does not repeat it.

The episode-outcome criterion carries the same posture one step further in. Both eval routes call a caller-supplied criterion after every applied action - `success_fn(observation)` on the `eval_policy` route, `is_success(sim)` / `is_failure(sim)` on the `evaluate_benchmark` route - and a criterion that *raises* is fatal, with a message naming the criterion, the episode and the step and chaining the original exception. That mirrors `run_policy`'s `stop_when`, which is fatal for the same reason: the caller asked for a semantics the runner can no longer honor, and a `success_rate` averaged over episodes whose outcome was never determined is not a measurement. It is deliberately *not* the `on_frame` posture - that hook is best-effort telemetry, so a generic failure is logged and the evaluation continues (a `RecordingFrameError` from it is data loss and propagates). Which surface the failure arrives on is each method's own: `run_policy` converts it to `status="error"` via its terminal handler, while `eval_policy` propagates rollout failures by design, exactly as a raising `get_actions` does. Verdicts are read with `bool()` rather than type-checked, so the NumPy scalar an ordinary predicate returns (`observation["x"] > 0.5` is a `numpy.bool_`, not a `bool`) is accepted unchanged. The best-effort posture covers a hook that *fails*, not one that cannot be called: a non-callable `on_frame` is a caller error and is refused before the first step by `eval_policy` / `evaluate_benchmark` (structured error) and by `PolicyRunner.run` / `PolicyRunner.evaluate` (`ValueError`), the same domain `run_policy` applies to its `observer`. Absorbing it per frame reported the same `TypeError` once per step and still returned a success rate the caller's telemetry had watched none of.

A clause authored in the predicate DSL (`stop_when`, and a benchmark spec's `success` / `failure` / `dense_reward`) is held to a numeric domain when it compiles, because the alternative is a clause that runs and never fires. Every numeric kwarg must be a finite number: `nan` compiles clean and then makes every comparison `False`, so the clause is unsatisfiable and the rollout spends its whole step budget reporting an honest miss. A kwarg that names a **tolerance** - `tol`, `threshold`, or any `*_tol` such as `xy_tol` / `z_tol` - must additionally be `>= 0`, for the same reason by a second route: a tolerance is a bound on a distance, an absolute difference or a squared magnitude, none of which is ever negative, so `{predicate: distance_less_than, threshold: -0.3}` is unsatisfiable rather than loose - and unlike `nan` it reads as a *wider* bound. Signed params keep both signs, because their sign is part of the value: `body_on`'s `z_offset` is an offset a caller lowers below zero to accept a body resting slightly under the reference, `base_velocity`'s `vx` is a velocity component whose sign is a direction, and `body_below_z`'s `z` is a coordinate. A kwarg that names a scene **entity** - a body, a joint, a geom, a container, or `grasped`'s `gripper_prefix` - must be a non-empty string, and this is the one domain whose violation can report a success that never happened. A blank name is invisible to the arm-time probe that catches a typo, because the collector feeding it gathers only non-empty strings, so `body: ""` is never looked up and the term is pinned to a constant for the whole rollout. `gripper_prefix` inverts the constant: it is matched with `startswith`, and the empty string is the identity prefix, so a blank one selects **every** geom in the scene rather than the gripper's - `{predicate: grasped, body: cube, gripper_prefix: ""}` fires on the cube's contact with the floor it was placed on, stopping the rollout on step 1 with nothing moved and scoring a benchmark `success_rate` of `1.0` under `success_measured: true`. A non-string name is refused for the same reason a non-numeric threshold is: unchecked it surfaces as a bare `TypeError` from inside the evaluation loop, naming neither the predicate nor the clause. The optional `robot` selector of the `base_*` family keeps its `None` ("the sole robot"), which is a documented value rather than a missing name. Each of these domains is enforced in `make_predicate`, the one choke point every clause passes through - including the per-stage calls a `staged_reward` compiles by calling back into it - so the refusal names the predicate, the parameter and the value at authoring time rather than one rollout later.


The same reason covers the *names* a clause spells, and it is probed against the live scene rather than the compiler: a `stop_when` clause is walked before the rollout starts and every entity it references is resolved through the exact lookup the predicate uses at evaluation time, so a typo is a structured `status="error"` naming the offending name instead of a rollout that burns its whole budget reporting `stopped_reason="budget"`. Bodies resolve through `get_body_state` (`can_resolve_body`), joints through `get_observation` (`can_resolve_joint`), and the `base_*` family through the floating-base signals `base_pos` / `base_quat` (`can_resolve_base`). The base family is collected by **predicate** rather than by kwarg, because its `robot` kwarg defaults to the sole robot: `{predicate: base_tipped, tol: 0.7}` references a base whether or not it names one, so the spelling that omits the kwarg is probed too. That probe also catches the mistake no name check can - arming a base term on a robot that has no floating base at all. A fixed-base arm reports neither signal, so `base_tipped` on an SO-100 is permanently `False`; it is now refused up front naming the cause, and a mobile base is unaffected. Geom names (`contact_between`) remain uncollected: there is no generic geom lookup on the engine ABC to probe them with. `evaluate_benchmark` runs that same probe over the spec's own `success` / `failure` / `dense_reward` clauses against the scene loaded at the time of the call, for a reason one step sharper than a wasted budget: a term that degrades to a constant makes a success clause unsatisfiable, so every episode scores a miss and the eval reports `success_rate: 0.0` beside `success_measured: true` under `status="success"` - the number a caller publishes, and indistinguishable from an honest policy failure. One character was the whole difference between `1.0` and `0.0` (`body: cube` against `body: cubeee`), and a `dense_reward` term over a missing body went dead at `0.0` the same way. It is refused before the policy is built, so it costs no checkpoint download. Two cases report no entities and are evaluated unchanged, because a pre-eval probe cannot decide either: a spec declaring its own `scene`, whose bodies `on_episode_start` creates only after the probe would have run, and a `DeclarativeBenchmark` built by passing already-compiled `success_fn` / `failure_fn` callables to the constructor - the same contract `stop_when` applies to a callable. A benchmark written in Python rather than the DSL opts into the probe by exposing `referenced_entities()`. The probe covers the clause that can never fire; the opposite corruption is a clause the scene *already* satisfies, and no compile-time or name-resolution check can see it because it is a fact about the initial state rather than about the clause. Both eval routes and `stop_when` sample their condition only *after* an applied action, so a clause already true at the start fires on the first step whatever the policy commands: `run_policy` returns `stopped_reason: predicate` after one step - indistinguishable from a rollout that drove the world there - and a collection loop gating episodes on `stop_when` writes one frame per episode, each tagged as having reached the condition. The blank-`gripper_prefix` case above is one route to it that a name domain can refuse; a `body_above_z` whose height sits below where the object already rests, or a `contact_any` on a body resting on its support, are the same harm through a clause that is entirely well-formed. It is therefore reported rather than refused, the posture `episodes_successful_at_reset` already takes on the eval routes: `run_policy` samples the clause once before the policy acts and reports `stop_when_true_at_reset` (bool, always present) with `stop_when_reset_warning` beside it, the `run_policy` tool aggregates `episodes_stop_when_true_at_reset` over the loop and carries the flag per episode, and every other reported figure is left as measured - domain randomisation legitimately draws an initial state that satisfies a clause on some episodes, so a partial count is a fact about those draws rather than a broken clause. A benchmark spec's `failure` clause is the fourth cell of that grid and the last one without a verdict: unlike a `success` clause it is harmless when it can never fire, but one already satisfied at reset ends every episode on its first step and reports `success_rate: 0.0` beside `success_measured: true` - the same number an honest policy failure reports, reached without the policy. It costs more than the success mirror, because the eval loop reads `is_failure` before `is_success`: the episode is scored a failure with the success criterion never consulted, so a success the policy had already earned on that step is discarded. A "the object fell" height above where the object already rests does it, as does a `base_below_z` collapse line above the robot's spawned stance - which is why the shipped humanoid benchmarks tune that line to each biped's own measured standing height. `evaluate_benchmark` therefore samples the failure clause at the same pre-episode probe and reports `episodes_failed_at_reset` with `reset_failure_warning` beside it, carrying `failure_at_reset` per episode and leaving every figure as measured. `eval_policy` takes a `success_fn` and has no failure criterion to sample, so it reports neither.
The same nesting matters at the episode boundary. `staged_reward` is a registered predicate, so a stage's `reward` may be another `staged_reward` and a curriculum can be authored as a machine of machines. Every such machine carries per-episode state, and `SimEnv.reset` / a benchmark's `on_episode_start` can only reach the terms they hold directly, so each machine clears its own sub-terms - by the same rule, anything exposing a zero-arg `reset()` - and a nested one clears its children in turn. Without that a second episode opens inside a sub-curriculum it has not earned: its earlier shaping signal is never emitted again and its one-time `bonus` is paid once per process instead of once per episode, while the outer phase reports a clean reset either way.

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

**Hardening.** A policy that returns no actions at all on its FIRST query ends the rollout with `status="error"` after that one query, on both the synchronous and the async path - there is no budget a chunk of zero actions can ever spend, so re-querying only burns inference. If a *prefetched* chunk arrives empty the runner instead degrades to one synchronous re-query before erroring (a transient hiccup does not kill an otherwise-healthy rollout). When a prefetch blocks at the seam (inference slower than chunk execution) the runner logs a starvation warning so you can shorten the chunk or fire the prefetch earlier. Set `rtc_inference_timeout_s` to bound a stuck inference: the swap then returns a structured `status="error"` result (carrying the telemetry below) instead of waiting for every remaining chunk - bounded by the single in-flight inference the executor joins on shutdown (Python cannot forcibly kill a running worker thread). That deadline must be a positive finite number of seconds, or `None` (the default) to wait without one - `0`, a negative value and `nan` all make the wait give up before any inference can answer, and `inf` overflows the platform's timestamp arithmetic, so each is refused at the call naming the parameter rather than reported one rollout later as a stuck policy.

**Telemetry.** Every `run_policy` result `{"json": {...}}` block carries the chunk-prefetch fields so latency masking is provable from the payload, not the logs. The prefetch pipeline runs for any chunk-emitting policy (an ACT checkpoint included); only `policy_rtc_enabled` says whether the policy blended the seams with real-time chunking:

| Field | Meaning |
|-------|---------|
| `chunk_prefetch_enabled` | Whether the overlap pipeline ran (the resolved `async_rtc`) |
| `policy_rtc_enabled` | The policy's own `supports_rtc` - real seam blending, independent of the pipeline |
| `chunk_prefetch_chunks_acquired` | Chunks the rollout acquired (cold start + swaps + re-queries), counted on the synchronous path too |
| `chunk_prefetch_hits` | Seams where the next chunk was already computed (stall hidden) |
| `chunk_prefetch_blocks` | Seams where the runner had to wait for inference (seam starved) |
| `avg_inference_ms` | Mean `get_actions` wall time across the rollout |
| `max_inference_ms` | Slowest `get_actions` wall time |

The previous `rtc_async_enabled`, `rtc_chunks_acquired`, `rtc_prefetch_hits`, `rtc_prefetch_blocks`, `rtc_avg_inference_ms` and `rtc_max_inference_ms` spellings are still emitted with the same values for one release.

A healthy masked rollout shows `chunk_prefetch_hits` near the chunk count and `chunk_prefetch_blocks == 0`; persistent blocks mean inference is slower than chunk execution and the seam cannot be fully hidden.

**Async-RTC in `eval_policy` (opt-in).** The success-rate eval path (`eval_policy` / `evaluate(success_fn=...)`) accepts the same `async_rtc` and `rtc_inference_timeout_s`, but defaults to `async_rtc=False`. The synchronous eval pauses the world during inference, so the success-rate is bit-stable and reproducible (the policy always sees the seam observation). Setting `async_rtc=True` evaluates a chunk-emitting policy under the realistic control latency it faces in deployment: the prefetch feeds the policy a slightly staler (mid-chunk) observation at the seam, so the measured success-rate can shift - that is the point, it measures robustness to inference latency. Either way the eval `{"json": {...}}` payload now carries the same six `rtc_*` fields (inference timing is reported even on the synchronous path). `async_rtc=True` is rejected on the benchmark/spec path (`evaluate_benchmark` / `evaluate(spec=...)`), which stays synchronous for bit-stable reproducibility; use `run_policy(async_rtc=...)` for benchmark-style wall-clock latency masking. Being synchronous, the spec path declares an observed delay of exactly `0` to the policy before every inference, like the other two loops - so a policy object carried over from an async rollout cannot keep slicing its chunk seam against that rollout's stale step count.

`run_policy` returns a `{"json": {...}}` content block alongside the human-readable `text`, mirroring `eval_policy`. The json block carries the rollout facts as typed fields - `robot_name`, `policy`, `instruction`, `n_steps`, `elapsed_s`, `stopped_early`, `action_errors`, `video_path` (`None` when no MP4 was written), `video_frames`, `video_fps` (the rate the MP4 *plays* at - the requested `fps` capped to `control_frequency`, since a rollout renders at most one frame per control step, so `video_frames / video_fps` is the file's real length), `sim_time_s` (when the backend reports it) and the six `rtc_*` async-RTC telemetry fields above - so an agent can read the outcome programmatically (did it move? how many steps? was inference masked?) without regex-parsing the prose. `elapsed_s` is measured on a monotonic clock, so it is the time that actually elapsed rather than the difference between two readings of a date that a correction can move. The `status` reflects whether the robot *moved*, not merely whether every key resolved: a run where **no** step resolved any key (the robot never moved) returns `status="error"`, and so does a run where no step named a key in the first place - the same physical outcome reached without producing an unresolved key to count, which the payload reports as `actions_applied: 0`. That second route is how a robot whose model declares no `<actuator>` block behaves (the shipped `asimov_v0` description compiles that way, so `robot_action_keys` reports zero keys and a policy bound to that list can only emit empty actions - `actuate_robot` adds a position servo per joint to such a model, after which `robot_action_keys` names those servos and a policy can drive them), and how a policy whose decode yields rows carrying no joint value behaves on a fully actuated robot. A run where some keys resolve every step - e.g. a policy trained on a superset embodiment that emits one extra key the robot lacks - is operational and returns `status="success"` with a non-fatal `N/M action steps had unresolved keys` note and a `partial_action_failure_rate`.

At `n_episodes > 1` the same call returns an aggregate, and the aggregate keeps every field above whose value the call already knows. That is the identity fields, the policy-binding flags (`positional_fallback_used`, `generic_state_keys_used`, `missing_state_keys_used`) and the policy-load telemetry (`policy_load_time_s`, `policy_load_cache_hit`, `policy_resident_rss_mb`) - all read off the ONE policy object every episode ran with, which is how `eval_policy` and `evaluate_benchmark` already report them for their own multi-episode aggregates. Two of those fields are only meaningful across a loop in the first place: a `policy_load_cache_hit` of `false` on episode 2+ means the caller rebuilt the policy instead of reusing `policy_object=`, and a flat `policy_resident_rss_mb` is what says the model stayed resident rather than reloading per episode. The binding flags matter most here for a different reason - the multi-episode shape is the one that collects a dataset, and a `true` flag means those episodes recorded a robot moving on meaningless inputs while `status` stayed `"success"`. Read them the same way at any episode count.

What the aggregate adds is `total_steps`, `stopped_reasons` (aligned with `episodes`), `video_paths` and the per-episode `episodes` records; `steps_used` equals `total_steps` and `stopped_reason` is the last episode's. Per-episode action health is *not* summarised, because a per-step rate has no single aggregate an N-episode call can report without choosing one - each record in `episodes` carries its own `action_errors`, `action_resolution_rate` and `partial_action_failure_rate`, so the worst episode is `max(e["partial_action_failure_rate"] for e in report["episodes"])`.

### Watching a rollout: the `observer` lane

`run_policy(observer=...)` takes a read-only callable that receives one `RunPolicyStarted`, one `RunPolicyStep` per completed `send_action` call, and one `RunPolicyEnded`. A complete backend breakdown says what physically applied; a coarse error keeps that state explicitly `unknown`. It is a *second* lane beside the backend's `on_frame` hook, not a use of it - that hook is filled from `_make_run_policy_hook` (cooperative cancellation, the trajectory mirror, mesh step telemetry, dataset recording) and is not available to callers, so supplying one would remove all of that rather than add observation.

```python
from strands_robots.simulation.observers import RunPolicyStep

def watch(event):
    if isinstance(event, RunPolicyStep) and event.action_resolution != "full":
        print(event.applied_action_index, event.unresolved_action_keys)

sim.run_policy(robot_name="alice", policy_provider="mock", observer=watch)
```

The events use observer schema version **2** and report four things `on_frame`'s `(step, obs, action)` signature cannot carry:

| Field | Why it is not derivable from `on_frame` |
|-------|------------------------------------------|
| `action_resolution` | The backend's per-key `send_action` verdict, normalised to `full` / `partial` / `none` / `unknown`. `partial` and `none` require a valid, complete per-key breakdown; a coarse backend error is `unknown` with empty explicit key tuples, because input keys are not proof of what reached physical state. Coarse steps remain in `action_errors` and result text but are excluded from aggregate action-rate denominators rather than counted as physical misses. |
| `observation_is_chunk_reused` | Narrow chunk-position signal: `true` only for a later action using the same chunk-start snapshot. It is not authoritative freshness, because the first action after an async prefetch swap can already use an old snapshot. |
| `observation_age_steps` | Authoritative nonnegative age in control-step terms: completed rollout action attempts since the snapshot was sampled. Sync chunks report their chunk index. Async chunks carry the prefetch sample's remaining-old-chunk-attempt count across the swap and add the new chunk index. An active recording refreshes every step and reports `0`. On an `unknown` action resolution this field does not claim physical advancement. |
| `legacy_hook_outcome` | What the backend's hook did - `ok`, `cancelled`, `recording_error`, `error`, or `absent`. |

`applied_action_index == legacy_step_index` for every `RunPolicyStep`, including a cancelled or recording-failed step: both identify the same zero-based action passed to the hook. The abort is identified by `legacy_hook_outcome`, not by an index offset. Terminal counts have a different boundary: the hook runs *after* `send_action` and `step_count` increments *after* the hook, so `RunPolicyEnded.applied_actions` can exceed `legacy_steps_used` by one when that final hook aborts.

Ordering is explicit: `event_seq` is dense and 0-based within one `run_id`, so a gap is observable; `monotonic_ns` orders the stream (no NTP correction or `date -s` can move it) and `utc_ns` is derived from a single rollout anchor so a wall-clock label can never disagree with it. At `n_episodes > 1` each episode is its own lifecycle with its own `run_id`. Once dispatch of `RunPolicyStarted` is attempted, dispatch of exactly one `RunPolicyEnded` is attempted on every Python exit, including process-control/cancellation exceptions and result-assembly failures. Those non-cooperative exceptions retain their identity and traceback and still propagate. If Step or Ended observer dispatch raises a second non-cooperative exception while one is already unwinding from the legacy hook or rollout, the original remains primary and the secondary failure is logged and attached as an exception note. A preflight refusal opens no lifecycle. `sim_time_s` is read only from a cached `_world.sim_time` or engine `_sim_time`; observation telemetry never calls `get_state`.

Five rules the lane holds to, and one it does not:

- **Additive.** Installing an observer changes no action applied, no existing result-json field, and no observation or render call. It adds one key, `observer_failures`.
- **Contained, for the classes it is an observer's business to raise.** An `Exception` never alters the rollout outcome and never reaches the `max_onframe_failures` watchdog - that exists for a recorder losing dataset frames, not a visualiser that cannot draw. `CooperativeStop` is contained too, and by name: it is a `BaseException` precisely so a hook's broad `except Exception` cannot swallow a cancellation, so without naming it here any observer could cancel a rollout it is only supposed to watch. Every contained failure is counted and reported as `observer_failures`, so a stream with holes says so.
- **Not contained: the four signals that are nobody's telemetry.** `KeyboardInterrupt`, `SystemExit`, `GeneratorExit` and `asyncio.CancelledError` propagate. None of the four is an `Exception` subclass, so the guard's `(CooperativeStop, Exception)` clause passes them through by construction. A generator closed underneath a visualiser, or a task cancelled while one was drawing, is a real teardown rather than a drawing failure, and reporting it as `observer_failures` on a rollout that then ran to its full budget said the opposite. If Step or Ended dispatch raises one while another exception is already unwinding from the legacy hook or rollout, the original exception remains primary; the secondary observer failure is logged and attached to it as a note.
- **Borrowed, not copied.** `observation` and `action` are the same objects the hook received. Treat them as read-only and do not retain them past the call; snapshot what you need synchronously.
- **Not isolated.** Dispatch is synchronous on the rollout thread, so a blocking observer blocks the robot. This is telemetry, not a sandbox. The rollout loop paces on a *deadline* rather than a delay, so a consumer has a budget of one control period (`1 / control_frequency`) that costs the rollout no wall clock at all - work inside it is absorbed by the period instead of added to it. Overrun that budget and the pace is what gives: the loop drops the missed deadline rather than firing a burst of catch-up actions at the arm, so the arm sees a gap. Keep the callback short and hand anything slower to another thread.

Scope: `run_policy` (including its `n_episodes > 1` path) and `PolicyRunner.run`. `eval_policy`, `evaluate_benchmark` and `run_multi_policy` are separate loops with different step semantics and carry no observer yet.

`eval_policy` accepts the same `video={...}` recording config as `run_policy` (`path` enables it, plus `fps` / `camera` / `width` / `height` - an unknown key, a non-positive size, or one field spelled twice with two different values (`camera` and its legacy `camera_name`) is a caller error, never silently ignored), but writes **one MP4 per episode** with `_ep{i}` inserted into the filename (`eval.mp4` -> `eval_ep0.mp4`, `eval_ep1.mp4`, ...), so a multi-episode evaluation can be *watched* to see why episodes fail rather than only read as an aggregate `success_rate`. The written files are listed in the result json `video_paths`; the output path is validated and the camera probed up-front, so a bad camera fails the eval immediately instead of after N episodes of empty MP4s. `evaluate_benchmark` accepts the same `video={...}` config and records one MP4 per episode too, so a benchmark evaluation can be watched to see why episodes fail. Frames are captured synchronously on the eval thread (render is read-only over `mjData`), so recording does not perturb the bit-stable benchmark rollout.
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
| `stop_recording(push_to_hub=False, bucket=None, run_id=None)` | Finalise dataset (flushes any trailing rollout) |
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
| `list_urdfs` | Built-in robot table, plus a `Registered URDFs:` section naming every `register_urdf` asset and whether it resolves |
| `register_urdf(name, path)` | Register additional asset - it is named by `list_urdfs` from then on |
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
