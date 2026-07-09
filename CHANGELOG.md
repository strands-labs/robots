# CHANGELOG

All notable behavioural changes to `strands-robots` are logged here. Follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

### Fixed: scene mutations silently swallowed a cached-XML refresh failure

Every MuJoCo scene mutation keeps a legacy XML string in
`_backend_state["xml"]` in sync with the live `MjSpec` so the `load_scene` +
`add_robot` round-trip can read it. That refresh calls `spec.to_xml()`, which
can fail on specs MuJoCo cannot serialise. Two of the four mutation paths
(`replace_scene_mjcf` and `patch_scene_mjcf`) wrapped it in a bare
`except Exception: pass`, so a failure left the cache stale and diverged from
the live spec with no diagnostic -- a subsequent XML-cache reader would see
outdated scene contents and nothing explained why. The other two paths
(`add_object`/recompile and `eject_robot_from_scene`) already logged the reason.
All four now funnel through one `_sync_cached_xml` helper that logs the failure
at debug and leaves the prior cache intact: still never fatal (the live
model/spec are already updated), but no longer silent.

### Added: remote policy inference (client/server split) for edge robots + remote GPU

A resource-constrained robot host (edge device / laptop CPU) often cannot run a
large VLA (pi0, SmolVLA, MolmoAct2) at control rate. The new
`strands_robots.inference` package splits inference across two machines over a
portable WS-JSON WebSocket protocol: `PolicyServer` wraps ANY `Policy` and
serves it (run it on the GPU box), and `RemotePolicy` is a drop-in `Policy` that
forwards observations to the server and returns the action chunk (construct it
on the robot host). `RemotePolicy` is wired into `create_policy` as the `remote`
provider, so `create_policy("remote", endpoint="ws://gpu-box:8765")` -- or the
smart string `create_policy("ws://gpu-box:8765")` -- yields one, usable anywhere
a local policy is (`run_policy`, `eval_policy`, hardware loops). The client
mirrors the server policy's `requires_images` / `execution_horizon` /
`actions_per_step` / `supports_rtc` metadata, and the Real-Time Chunking
contract is preserved end to end: the runner-counted `rtc_observed_delay_steps`
is forwarded per request and applied server-side before chunk-seam blending.
Install with `pip install 'strands-robots[inference]'` (pulls only
`websockets`). See `docs/inference/remote.md`.

### Added: regression guard that the lerobot policy resolver covers every registered policy family

The `lerobot_local` policy-class resolver enumerates policy types dynamically:
`_ensure_policy_configs_registered` walks `lerobot.policies` at import time and
lets each `@PreTrainedConfig.register_subclass("<type>")` decorator populate
lerobot's draccus choice registry, which `list_policy_types()` reports. This
means a new policy family a future lerobot release ships is picked up with no
strands-side edit -- but nothing pinned that the dynamic walk actually reaches
every registered family, so a future lerobot layout change that made the walk
under-populate would surface only as a runtime `create_policy(policy_type=...)`
resolver miss for a user. A new version-agnostic test derives the ground-truth
set of families directly from the installed lerobot source (the
`register_subclass` decorator arguments under
`lerobot/policies/*/configuration_*.py`) and asserts `list_policy_types()`
covers all of them. It never hard-codes a policy count or name list, so it
tracks whatever lerobot the environment resolves (currently 19 families in
lerobot 0.6.x). Verified the resolver already covers the full lerobot 0.6.x
roster; `rtc` (an inference-time action-chunking wrapper, `RTCProcessor`) is
correctly excluded because it is not a `PreTrainedPolicy` and registers no
config.

### Fixed: remote inference delivered read-only observation arrays to the wrapped policy

`RemotePolicy` / `PolicyServer` decoded NumPy observations (camera frames, the
state vector) with `np.frombuffer` over the immutable `bytes` returned by
`base64.b64decode`, so every array handed to the server-side policy was
read-only. That silently diverged from the local-inference path, where a freshly
rendered observation is always writable: a VLA preprocessor that normalizes the
image/state in place (`image /= 255`, joint rescaling) raised
`ValueError: output array is read-only`, and `torch.from_numpy(obs)` (zero-copy,
as pi0/SmolVLA/MolmoAct2 pipelines do) produced a tensor whose in-place mutation
is undefined behavior. Decoding now wraps the bytes in a mutable `bytearray` so
the reconstructed array is writable (one allocation, no extra copy, still
byte-exact), making the remote path transparent to the wrapped policy. The
MuJoCo rollout test used a `MockPolicy` (`requires_images=False`) so it never
decoded an observation array and missed this.


### Fixed: session `status` tool results dropped their telemetry via a `**spread` top-level smuggle

The `status` action of `lerobot_teleoperate` and `lerobot_train` returned the
session telemetry (`session_name`, `pid`, `uptime`, `is_running`, and every
key of the persisted `session_info`) as *extra top-level keys* on the result
dict via `**session_info`. Per the tool-result contract only `status` and
`content` survive into the agent turn, so an agent that asked for session
status received only the human-readable text block -- the structured fields
it needs to decide whether a session is healthy were silently dropped by the
runtime. The telemetry now rides in a `{"json": {...}}` content block. The
static contract guard missed this because a `**spread` key made it skip the
whole dict; it now flags any `**spread` inside a tool-result-shaped dict as a
violation, closing the smuggling loophole.

### Fixed: `get_energy` reported the energy of a stale pose after a direct `qpos`/`qvel` write

`get_energy` called `mj_energyPos` / `mj_energyVel` on whatever derived state
happened to be in `data`, without first recomputing the forward pipeline.
Potential energy is a function of `data.xipos` (inertial body positions in the
world frame) and kinetic energy of the config-dependent inertia `data.qM`, all
of which are position-stage derived state that a bare `qpos`/`qvel` write does
not refresh -- so after a direct `data.qpos` write (a planning/IK loop) or
`set_joint_velocities`, `get_energy` silently returned the energy of the
*previous* configuration. It now runs `mj_forward` under the sim lock before
reading, matching the defensive forward already in `get_mass_matrix` and
`inverse_dynamics`. The explicit `mj_energyPos`/`mj_energyVel` calls are kept
because `mj_forward` only recomputes `data.energy` when `mjENBL_ENERGY` is
enabled (it is not by default), whereas the explicit calls populate it
unconditionally.

### Added: `run_policy(policy_object=...)` on the hardware `Robot` -- sim parity for pre-built policies

The simulation side has a one-call rollout for a policy constructed in-process
(`Simulation.run_policy(policy_object=..., n_steps=...)`), but the hardware
path's only control loop (`start_task`) builds its own server-backed policy
from `policy_provider` + `policy_port` and accepts no policy object -- so
deploying a local checkpoint on an edge device (in-process policy, no server
on a port) meant hand-writing the connect -> observe -> get_actions ->
send_action loop the library already contains. The hardware `Robot` now has
`run_policy(policy_object=..., instruction=..., duration=..., n_steps=None)`
(blocking): it drives the caller's object through the exact `start_task` loop
(connect with half-open-port rollback, state-key initialization, the RTC
control-frequency / observed-delay contract, `resolve_chunk_length` chunk
consumption), stops at `duration` or after `n_steps` applied actions (the sim
parameter), and returns a text + `{"json": ...}` result per the tool-result
contract. `start_task` and the provider+port path are unchanged.

### Fixed: `TrainSpec.learning_rate` was silently ignored by the `lerobot_local` trainer

`learning_rate` was the one universal `TrainSpec` field with zero references in
`training/lerobot.py`: `build_config()` never wired it into the typed config and
`build_command()` emitted no `--policy.optimizer_lr` flag, so the policy's
training preset always won and a caller-set value was silently dropped (every
other backend honored it). The field is now opt-in like `seed` -- it defaults to
`None` (use the backend's own default: the policy preset for LeRobot, the
`FinetuneConfig` default for GR00T, the TOML default for Cosmos), and an
explicit value is honored everywhere. For LeRobot it maps to
`policy.optimizer_lr`; a policy with no such field (no Adam-style single LR)
raises loudly instead of dropping the value. RL trainers (PPO/FastSAC) keep a
concrete `1e-4` default on `RLTrainSpec`, since from-scratch RL has no preset to
defer to. The `train_policy` tool's `learning_rate` parameter defaults to `None`
to match.

### Added: ROS 2 action support in `use_ros` + goal-level `navigate_to` on `RosBridgedRobot`

`use_ros` gains `list_actions` and `action_send_goal` (in-process `rclpy.action`,
dynamic `get_action` type resolution, one end-to-end timeout budget). Timed-out
goals are cancelled before the error returns, never orphaned on the robot;
feedback is capped at 5 samples (first 4 + latest) to protect agent context.
`RosBridgedRobot` gains `nav_action` wiring and `navigate_to(x, y, yaw)`,
exposed as a `navigate_<node_name>` agent tool only when configured.

### Fixed: `examples/train_ppo_reach.py` aborted in its own `validate()` preflight

The example shipped `rollout_steps=250` with `num_mini_batches=4`, but PPO spec
validation requires `rollout_steps % num_mini_batches == 0` (a non-divisible
split would silently drop samples), so the reference PPO entry point exited
with `spec invalid` before training ever started. The example now uses
`num_mini_batches=5` (`250 % 5 == 0`, 50 samples/mini-batch).

### Fixed: registry hot-reload ignored the user-overlay file's mtime

The `robots` registry the read API serves (`get_robot` / `list_robots` /
`resolve_name`) is the package `robots.json` merged with the user-local overlay
`$STRANDS_BASE_DIR/user_robots.json` (`loader._merge_user_robots`). The loader's
mtime hot-reload -- documented to "re-read when the source changes" -- keyed its
cache on the *package* file's mtime only. `register_robot` / `unregister_robot`
invalidate the cache explicitly, so the in-process path was fine, but any other
writer (a second process, a manual edit, or a tool that writes the overlay
directly) was silently ignored: the merged result stayed stale until the package
mtime happened to change or `invalidate_cache()` was called by hand.

The cache-validity signature now tracks the mtimes of *both* files for the
`robots` registry, so external create / modify / delete of `user_robots.json` is
picked up on the next read -- honoring the hot-reload contract for the overlay
just as for the package file, while still avoiding a re-read (only two `stat()`
calls) when nothing changed. A public `registry.user_registry_mtime()` helper
keeps the overlay path a single source of truth.


### Fixed: examples defaulted `MUJOCO_GL` to the macOS-only `cgl` on every platform

`07_post_tune_any_policy.py`, `06_agent_collect_and_stream.py`,
`lerobot_hardware_catalog.py` and `vla_g1_workflow.py` ran
`os.environ.setdefault("MUJOCO_GL", "cgl")`, a value only valid on macOS -- on a
bare Linux box the first offscreen render died with `RuntimeError: invalid value
for environment variable MUJOCO_GL: cgl`. The default is now platform-aware
(`cgl` on macOS, headless-safe `egl` elsewhere, matching what
`04_mesh_peer_discovery.py` and the cosmos3 examples already do); a
user-exported `MUJOCO_GL` is still never overridden.

### Fixed: `move_object` silently no-op'd on static objects

`Simulation.move_object(name, position=...)` moves an object by writing its
freejoint `data.qpos`. Static objects (`is_static=True`) are welded to the
worldbody and have no freejoint, so the write was skipped -- but the method
still returned `{"status": "success", "text": "'<name>' moved to ..."}` while
the body never moved and its stored `SimObject.position` stayed stale. This is
the "success contract, no physical effect" failure mode the project forbids.

`move_object` now repositions static objects by editing the body pose in the
live `MjSpec` and recompiling the scene (preserving other joints' state), via a
new `scene_ops.reposition_body_in_scene` helper -- mirroring how `add_object` /
`remove_object` mutate the scene. Dynamic objects keep the cheap `data.qpos`
path. A static-body recompile failure now reports `status="error"` rather than a
misleading success.

### Added: `[ollama]` extra -- the local-LLM agent path installs out of the box

`strands.models.ollama.OllamaModel` does a top-level `import ollama`, but no
strands-robots extra pulled the client, so the no-cloud agent path (the natural
choice on an edge device) failed with `ModuleNotFoundError: No module named
'ollama'` on a fresh install. The new `[ollama]` extra delegates to
strands-agents' own `[ollama]` extra (bounded pin, `ollama>=0.4.8,<1.0.0`) to
keep a single source of truth for the client version, and is included in
`[all]`. Opt-in; the base install is unchanged.

### Fixed: `replay_episode` / `send_action` vector bound to joints, not actuators

`PolicyRunner.replay` (`Simulation.replay_episode`) mapped a recorded
`LeRobotDataset` action vector positionally onto `robot_joint_names`, and
`SimEngine._coerce_action` bound a raw numeric action vector passed to
`send_action` the same way. But the dataset recorder writes the `action`
column in the robot's *actuator* order (`robot_action_keys`), and `send_action`
resolves actuator keys -- these diverge from the joint names whenever a robot
has passive/mimic joints with no driving actuator or a tendon-driven gripper
(e.g. `aloha`: 14 actuators vs 16 joints; `xarm7`; dexterous hands).

The result was a silent record->replay round-trip corruption: replaying a
dataset recorded by `start_recording` shifted the action vector across the
wrong DOFs and dropped the grippers (their names have no matching joint), while
`replay_episode` still returned `status="success"`. This is the "success
contract, no physical effect" failure mode the project forbids, and mirrors the
policy-keying fix already applied to `run_policy`/`eval_policy`.

Positional action-vector binding now uses `robot_action_keys` in both paths, so
a self-recorded episode round-trips onto the actuators it was recorded from. For
robots whose actuators mirror their joints (e.g. `so101`) behaviour is
unchanged. Pass `action_key_map=` to `replay_episode` to override the ordering
for third-party datasets.

### Fixed: `eval_policy` no longer reports a silently-meaningless `success_rate`

`eval_policy` / `PolicyRunner.evaluate` default `success_fn=None`. With no
success criterion (and no benchmark spec) an episode can never be marked
successful, so the loop reported a hard `success_rate: 0.0` for every episode
regardless of what the policy did - a value indistinguishable from a policy
that genuinely failed every episode, emitted with no warning. Callers
comparing checkpoints saw `0.0` for all of them and (correctly) concluded
nothing, while the number read like a real measurement.

The no-criterion path now logs a warning, annotates the human-readable text
(`... [no success criterion - not measured]`), and adds a `success_measured`
boolean to the returned json (`False` on the legacy path with no `success_fn`,
`True` on the `success_fn`/benchmark-spec paths). `success_rate` stays numeric
(`0.0`) for backward compatibility; check `success_measured` before trusting
it. This extends the entry-point guards that already reject other
fabricated-success-rate configs (non-positive `n_episodes`/`max_steps`, empty
goal payloads).

### Fixed: warn when a checkpoint's normalization pipeline is present but inert

`lerobot_local` warned when a checkpoint shipped *no* `policy_postprocessor.json`
(actions emitted in raw space). It stayed silent for a subtler, equally broken
case: a pipeline that IS present and active but whose normalizer stats do not
cover the declared `action` / `observation.state` keys. LeRobot's
`NormalizerProcessorStep` returns a tensor unchanged when its lookup key is
absent from the loaded stats, so such a pipeline normalizes NOTHING -
`observation.state` reaches the model raw and predicted actions reach the robot
without unnormalization - while `has_postprocessor` stays `True`, so the
existing guard never fired.

Pretraining *base* checkpoints hit this: `lerobot/smolvla_base` ships stats
keyed by the training dataset (`so100.buffer.action`) with no
`observation.state` stats and no bare `action` key, so both state normalization
and action unnormalization silently pass through. The result looks like an
out-of-distribution / proprioception-ignoring policy with no diagnostic.

`ProcessorBridge.inert_normalization_features()` now reports declared,
non-IDENTITY normalization features that will be skipped, and the load path
warns (pointing at fine-tuning or `processor_overrides` stats) - matching the
no-silent-passthrough intent of the missing-postprocessor warning. Diagnostic
only; no change to action values.

### Fixed: `dataset_cameras` scoping crashed on the Newton backend

`run_policy(dataset_cameras=[...])` scopes a recorded dataset to a chosen
subset of cameras by forwarding `start_recording(cameras=...)`. Only the
MuJoCo backend accepted that keyword; the Newton backend's `start_recording`
had no `cameras` parameter, so a scoped rollout on a Newton-backed simulation
raised an uncaught `TypeError: start_recording() got an unexpected keyword
argument 'cameras'` out of the otherwise backend-agnostic `run_policy` tool.

`NewtonSimEngine.start_recording` now accepts `cameras=` with the same
semantics as the MuJoCo backend: `None` records every named scene camera,
a subset scopes the dataset schema (and the per-step capture hook) to exactly
those views, names may be given in raw or schema-safe (`/` -> `__`) form, and
an unknown name fails loudly listing the available cameras. `dataset_cameras`
now behaves identically on both engines.

### Fixed: the Newton recorder dropped a floating base's state from the dataset

A floating-base robot (a humanoid or a mobile base) exposes its full base
kinematics via `get_observation` on both backends -- position (`base_pos`,
world x/y/z including height), orientation (`base_quat`, w/x/y/z), linear
velocity (`base_lin_vel`, m/s) and angular velocity (`base_ang_vel`, rad/s).
The MuJoCo recorder preserves these as per-component `observation.state`
columns, but the Newton recorder derived its schema from scalar joint names
only and merely *warned* that the base state was being dropped -- so a
dataset recorded on the Newton backend was silently base-blind, and a
locomotion / velocity-tracking / whole-body-control policy trained on it was
missing exactly the base state it needs.

`NewtonSimEngine.start_recording` now preserves the floating base as the same
per-component scalar columns (`base_pos.x` .. `base_ang_vel.z`) as the MuJoCo
backend, reusing the `DatasetRecorder` `extra_state_specs` machinery that
flattens the vector observation keys into `observation.state` each frame.
Multi-robot base columns are namespaced (`alice__base_quat.w`) to match the
recorded observation keys, a resumed dataset is validated against the full
(joints + base) schema, and a fixed-base arm gains no base columns (the schema
is unchanged). The now-superseded base-state-dropped warning is removed from
both backends.

### Fixed: `render_depth` ignored a camera's configured resolution

`render()` and `render_all()` honor a named camera's configured resolution
(from `add_camera(width=, height=)`) when the caller omits `width`/`height`, so
the rendered frame matches the camera - and the recorded dataset - it belongs
to. `render_depth()` was left out of that contract: it always fell back to the
engine default (640x480) for a named camera, so `render_depth("cam")` came back
at a different size than `render("cam")` for the same camera. Any depth-aware
consumer pairing the RGB frame with the depth map per pixel got mismatched
dimensions.

`render_depth()` now resolves resolution exactly like `render()`: omitted dims
-> the named camera's configured resolution; explicit dims -> override; the
free/model-only cameras -> engine default. RGB and depth for the same camera
are now pixel-aligned.

### Added: `set_obs_noise` on the MuJoCo backend (sensor-noise parity with Newton)

`set_obs_noise` is a declared `SimEngine` method for additive sensor
measurement noise (a sim-to-real robustness lever), and the Newton backend
implemented it, but the default MuJoCo backend inherited the base
`NotImplementedError`. Robustness code that ran on Newton crashed on MuJoCo -
the reference backend - and the two engines diverged on a shared contract.

The MuJoCo backend now implements `set_obs_noise(joint_pos_std, joint_vel_std,
camera_jitter_px, seed)` with the same semantics as Newton: joint-position and
per-joint-velocity Gaussian noise applied on `get_observation` (positions, the
`<joint>.vel` entries, and camera frames) and `get_robot_state`, plus integer
pixel jitter on rendered frames. Values must be finite and non-negative
(`status=error` otherwise). The noise stream is seedable/reproducible, and the
default (never-configured) path is an exact no-op, so unconfigured observations
and renders are byte-for-byte unchanged. `describe()` now advertises the method.

### Added: `list_cameras()` on the MuJoCo backend -- camera discovery at backend parity

The Newton backend exposed a public `list_cameras()` and advertised it in
`describe()`, but the default MuJoCo backend had no such method: `sim.list_cameras()`
raised `AttributeError`, and `describe()["cameras"]` was built from a raw
`model.ncam` loop whose contents depended on whether the loaded MJCF happened to
bake a camera literally named `"default"`. The two backends therefore reported
different camera sets for the same query, and the built-in `"default"` free view
that `render()` (its default argument) always accepts could be absent from
discovery.

`MuJoCoSimEngine.list_cameras()` now returns every renderable camera name --
`"default"` first, then all model-defined and `add_camera` cameras, deduplicated
-- mirroring Newton. `describe()["cameras"]` delegates to it (so the two are
always equal and `"default"` is always advertised), `describe()["methods"]`
documents it, and `list_cameras` is a dispatchable agent action alongside
`list_robots` / `list_objects` / `list_bodies`, completing the scene discovery
surface on the default backend.


### Fixed: `inverse_dynamics` reported ~0 torques regardless of pose (stale `qacc`)

`inverse_dynamics()` runs `mj_inverse`, which reads `data.qacc` as the
*desired* acceleration. The method used whatever value was left in `qacc` by
the previous forward pass -- the unforced free-fall acceleration -- and so
asked `mj_inverse` to reproduce free-fall, which needs ~0 generalized force.
It therefore returned near-zero torques at every pose instead of the
gravity-/velocity-bias-compensation torques the query exists to provide (the
standard "hold this configuration" inverse-dynamics result). It now runs
`mj_forward` first (so the position/velocity kinematics match the current
`qpos`/`qvel`, matching the defensive forward in `get_mass_matrix`), zeroes
`qacc` for the solve, and restores the buffer afterwards, so the result is
correct and independent of any leftover acceleration.

### Fixed: `get_jacobian` / `get_body_state` returned values for a stale configuration

Both physics-query methods read derived MuJoCo state without first running the
forward pipeline. `get_jacobian` reads `data.xpos`/`site_xpos`/`geom_xpos`,
`data.subtree_com` and `data.cdof`; `get_body_state` reads `data.xpos`/`xquat`/
`xmat`/`xipos` and, via `mj_objectVelocity`, `data.cvel`. All of these are only
recomputed by a forward pass, so after any state change that did not itself
forward -- a direct `data.qpos` write, or `set_joint_velocities` (which writes
`qvel` without forwarding) -- the methods silently returned the Jacobian /
pose / 6D velocity of an *earlier* state while reporting `status="success"`.
`get_body_state` even disagreed with its sibling `forward_kinematics` on the
same body at the same `qpos`. `get_jacobian` now recomputes the position
pipeline (`mj_kinematics` + `mj_comPos`, matching `forward_kinematics`) and
`get_body_state` runs a full `mj_forward` (matching `get_mass_matrix` /
`inverse_dynamics` / `get_sensor_data`), so both always reflect the current
`qpos`/`qvel`.

### Fixed: `replay_episode` under-integrated physics (single dt/frame instead of a full control period)

`PolicyRunner.replay` advanced physics by only a single `send_action` default
`n_substeps=1` (~2 ms) per recorded frame, while each recorded frame represents
one control step taken at the dataset's fps and the recording itself integrated
a full `1/fps` control period per action (the substep convention `run()` and
`evaluate()` already derive via `_control_substeps`). On a position-servo robot
this meant replay got ~1/17 of the integration time per target at 30 Hz control
(2 ms physics dt), so the servo could not track the recorded targets: replay
produced a heavily under-integrated, attenuated trajectory (an SO-101 self-record
diverged ~0.95 rad from its recording and its joint sweeps were attenuated up to
11x) that did not reproduce the recording, while still reporting `Frames: N/N`
and `status="success"` -- a silent record -> replay fidelity gap. Replay now
derives its physics substeps from the dataset fps and steps a full control
period per frame, so a recorded episode replays back to the pose it was recorded
at. `speed` continues to scale only the wall-clock playback rate.

### Docs: `action_horizon` is a lower bound, not a maximum

The public `run_policy`/`eval_policy` (and `run_policy` tool) docstrings
described `action_horizon` as the *maximum* actions consumed from each policy
chunk before re-querying. The effective interval is actually
`max(action_horizon, policy.execution_horizon)` (`resolve_chunk_length`): it is
a *lower* bound, and RTC policies ignore it entirely. So for a chunk-emitting
policy (e.g. a VLA with `execution_horizon=50`) any smaller `action_horizon`
has no effect -- the full trained chunk is always consumed. In particular
`eval_policy`'s "set to `1` for closed-loop control" advice holds only for
single-action policies. The docstrings now state the clamp semantics so a
caller does not silently get open-loop chunk execution when expecting a tighter
re-query interval.

### Fixed: `set_body_properties(mass=...)` left the body's inertia stale

Setting a body's mass at runtime updated `model.body_mass` but not
`model.body_inertia`, leaving a physically inconsistent body -- heavy in
translation but retaining the old rotational resistance -- which silently
corrupted the rotational dynamics. Since `mass` is the only settable property,
the caller had no way to correct it. Because a rigid body's inertia tracks its
mass at fixed geometry (a uniform density change scales `I = integral of r^2 dm`
by the same factor), the inertia tensor is now scaled by `mass / old_mass`,
matching `randomize(randomize_physics=True)` and the Newton backend. In a
compound-pendulum check, a mass-only change wrongly shrank the swing period ~2x
(0.87 s vs 1.76 s); with the fix the period stays mass-invariant as physics
requires. Massless frames (mass 0) are guarded against division by zero.

### Fixed: `set_geom_properties(size=...)` left stale collision bounds so grown geoms were passed through

Resizing a primitive geom at runtime wrote the new `geom_size` but never
refreshed `geom_rbound` (the broadphase bounding-sphere radius) or `geom_aabb`
(the mid-phase AABB), both of which are derived from `geom_size` at compile time
and are not recomputed by `mj_forward`/`mj_step`. A geom grown past its original
bounds was therefore silently culled from the broadphase, so other bodies passed
straight through it while the call still reported `status="success"`. In a
repro, a ball dropped onto a small platform grown into a wide table fell through
to the floor (rest z 0.03) instead of landing on it (rest z 0.55). The bounds
are now recomputed from `geom_type` + `geom_size` for the size-defined
primitives (sphere/capsule/cylinder/ellipsoid/box) so a grown geom collides
correctly, matching a fresh compile at the new size. Mesh/plane/height-field
geoms take their extent from asset data, so a `size` write stays inert for them.

### Fixed: `raycast`/`multi_raycast` intersected stale geom poses (no forward, no lock)

`mj_ray` intersects a ray against `data.geom_xpos`/`geom_xmat` -- the world-frame
geom poses, which are derived state populated by kinematics and NOT recomputed on
a bare `qpos` write. Both `raycast` and `multi_raycast` called `mj_ray` directly
without refreshing those poses and without holding the sim lock, unlike every
other physics query (`get_jacobian`, `get_mass_matrix`, `get_body_state`,
`inverse_dynamics`, `get_sensor_data`), which lock and forward before reading.
As a result a cast issued after a direct pose change (a planning/IK loop that
writes `qpos`) silently reported a hit against a geom's *previous* location while
returning `status=success`, and a cast issued while a per-robot policy thread was
stepping could tear the read mid-`mj_step`. Both methods now refresh geom world
poses with `mj_kinematics` (the minimal forward for `mj_ray`, cheaper than a full
`mj_forward`) and serialize the cast under the sim lock; `multi_raycast` refreshes
once and holds the lock for the whole batch so all rays sample one consistent
snapshot.

### Added: `eval_policy(video=...)` records one rollout MP4 per episode

`run_policy` could already record rollout video, but `eval_policy` -- the
multi-episode, success-measuring path -- could not, so an evaluation could only
be read as an aggregate `success_rate` and never *watched* to see WHY episodes
failed. `eval_policy` (and `PolicyRunner.evaluate`) now accept the same
`video={...}` config as `run_policy` (`path` enables it, plus `fps` / `camera` /
`width` / `height`) and write one MP4 per episode with `_ep{i}` inserted into the
filename (`eval.mp4` -> `eval_ep0.mp4`, `eval_ep1.mp4`, ...), returning the
written files in the result json `video_paths`. The output path is validated and
the camera probed up-front, so a bad camera fails the eval immediately rather
than after N episodes of silent 0-frame MP4s. Recording is scoped to the
`success_fn` path; passing `video` with a benchmark `spec` is rejected with a
clear error (use `run_policy` for a benchmark rollout video). The writer
lifecycle (path validation, camera probe, fps-cadence frame capture) is now a
single `_RolloutVideoWriter` helper shared by `run` and `evaluate` instead of two
copies.

### Fixed: LeRobot fine-tuning validation discovers policy types from lerobot's live registry

`LerobotTrainer.validate` guarded `extra['policy_type']` against a hardcoded set
of LeRobot-native types, and gated `relative_actions` against a hardcoded
`{pi0, pi05, pi0_fast}` set. Both had drifted behind lerobot: the native-type set
omitted policies lerobot already ships (e.g. `eo1`, `molmoact2`, `vla_jepa`,
`wall_x`, and newer additions), so validation wrongly rejected them as "not
LeRobot-native" even though `make_policy_config` builds them and the inference
side resolves their classes; and `groot` now exposes `use_relative_actions`, so a
valid `groot` + relative-actions run was wrongly rejected. Both gates now read
lerobot's live `PreTrainedConfig` ChoiceRegistry (the relative-action check is
probed off each config class), matching the zero-maintenance dynamic discovery
the reward-model, robot, teleop, and camera surfaces already use. A static
fallback is kept for the offline case (lerobot not importable). Genuinely unknown
policy types are still rejected.

### Added: `evaluate_benchmark` honors `policy_object` + control frequency (parity with `run_policy` / `eval_policy`)

`evaluate_benchmark(policy_object=..., control_frequency=..., control_substeps=...)`
brings the benchmark evaluation entry point to parity with `run_policy` /
`eval_policy`. It could previously neither evaluate a pre-built `Policy` (it
always ran `create_policy`, forcing a redundant reload of a multi-GB VLA
checkpoint) nor set the control-loop rate (physics stepped at a hardcoded
50 Hz). A benchmark's `max_steps` maps to a wall-clock episode length that
depends on the control frequency, so a policy trained/evaluated at a
different rate was scored over a mismatched horizon; `control_frequency`
now lets the benchmark run at the policy's rate. The shared
`PolicyRunner.evaluate` plumbing already supported these; only the facade
exposes them now. A non-positive `control_frequency` is rejected with a
structured error, and the `control_frequency` tool parameter is forwarded on
the agent-dispatch path.


### Docs: `run_policy(async_rtc=...)` no longer claims SmolVLA/MolmoAct2 blend the chunk seam internally

- The `PolicyRunner.run` `async_rtc` docstring stated that "RTC-capable
  policies (pi0, pi0.5, SmolVLA, MolmoAct2) blend the seam internally through
  their own prev-chunk state (`rtc_config.execution_horizon`)". That conflates
  two independent things and is wrong for the checkpoints most users load.
  The async OVERLAP (latency masking) auto-enables for *any* chunk-emitting
  policy via `is_chunk_emitting()`; RTC SEAM BLENDING is a separate,
  checkpoint-level property (`supports_rtc`) that requires an enabled
  `rtc_config`. The public `lerobot/smolvla_base` checkpoint ships
  `rtc_config=None` and MolmoAct2 has no `rtc_config` at all, so both report
  `supports_rtc=False`: they get the overlap but a plain chunk swap at the
  seam, not a blended one. Reading the old text, a user would deploy
  `smolvla_base` expecting a smoothly-joined trajectory and instead get a
  velocity discontinuity at every chunk boundary. The docstring now describes
  overlap and seam-blending as the two distinct capabilities they are, and a
  regression test pins that `rtc_async_enabled` can be `True` while
  `supports_rtc` is `False`.

### Fixed: `run_policy`/`eval_policy` rollout video plays back at real time when `fps > control_frequency`

- The MP4 recorder (`video=...`) renders at most one frame per applied control
  step, so it cannot carry more than `control_frequency` unique frames per
  second of sim time. When the requested `fps` exceeded `control_frequency` the
  capture cadence still grabbed every step (it cannot up-sample) but the writer
  used the requested `fps`, so the video played back FASTER than real time by
  `fps / control_frequency` (e.g. a 110-step rollout at `control_frequency=15`
  with the default `fps=30` produced a 3.7 s MP4 for 7.3 s of sim - a silent 2x
  speed-up). The writer already down-samples to preserve real time when
  `control_frequency >= fps`; the `fps > control_frequency` case was unhandled.
  The writer fps is now capped at `control_frequency` (with a warning) so the
  rollout always plays back at real time; the common default
  (`control_frequency=50`, `fps=30`) is unchanged.

### Added: `describe()` advertises the physics-introspection / grounding surface

- `MuJoCoSimEngine.describe()` -- the single-call discovery surface an agent
  reads first to learn the engine's contract -- taught how to build a scene, run
  a policy, and record a dataset, but listed no way to READ the physics result.
  An agent that ran a rollout could not discover how to verify it (read a body's
  world pose, check gripper-object contact, query a sensor) without guessing
  method names, even though `get_body_state`, `forward_kinematics`,
  `get_contacts`, `get_contact_forces`, `get_sensor_data`, `get_energy`,
  `get_mass_matrix`, `inverse_dynamics`, `get_jacobian`, `get_total_mass`,
  `raycast`, and `multi_raycast` are all public methods the tool spec and action
  dispatcher already dispatch. `describe()` now advertises this read/verify
  surface alongside the act/record surface, so one call reveals how to ground a
  claim on a body-state delta (the documented way to verify a rollout) rather
  than on a rendered caption. The `start_recording` signature in `describe()`
  also now names its `cameras=` dataset-scope parameter, which was omitted.

### Fixed: the `grasped` predicate missed LIBERO/robosuite multi-geom bodies (a successful grasp scored as a failure)

The `grasped` benchmark/DSL predicate now matches the grasped body's geoms
across the `<body>_g<idx>` LIBERO/robosuite multi-geom convention, not only
the exact `body` / `<body>_geom` names. LIBERO objects (a BDDL object
`cube_1` owns collision geoms `cube_1_g0` / `cube_1_g1` ...) were never
matched, so `(grasped cube_1)` BDDL goals silently resolved to `False` even
when the gripper was in contact - a successful grasp was scored as a failure.
Body-geom matching now mirrors `_body_contact`'s `<body>_g` prefix so the two
contact predicates agree on what counts as a body's geom; strands-native
`add_object` (`<body>_geom`) and single-geom scenes are unchanged.


### Fixed: `CooperativeStop` from an `on_frame` hook crashed `eval_policy` / `evaluate_benchmark` instead of stopping gracefully

`CooperativeStop` is the documented signal an `on_frame` hook raises to stop a
rollout early, and it inherits `BaseException` (not `Exception`) specifically so
a hook author's broad `except Exception` cannot swallow it. `run()` honors it
(clean stopped-early success), but the two evaluation paths (`eval_policy`'s
legacy `success_fn` loop and the `evaluate_benchmark` spec loop) only caught
`except Exception` around the hook -- so a `CooperativeStop` propagated *uncaught*
out of the eval and crashed the whole evaluation, contradicting the `on_frame`
docstring's "never aborts the eval" contract. Both eval loops now catch
`CooperativeStop` and end the evaluation gracefully after the episodes completed
so far, exactly like `run()`. The eval result payload gains `stopped_early`
(bool) and `episodes_completed` (int, `<= n_episodes`); aggregate metrics are
computed over the completed episodes when a stop fires. Normal evaluations are
unchanged (`stopped_early=False`, `episodes_completed == n_episodes`).

### Fixed: an incompatible embodiment silently discarded a working normalization pipeline and misdirected the diagnostic

When a `lerobot_local` policy's processor pipeline loaded and was *active* but
its declarative embodiment / `image_keys` did not match the model's declared
features, `_configure_embodiment` raised a `ValueError` that was swallowed at
`debug` level: the entire (working) preprocessor + postprocessor -- including
normalization -- was silently discarded and the policy fell back to the raw
obs/action flow. The load-time diagnostic below then fired the generic
"loaded WITHOUT an action postprocessor (no `policy_postprocessor.json`)"
warning, which is *false* for these checkpoints (they ship a postprocessor; it
was discarded here) and sends the user down the wrong debugging path when the
arm barely moves. The embodiment-configuration failure is now surfaced as a
warning that names the real cause (or raised as a `RuntimeError` when
`processor_overrides` were supplied, mirroring the bridge-load path), and the
misleading missing-postprocessor message is suppressed for that case. A
malformed embodiment *spec* still raises loudly. Behaviour is unchanged for a
correctly-configured policy and for a checkpoint that genuinely ships no
processor configs.

### Added: `evaluate_benchmark(video=...)` records a per-episode rollout MP4

`eval_policy` could already record one rollout MP4 per episode, but
`evaluate_benchmark` (the spec/benchmark eval path) could not - the spec route
hard-rejected `video`, so a benchmark evaluation could only be read as an
aggregate `success_rate` and never watched to see *why* episodes fail.
`evaluate_benchmark` now accepts the same `video={"path", "fps", "camera",
"width", "height"}` config as `run_policy` / `eval_policy` and writes one MP4
per episode (`_ep{i}` filename templating), returning the written paths in the
result JSON `video_paths`. Frames are captured synchronously on the eval thread
at the `on_frame` point (render is read-only over `mjData`), so recording does
not perturb the bit-stable benchmark rollout. Bad path/camera fails up-front;
omitting `video` records nothing (opt-in). The agent-tool router folds the flat
`output_path`/`fps`/`camera_name` keys into `video` for `evaluate_benchmark`
too.

### Added: `verify-dataset` flags a truncated / partial-encode video (fewer frames than recorded)

`verify_dataset` (`strands-robots verify-dataset`) already flagged missing and
empty per-episode video files, but a *present, non-empty* MP4 whose decoded
frame count is fewer than the frames the parquet maps into it passed silently -
correct episode counts, a real file, but missing pixels (the encoder crashed
mid-episode, the file was partially synced, or the write was interrupted). Check
5 now also compares each video file's frame count, read from the container
header via PyAV (`av`) without decoding, against the sum of the `length` of
every episode packed into that file (LeRobot v3 concatenates whole episodes into
one shared file per camera). The comparison is best-effort: it is skipped when
`av` is unavailable, when a packed episode carries no `length`, or when the
codec header omits the frame count, so it never yields a false positive on a
header it cannot read - it reports only a confidently-read mismatch. This is the
frame-count sibling of the existing missing/empty-video and dead-column
integrity checks. Disable with `--no-check-videos`.

### Fixed: `download_assets(action="status")` marks each robot available or missing

The per-row availability marker in the `status` listing was an empty-both
`'' if r['available'] else ''` ternary (a fossil of an emoji marker that was
stripped), so downloaded and missing robots rendered identically and the
per-row output conveyed no availability information -- defeating the action's
purpose. Each row now carries an ASCII marker: `[ok]` when the robot's assets
are present, `[--]` when missing. The summary count line and cache path are
unchanged.

### Fixed: `move_object` retained a dynamic object's velocity, so a repositioned object kept its prior momentum

The dynamic-object path of `move_object` (objects with a freejoint) wrote the
new pose into `data.qpos` but never touched the freejoint's 6 velocity DOF, so
a bare position/orientation write left the object's prior linear and angular
velocity intact. A repositioned object therefore kept whatever momentum it had:
a settling object teleported to its new pose and immediately shot off, and an
eval/benchmark loop that repositions objects between episodes started each
episode with the object drifting from its "placed" pose (silently
non-reproducible). The dynamic path now zeroes the freejoint velocity so the
object is placed **at rest** at the new pose, matching `add_object` (spawns at
rest), `reset` (zeroes velocities), and the Newton backend (rebuilds from the
builder at rest). A `move_object` call with neither `position` nor `orientation`
remains a true no-op and leaves velocity untouched.

### Fixed: benchmark/reward DSL silently swallowed unresolvable body/joint names

A typo (or otherwise unresolvable name) in a benchmark success predicate or a
reward-term body/joint used to be completely silent. On a backend that
*supports* the lookup (`get_body_state` / `get_observation`), an unresolvable
name made the term degrade to a constant with no diagnostic: a bool predicate
to `False` (the episode silently never succeeds, indistinguishable from a
failing policy) and a `distance_neg` / `joint_progress` reward to `0.0` -- which
is that term's *maximum* (`-w * dist <= 0`), so a dead term pins at its ceiling
and inflates the reported return. During a multi-hour RL run or a benchmark
eval this quietly corrupts success rates and reward signals. The lookup helpers
now log the unresolvable name once at `WARNING` (deduplicated so the hot loop
never spams); returned values are unchanged. A missing lookup *method*
(unsupported backend) stays silent -- that is a capability gap, not a spec typo.
Additionally, `body_upright` now resolves the LIBERO `<name>_main` root-body
convention (the `_body_position` fallback was never mirrored to the quaternion
path), so `(upright X)` on a procedurally-generated LIBERO object no longer
silently evaluates to `False`.

### Fixed: OLD-FORMAT in-model normalization was silently dropped -> canonical checkpoints ran un-normalized

Loading a pre-processor-era lerobot checkpoint through the `lerobot_local`
provider ran it with normalization dropped. Those checkpoints (the canonical
zoo -- `lerobot/act_aloha_sim_transfer_cube_human`, `diffusion_pusht`, and the
tdmpc/vqbet entries a user grabs first) store their Normalize modules *inside*
the policy, so `model.safetensors` carries `normalize_inputs.*` /
`unnormalize_outputs.*` buffers and `config.json` carries a
`normalization_mapping`, with no `policy_preprocessor.json` /
`policy_postprocessor.json`. Current lerobot no longer registers those modules,
so `PreTrainedPolicy.from_pretrained` drops the buffers as "unexpected keys"
(only a `WARNING:root` line) and, with no processor JSON to replace them, the
policy ran with normalization dropped: observations reached the model raw and
predicted actions reached the robot un-unnormalized. For a MEAN_STD checkpoint
that is not an "arm barely moves" under-motion -- raw z-scored actions applied
as robot units make the arm *flail* (measured on `act_aloha_sim_transfer_cube_human`:
right-gripper path 3.36 m spanning z [0.03, 0.90] m, vs 0.62 m spanning
z [0.16, 0.32] m once normalized). `ProcessorBridge` now reconstructs the
pre/post pipelines from those same in-model buffers using lerobot's own
`extract_normalization_stats` + `make_pre_post_processors` factory (the exact
machinery of `migrate_policy_normalization`), so an old-format checkpoint runs
normalized with zero user action. The reconstruction is best-effort and only
fires when a checkpoint ships no processor configs, no `norm_stats.json`, but
does carry in-model buffers -- modern checkpoints are untouched. The
reconstruction also degrades to passthrough when the lerobot recovery
helpers cannot be imported for any reason (not only `ImportError`): an
unrelated broken sibling policy module -- e.g. a dataclass that fails at
definition time while importing the `lerobot.policies` package -- must not
crash an ACT/diffusion checkpoint load. The `make_pre_post_processors`
factory is imported from its canonical module `lerobot.policies.factory`
(where lerobot's own `migrate_policy_normalization` sources it) rather than
the `lerobot.policies` top-level re-export, which is not stable across
lerobot releases.

### Fixed: a partial `robot_state_keys` mismatch silently mis-aligned `observation.state`

When the configured `robot_state_keys` are keyed by a robot's actuator names
but some of those names are absent from `get_observation()` -- the canonical
case being a mimic/tendon gripper whose actuator (`left/gripper` /
`right/gripper` on aloha) is not among the observation's finger-joint names
while the arm joints are -- both state-build paths in the `lerobot_local`
provider iterated the resolved key order and appended only the keys present in
the observation. An absent key was therefore *dropped*, shifting every
following joint value up one index before the trailing zero-pad, so the model
received a garbage `observation.state` while the run reported success. On a real
MuJoCo aloha sim this shifted 8 of the 14 state dims (the entire right arm slid
into the wrong slots and both gripper dims landed wrong). `_resolve_state_order`
already made the *all*-missing case loud (#897); the *partial*-missing case was
still silent. A missing key is now zero-filled IN PLACE via a shared
`_collect_state_values` helper, so present joints keep their model index, and
the degradation is surfaced -- `strict_keys=True` raises naming the missing
keys; otherwise it warns once and sets the new `missing_state_keys_used`
telemetry flag (surfaced in the `run_policy` / `eval_policy` result alongside
`generic_state_keys_used`). Robots whose keys are all present (e.g. `so101`)
are unaffected.

### Fixed: warn when an action value is clamped by a ctrl-limited actuator

`send_action` (and therefore `replay_episode`) wrote an action value verbatim
to `data.ctrl` for an actuator addressed by name. When that actuator is
`ctrllimited` and the value falls outside its `ctrlrange`, MuJoCo clamps it
inside `mj_step` -- so the commanded value is silently NOT reproduced for that
actuator while the call still reports success. Replaying a dataset whose action
units differ from the target robot's actuator ctrl units hit this hard: a
normalized gripper action (e.g. the ALOHA convention, values in `[0.19, 1.12]`)
replayed onto a joint-position gripper whose ctrlrange is `[0.002, 0.037]`
clamps every value to the maximum, pinning the gripper fully open and silently
destroying the grasp channel while `replay_episode` reports `Frames: N/N`. The
direct-actuator ctrl write now warns once per `(prefix, key)` when the value is
meaningfully outside the actuator's `ctrlrange`, naming the actuator and range
so the unit mismatch is actionable instead of silent. A small tolerance absorbs
boundary rounding; unlimited actuators (which never clamp) are skipped; the
warning is de-duplicated so a 50Hz control loop never spams the log. No
trajectory behaviour changes.

### Added: `add_robot(keyframe=...)` spawns a robot in its canonical home pose

MuJoCo Menagerie robots ship a canonical start pose in a MJCF `<keyframe>`
(panda/ur5e/fr3/kuka `home`, aloha `neutral_pose`, quadruped/humanoid standing
`home`). `add_robot` and `reset` ran `mj_resetData` -- the all-zero
configuration -- so that shipped pose was unreachable outside the LIBERO
benchmark adapter, and a robot spawned folded/collapsed rather than in its
ready pose. A policy trained from the home pose then saw an out-of-distribution
start that measurably suppressed its rollout. `add_robot(keyframe="home")` (or
an integer index) now reads the named keyframe from the robot's source model,
applies its `qpos` to the robot's joints by name at spawn, and records it so
`reset()` restores it -- a keyframe spawn is sticky across resets, mirroring how
a benchmark restores its canonical start each episode. `keyframe=None` (the
default) keeps the historical zero-pose spawn byte-for-byte; an unknown
keyframe name/index is a hard error that names the available keyframes rather
than silently falling back to zeros. Newton `add_robot` accepts the argument
for signature parity and rejects a non-`None` value with a clear
not-yet-supported error. Exposed on the `simulation` agent tool as a `keyframe`
string.

### Fixed: the `aloha` embodiment mis-aligned `observation.state` and mapped gripper actions onto non-actuators

The declarative-embodiment state builder (`PackStateProcessorStep.observation()`)
appended a value only for `state_keys` *found* in the observation and skipped the
absent ones, so a missing key DROPPED its slot and shifted every following joint
up one index before the trailing pad -- the embodiment-path analog of the
generic-path fix from "a partial `robot_state_keys` mismatch silently mis-aligned
`observation.state`" above. A missing `state_key` is now zero-filled IN PLACE so
the present joints keep their model index, with a one-time warning naming the
absent keys.

The `aloha` config compounded this: it declared the 16 finger-JOINT names
(`left/left_finger`, `left/right_finger`, ...) for both `state_keys` and
`action_keys`, but the model's 14 ACTUATORS follow the canonical gym-aloha /
LeRobot ACT convention `[6 arm + 1 gripper] x 2` (`left/gripper` / `right/gripper`
at indices 6 and 13, each driving both fingers). Against a canonical 14-D ACT the
16-key state build raised `observation.state dim 16 > model expected 14`, and the
finger-joint `action_keys` mapped the policy's gripper command onto non-actuators
(the gripper was never driven). The config now uses the 14 actuator keys, so the
action maps 1:1 onto actuators and the arm proprioception is index-aligned; the
two gripper STATE slots -- absent from the sim observation, which exposes finger
joints -- are zero-filled in place (a units-correct finger->gripper state mapping
is a checkpoint-specific follow-up). Loading the canonical
`lerobot/act_aloha_sim_transfer_cube_human` through `embodiment="aloha"` now runs
a rollout instead of crashing.

### Changed: track lerobot 0.6 -- require `lerobot>=0.6.0` and drop the 0.5.1-era torch/torchcodec overrides

The `[lerobot]` / `[molmoact2]` extras now require `lerobot>=0.6.0,<0.7.0`
(was `>=0.5.0,<0.6.0`). lerobot 0.6 ships mature, platform-correct dependency
markers -- `torch>=2.7,<2.12` with a `torchcodec>=0.11,<0.12` marker on linux
aarch64 -- so the codec/decoder stack now resolves ABI-consistently
(torch 2.11 + torchcodec 0.11.x + torchvision 0.26) on linux x86_64/aarch64 and
macOS arm64 without any strands override. The per-platform torchcodec pins in
the `[lerobot]` extra and the `torch`/`torchvision` entries in
`[tool.uv].override-dependencies` -- all added to compensate for lerobot 0.5.1's
deficient markers (its `torch<2.11` cap that skipped the NVIDIA Thor/Jetson
sm_110 cuBLAS fix, and its torchcodec marker that excluded linux aarch64) -- are
removed; the sole remaining uv override is the diffusers security floor. The
aarch64 torch 2.11 requirement (the Thor sm_110 fix) now falls out of lerobot
0.6's own torchcodec marker rather than a strands override, and the previously
unbounded aarch64 `torchcodec>=0.11` pin (which resolved a torch-ABI-mismatched
torchcodec 0.14) is bounded by lerobot 0.6's `<0.12`. `MolmoAct2Policy` (added
to lerobot after the 0.5.1 PyPI release) now resolves straight from PyPI via the
`[molmoact2]` extra -- the "install lerobot from source" step is gone.

### Docs: correct post-lerobot-0.6 VLA install guidance

The `lerobot>=0.6.0` requirement obsoleted a body of pre-0.6 install lore that
survived in the `train_policy` tool docstring and the training / lerobot-local
docs: `pip install 'lerobot[smolvla]==0.5.1'`, a `transformers==5.3.0` pin, the
"a newer transformers crashes the VLA import (`backbone_cfg`)" note, and
"MolmoAct2 requires lerobot from source". lerobot 0.6's `[smolvla]`/`[pi]`/
`[molmoact2]` extras now require `transformers>=5.4.0,<5.6.0`, so the old
`transformers==5.3.0` pin is a hard resolution conflict rather than a fix, and
`strands-robots[molmoact2]` resolves `MolmoAct2Policy` straight from PyPI (no
git-from-source). The guidance now matches the declared extras; a regression
test pins the docs to the pyproject requirements so the stale lore cannot
return.

### Fixed: `build_command` dropped `--peft.lora_alpha`, diverging from the LoRA config it trains

`LerobotTrainer.build_command` is the argv-parity helper that documents the
draccus CLI equivalent to the typed `TrainPipelineConfig` that `train(cfg)`
consumes in-process, and it powers the native-parity drift check. Its LoRA
branch emitted `--peft.method_type`, `--peft.r`, and `--peft.target_modules`
but omitted `--peft.lora_alpha`, even though `build_config` wires
`spec.lora_alpha` into `PeftConfig.lora_alpha`. `lora_alpha` sets the LoRA
scaling numerator (scaling = `lora_alpha / r`), so the documented "equivalent
command" for a LoRA fine-tune silently trained with lerobot's default alpha
rather than the requested one -- a real behavioral divergence between the CLI
description and the in-process run, and a hole in the parity guard. It is now
emitted (only when set, mirroring `--peft.r` / `--peft.target_modules`), and a
regression test pins the emitted flag to `cfg.peft.lora_alpha`.

### Fixed: lerobot "too old / absent" install hints recommended a from-source install after the `>=0.6.0` bump

The user-facing error hints for a missing/too-old lerobot -- MolmoAct2's
`_LEROBOT_VERSION_HINT` and the `lerobot.rewards` gates in `training.reward` /
`LerobotTrainer.validate` -- still claimed lerobot was "not yet on PyPI (latest
release 0.5.1)" and told the caller to install it from source
(`lerobot @ git+https://github.com/huggingface/lerobot.git`). Since the core
dependency is now pinned to `lerobot[feetech,dataset]>=0.6.0`, lerobot 0.6 --
including `MolmoAct2Policy` (lerobot PR #3604) and the `lerobot.rewards`
package -- ships straight from PyPI through the `strands-robots[lerobot]` /
`[molmoact2]` extras, so a from-source `git+` install is both unnecessary and
liable to conflict with the pinned floor. The hints now point at a plain PyPI
(re)install of the extra and name the correct `>= 0.6.0` floor. This is the
runtime-error counterpart of the docstring/docs guidance corrected in the
post-0.6 dependency-guidance pass, which left these `.py` strings stale.

### Fixed: mobile-base observation dropped all base state when the floating base is an unnamed free joint

`get_observation` surfaces floating-base IMU-style signals (`base_quat`,
`base_ang_vel`) only when the free joint was found while iterating
`robot.joint_names`. That holds for a humanoid whose base joint is named (e.g.
the Unitree G1's `floating_base_joint`), but a mobile manipulator such as
LeKiwi carries its base on an **unnamed** `<freejoint/>` that is not in
`joint_names` (those are the actuated wheel/arm joints). Such a robot was
silently observed as a fixed-base arm -- its observation carried no base
orientation or angular velocity -- so a locomotion/navigation controller (or a
recorder configured to log base state) could not sense the base heading or turn
rate. The base free joint is now recovered from the kinematic tree (walk
up from an actuated joint to its ancestor base body), so any floating base
surfaces base state regardless of whether its free joint is named. A sibling
free-jointed task object (a cube) is never mistaken for the base, and a
fixed-base arm still surfaces no base state.

### Fixed: `get_robot_state` misreported a floating base's free joint as a scalar joint

For a robot with a floating base (a 6-DoF free joint), `get_robot_state` read
`qpos[jnt_qposadr]` as a scalar joint "position" and `qvel[jnt_dofadr]` as a
"velocity". A free joint's qpos is `[xyz(3) + quat(4)]` and qvel is
`[linvel(3) + angvel(3)]`, so this reported the base's x-coordinate as a joint
angle and silently dropped the orientation and the rest of the twist -- while
`get_observation` (its policy-facing counterpart) already surfaces the base
correctly as `base_quat` / `base_ang_vel`. A humanoid's named
`floating_base_joint` hit the wrong-scalar path; a mobile base's unnamed
`<freejoint/>` (e.g. LeKiwi) was skipped entirely, so `get_robot_state` had no
base information at all. The free joint is now surfaced under a structured
`base` entry -- `position` (xyz), `quaternion` (w,x,y,z), `linear_velocity`,
`angular_velocity` -- recovered from the kinematic tree when the free joint is
unnamed, with the `quaternion` / `angular_velocity` matching `get_observation`.
A fixed-base arm still reports only its scalar joints (no `base` entry).

### Fixed: Newton `get_observation` / `get_robot_state` shifted every joint after a floating base

The Newton backend built its per-joint coordinate/DOF index maps
(`_joint_coord_index` / `_joint_dof_index`) from a per-joint ordinal offset --
one coordinate and one DOF per joint. That assumption breaks for any robot with
a multi-coordinate joint: a floating base (a free joint) spans 7 coordinates
(xyz + quaternion) and 6 DOFs, so every child joint after it was read from the
wrong index. A humanoid whose root is a free joint (e.g. Unitree G1/H1) reported
a base coordinate for its first leg joint and shifted the reading of every joint
after -- so `get_robot_state` and the policy-facing `get_observation` returned
garbage joint positions/velocities for the entire floating-base robot. The maps
are now built from Newton's authoritative per-joint coordinate/DOF starts
(`joint_q_start` / `joint_qd_start`), so each joint reads its own value. A
fixed-base arm (all revolute joints) is unaffected -- its indices are already
the joint ordinals.

### Fixed: remote inference server rejected every image observation over 1 MiB

`PolicyServer` opened its WebSocket with `serve(...)` at the library default
1 MiB (`2**20`) frame limit, while `RemotePolicy` correctly passes
`max_size=None` to `connect(...)`. A real VLA observation carries camera
frames -- a single 640x480 RGB frame base64-encodes to ~1.2 MiB and a
multi-camera observation is several MiB -- so the server closed the
connection with `1009 (message too big)` on the first image-carrying
`get_actions` request, before the wrapped policy ever ran. The whole remote
path was therefore usable only with an image-free policy. The server now
passes `max_size=None` (and, matching the client, `compression=None`) to both
`serve(...)` call sites so large multi-camera observations stream in. Verified
end to end by serving a SmolVLA policy over the wire and driving a MuJoCo
rollout to a recorded LeRobotDataset.

### Fixed: Newton backend surfaces a floating base as a structured pose, not a garbage scalar

On the Newton backend, `get_robot_state` reported a floating-base robot's free
root joint (a humanoid's named `floating_base_joint`) as a scalar joint
`{position, velocity}` -- reading its base x-coordinate as a "position" and
linear-velocity-x as a "velocity", silently dropping the orientation and the
rest of the twist -- and `get_observation` never surfaced the base at all. A
real `unitree_g1` therefore reported garbage for its base and no orientation to
a WBC / locomotion controller. Both now mirror the MuJoCo backend: the free
joint is excluded from the scalar `state` map and a structured `base` entry
(`position`, `quaternion` w,x,y,z, `linear_velocity`, `angular_velocity`) is
surfaced, and `get_observation` gains `base_quat` / `base_ang_vel`. Newton
stores the free joint's coordinates as `[xyz, quat_xyzw]`, so the quaternion is
reordered `xyzw -> wxyz` to match the MuJoCo (w,x,y,z) contract.

### Fixed: warn when a floating base's orientation + angular velocity are dropped from a recorded dataset

`get_observation` surfaces `base_quat` (orientation, w,x,y,z) and `base_ang_vel`
(rad/s) for a floating-base robot -- a humanoid's named `floating_base_joint` or
a mobile base's unnamed `<freejoint>` -- so a locomotion / whole-body-control
policy can read its base state. But `start_recording` derives the LeRobotDataset
`observation.state` schema from the robot's scalar joint names, so those base
signals never reached the dataset: the free base contributed only a single
scalar slot (its x-position) when its joint was named, and nothing at all when
it was unnamed. A dataset recorded from a humanoid/mobile rollout was therefore
silently base-blind, and a policy trained on it would be missing the exact
orientation/angular-velocity signals a locomotion controller needs. Both
backends (MuJoCo and Newton) now warn once at `start_recording` when a
floating-base robot is being recorded, naming the affected robot(s) and the
dropped signals, per the project's "no silent data loss" contract. The dataset
schema is intentionally unchanged (existing datasets are stable); this surfaces
the omission instead of dropping it silently. A new shared
`_robot_free_base_joint_id` helper centralizes the named+unnamed free-joint
detection.

### Fixed: preserve a floating base's orientation + angular velocity in recorded datasets

`get_observation` surfaces a floating-base robot's `base_quat` (orientation,
w,x,y,z) and `base_ang_vel` (rad/s), but `start_recording` derived the
`observation.state` schema from scalar joint names only, so those base signals
were dropped from every recorded frame - a humanoid's named `floating_base_joint`
contributed just one scalar slot (its x-position) and a mobile base's unnamed
`<freejoint>` contributed nothing. A locomotion / whole-body-control policy
trained on the resulting dataset was base-blind. `start_recording` (MuJoCo
backend) now writes the base orientation + angular velocity as per-component
scalar columns (`base_quat.w`.. `base_ang_vel.z`), so a recorded G1/mobile-base
episode carries the base state a WBC policy needs (verified end to end: the
recorded columns read back byte-for-byte after `LeRobotDataset` reopen). This
supersedes the earlier drop-warning (#1169): the base state `get_observation`
surfaces is now recorded, not merely flagged. A fixed-base arm is unaffected
(no base columns, no schema growth); multi-robot base columns are prefixed like
joint ids (`alice__base_quat.w`). `DatasetRecorder.create` gains an optional
`extra_state_specs` to declare per-component vector state columns whose source
key is read from the observation and flattened in schema order.

### Fixed: surface a floating base's full position + linear velocity (not just orientation + turn rate)

`get_observation` surfaced a floating-base robot's `base_quat` (orientation) and
`base_ang_vel` (turn rate) but not its `base_pos` (world x,y,z, including
HEIGHT) or `base_lin_vel` (m/s) - the two signals a locomotion, velocity-tracking
or mobile-manipulation controller most needs: you cannot detect a fall or track a
height target without base height, nor compute a velocity-tracking reward without
base linear velocity. A free joint's `qpos` is `[xyz, quat]` and its `qvel` is
`[linvel, angvel]`, so both were already available; only the orientation half was
being read. `get_observation` now surfaces the full 6-DoF base pose + twist -
`base_pos`, `base_quat`, `base_lin_vel`, `base_ang_vel` - on both the MuJoCo and
Newton backends, and `start_recording` (MuJoCo) preserves all four as
per-component scalar columns (`base_pos.x`.. `base_ang_vel.z`) so a recorded
humanoid/mobile-base episode is no longer base-blind (verified end to end: a
known base position + linear velocity read back byte-for-byte after
`LeRobotDataset` reopen). All four keys are additive and absent for fixed-base
arms (no schema growth); multi-robot base columns are prefixed like joint ids
(`alice__base_pos.x`). Completes the base-state surfacing begun in #1134/#1172.

### Fixed: `send_action` crashed on a non-scalar dict action value instead of returning an error

`send_action` returns a structured `{"status": "error", ...}` for every other
malformed input (a wrong-length vector, a non-numeric vector entry, a scalar, a
string, unresolved keys, no world), but a `{name: value}` mapping whose value was
non-scalar - a list / tuple / multi-element array, exactly what a policy emitting
a vector-valued key such as `base_velocity: [vx, vy, omega]` produces - slipped
past the contract and raised an unhandled `TypeError` deep in the actuator-apply
loop (`float(value)`), crashing the caller mid-rollout after already writing
`data.ctrl` for the earlier keys. The shared `_coerce_action` now validates that
every mapping value coerces to a scalar float up front, so the whole action is
rejected atomically with an actionable message naming the offending key - on both
the MuJoCo and Newton backends. Scalars, numeric strings and `numpy` float
scalars are accepted unchanged.

### Fixed: Newton reported a floating base's angular velocity in the world frame, not the body frame

`get_observation` / `get_robot_state` are a cross-backend contract, but the two
backends surfaced a floating base's `base_ang_vel` in different frames for the
same physical motion. The MuJoCo backend returns it in the BODY frame (its
free-joint `qvel` angular block is local) - the IMU-gyro convention a WBC /
locomotion controller consumes (the WBC observation builder feeds `base_ang_vel`
straight in alongside a body-frame projected-gravity cue). The Newton backend
returned the RAW world-frame angular velocity, so once the base yawed away from
identity the two backends disagreed by up to the full magnitude of the signal
(a base spinning about world-X at 2 rad/s reads `[2, 0, 0]` on Newton vs
`[0, -2, 0]` on MuJoCo at a 90-degree yaw). A humanoid/mobile policy evaluated
across backends - or a G1 WBC controller driven on Newton - was silently fed a
mis-framed gyro signal that destabilises the gait as the base turns. Newton's
`_free_base_pose` now rotates the angular velocity world -> body via the base
quaternion so `base_ang_vel` matches MuJoCo (verified equal to 1e-6 across a
0-90 degree yaw sweep on a real G1). The linear velocity stays world-frame on
both backends (already consistent); the frame of each base signal is now
documented on both backends' `get_observation`.


### Fixed: surface a mistyped RemotePolicy connection endpoint instead of silently defaulting to localhost

`RemotePolicy` (the `remote` inference client) tolerates unrecognized
constructor kwargs so a shared `policy_config` superset can be forwarded through
`create_policy` unchanged -- the cross-provider ignore-unknown-kwargs contract.
But it dropped them SILENTLY, so passing the server endpoint under the wrong
name (e.g. `RemotePolicy(uri=...)` -- `uri` is the object's own attribute name,
an easy slip) left the client connected to the default `ws://127.0.0.1:8765`
with no hint that the intended endpoint never took effect; the only symptom was
a confusing "connection refused" to a port the user never chose. `RemotePolicy`
now logs a WARNING naming the ignored kwarg(s) and the endpoint actually in use,
so the misconfiguration is visible at construction. Unknown kwargs are still
tolerated (no behavioural break); the normal `endpoint=`/`host=`/`port=` and
smart-string (`create_policy("ws://...")`) paths stay silent. Mirrors the
earlier no-silent-localhost-default fix for `cosmos3://` URLs.

### Fixed: Cosmos 3 msgpack decode returned read-only, buffer-aliasing arrays

The vendored NumPy msgpack codec the Cosmos 3 RoboLab client uses to unpack the
server's action chunk (and any returned observation) reconstructed each array
with `np.ndarray(buffer=<recv bytes>, ...)`. Because msgpack hands back an
immutable `bytes` object, that array is **read-only** and does **not own its
data** -- it aliases the transient wire buffer. Normalizing a decoded array in
place (e.g. `image /= 255`) then raises `output array is read-only`, and handing
it to `torch.from_numpy` (zero-copy) trips torch's "given NumPy array is not
writable ... undefined behavior" hazard. The sibling VERA packer
(`policies/vera/_msgpack_numpy.py`) already copies for exactly this reason, and
the remote-inference protocol decodes into a writable `bytearray`; the Cosmos 3
decode was the lone site left aliasing the recv buffer. It now copies to a
writable, owning array, matching both -- round-trip dtype/shape/value fidelity
is unchanged.

### Added: base_velocity velocity-tracking reward term for the predicate/reward DSL

The declarative predicate/reward DSL (`strands_robots.simulation.predicates`,
consumed by `DeclarativeBenchmark` specs and RL `SimEnv` reward stacks) grew a
`base_velocity(vx, vy, wz, weight=1.0, robot=None)` reward term -- the canonical
dense locomotion reward, previously impossible to express because the DSL could
only reference body positions, quaternions and scalar joint positions, never a
floating base's velocity.

It rewards a floating-base robot for matching a commanded BODY-frame velocity
(`vx` forward, `vy` lateral in the robot's own heading, `wz` yaw rate) as
`-weight * ||(v_body_x, v_body_y, w_body_z) - (vx, vy, wz)||`. It consumes the
floating-base observation surfaced by `get_observation`: `base_lin_vel` (world
frame) is rotated into the base frame via `base_quat`, and `base_ang_vel` is
already body-frame, so the tracked velocity is heading-relative -- matching the
IsaacLab / legged_gym locomotion-command convention rather than a fixed world
axis. On a fixed-base arm (no floating base) the term degrades to `0.0` and logs
the missing base once, consistent with the DSL's other name-resolution
degradation. Works on both the MuJoCo and Newton backends (both surface the base
twist with the same frame convention).

## [0.4.1] - 2026-07-01

### Security: Removed the unregistered `mimicgen` dependency (dependency-confusion RCE, CVE-pending)

The `vera-sim` extra pinned `mimicgen==1.0.0`, but NVlabs MimicGen has never
been published to PyPI -- the `mimicgen` name on PyPI is an unaffiliated
third-party package uploaded 2026-06-27. Installing `strands-robots[vera-sim]`
would therefore fetch and execute an attacker-controllable distribution at
install time, before any project code runs. The dependency was dead weight from
the source's perspective (`grep -rn "import mimicgen" src/` returns nothing);
the `mimicgen` VERA embodiment is only a config string and needs no package. The
pin is removed and replaced with a comment documenting that NVlabs MimicGen is
git-only (`pip install "mimicgen @ git+https://github.com/NVlabs/mimicgen.git"`).
A stdlib-only `scripts/audit_deps.py` supply-chain checker (denylist plus an
optional `--check-pypi` 404 gate for unregistered/typosquat names) and a
`tests/test_dependency_audit.py` regression guard block any re-introduction. The
`[all]` extra never referenced `vera-sim`, so `pip install 'strands-robots[all]'`
was not affected; exposure was limited to `[vera-sim]`. Anyone who installed
`[vera-sim]` since 2026-06-27 should `pip uninstall -y mimicgen` and reinstall
into a clean environment.

### Fixed: `teleoperate()` reported `status="success"` even for a dead follower

Running teleop with the follower unpowered still ended the session with
`status: "success"` -- a dead teleop was indistinguishable from a healthy one
by `status`. Two layers, both fixed:

- `_teleop_stats` hardcoded `"status": "success"`. It now derives the
  session-end status from the counters it already returns: `success` when
  `errors == 0`, `error` when every attempt failed (`frames == 0 or errors >=
  frames`, covering both the soft mode where `send_action` returns an error
  dict and the hard mode where the leader's `get_action()` raises), and
  `degraded` for a mixed session. `degraded` is a new value -- strict
  `status == "success"` callers now treat a partially-failing session as
  unhealthy, which is the point.
- Hardware validation exposed why honest counters alone were not enough:
  lerobot's `MotorsBus.connect()` opens the serial port *before* the motor
  handshake, so a failed handshake left `is_connected` True and fire-and-forget
  writes to the dead bus kept "succeeding" (a fully unpowered session counted 1
  error in 142 frames). A failed connect now rolls back by closing the
  half-open port -- skipping the default torque write, which would raise on
  the unreachable bus and leave the port open again -- on both the lazy teleop
  connect in `send_action` (every tick retries and keeps surfacing the
  failure) and the explicit `_connect_robot` path (which would otherwise
  short-circuit its next attempt on "already connected" and report success
  against a dead bus).

The live-status query `get_teleoperate_status` is unchanged.

### Added: RL parity -- `staged_reward`, a gym adapter, and vectorized PPO

The from-scratch RL lane reaches parity with the reference stacks: a
`staged_reward` composition, a gymnasium-compatible environment adapter, and a
vectorized PPO collection loop that drives `VecSimEnv` for N-trajectory-per-step
rollouts. The trainer's `render(output_path=...)` honors the sandboxed
output-path contract, and the env union is tightened so the trainer accepts only
the precise observation/action spec it can consume. Selected through the same
factory path as the imitation trainers.

### Added: from-scratch RL trainers -- on-policy PPO (`create_trainer("ppo")`) and off-policy FastSAC (`create_trainer("fast_sac")`)

`strands_robots/training/` previously shipped only imitation / post-tuning
trainers (`lerobot`, `groot`, `cosmos3`, `mock`) -- there was no policy-gradient
loop. Two from-scratch reinforcement-learning trainers now land as providers on
the existing `create_trainer` factory: `PpoTrainer` (`create_trainer("ppo")`),
an on-policy Proximal Policy Optimization backend, and `FastSacTrainer`
(`create_trainer("fast_sac")`), an off-policy soft actor-critic. Both subclass
`BaseRLAlgo` and are registered through lazy loaders (the training package stays
torch-free until an RL provider is resolved on first use), so they honor the
existing `validate -> prepare -> train -> export` lifecycle with no new
abstraction. They pair with `VecSimEnv` for parallel trajectory collection and
`BaseRLAlgo.evaluate()` for deterministic scoring.

### Added: auto-register the NVIDIA EGL vendor ICD so `MUJOCO_GL=egl` renders on the GPU

Complements the software-rasterizer warning by removing the misconfiguration
instead of only reporting it. When the effective GL backend is `egl` and
`libEGL_nvidia` is installed but no NVIDIA vendor ICD is registered,
`_configure_gl_backend` now stages a vendor ICD JSON in the user-writable
strands-robots base dir and points glvnd at it via the documented
`__EGL_VENDOR_LIBRARY_FILENAMES` override (NVIDIA first, system ICDs as
fallback) before `import mujoco`. No root required. It is a strict no-op when an
explicit vendor override is set, an NVIDIA ICD is already registered
system-wide, no NVIDIA EGL library is installed (Mesa is correct there), or off
Linux. On a headless NVIDIA host with the system ICD absent, the staged path
renders on the GPU (~0.69 ms/frame) byte-identical to the default GPU render,
versus hundreds of ms/frame on the Mesa `llvmpipe` fallback it removes.

### Added: warn when MuJoCo EGL silently falls back to a CPU software rasterizer

`_can_render()` probes render availability by creating a `mujoco.Renderer`, but
EGL can load and still route to Mesa `llvmpipe` (CPU software rasterization) when
no GPU EGL vendor ICD is registered -- common on NVIDIA hosts/containers missing
`10_nvidia.json`. Offscreen rendering then still works but runs roughly two
orders of magnitude slower (measured ~268 ms/frame on `llvmpipe` vs ~0.67
ms/frame on an NVIDIA L40S for the same 256x256 scene), silently throttling every
policy observation, rollout video, and dataset recording with no signal. The
render probe now also reports the active `GL_RENDERER` (best-effort, fully
wrapped so it can never change the probe's pass/fail), and a one-time
`logger.warning` fires when a software rasterizer (`llvmpipe` / `softpipe` /
`swrast` / `kms_swrast`) is active, naming the concrete fix. Purely diagnostic:
no change to rendered output or render availability.

### Added: `add_object` material / texture / reflectance for MuJoCo scenes

`Simulation.add_object()` exposed only `rgba`, so every object compiled with no
material assigned and rendered as flat, glossy plastic -- an obviously-synthetic
input for a VLM/VLA policy trained on real-camera footage. An optional `material`
spec now attaches a real MuJoCo material (`reflectance` / `specular` /
`shininess` / `texrepeat`) and, optionally, a texture: an image file or a
procedural builtin (`checker` / `gradient` / `flat` with `rgb1` / `rgb2` /
`texdim`). The geom keeps its `rgba`, which tints a textured or solid material.
Additive and backward-compatible: gated on `material is not None`, so the
rgba-only path is byte-for-byte unchanged (`matid == -1`). The material is built
before the body is added to the spec, so an invalid material (missing texture
file, unknown builtin, or both `texture` and `builtin` set) raises `ValueError`
before any spec mutation -- no silent fallback to flat plastic and no orphan
body left behind. The Newton backend rejects a non-`None` `material` loudly. The
schema is documented in `describe()`, the agent tool schema, and the
world-building docs.

### Added: MotionBricks generative-motion provider for the Unitree G1

A `motionbricks` policy provider that generates G1 locomotion motion. It
consumes `locomotion_style` from `policy_kwargs` (a plain-string contract, never
a planner import) to steer the gait, owning the accepted style vocabulary as
`LOCOMOTION_STYLES` and the clip mapping.

### Added: `WBCGaitPolicy` -- gait-clock variant of the whole-body-control policy

A gait-clock variant of `WBCPolicy` for the Unitree G1: a 95-dim observation
with an explicit bipedal phase clock, alongside the base SONIC whole-body-control
provider.

### Added: `CompositePolicy` -- stack locomotion and manipulation on one robot

A `CompositePolicy` provider that stacks policies across DOF groups, so a single
robot can run a locomotion policy and a manipulation policy composed together
(for example a walking humanoid that also drives its arms).

### Added: reBot B601-DM single + bimanual hardware registry entries

LeRobot registers the Seeed Studio reBot B601-DM follower as
`rebot_b601_follower` and its bimanual variant as `bi_rebot_b601_follower`, but
the strands registry had no entry mapping a canonical name to those types, so
`Robot("rebot_b601", mode="real")` failed even though the `lerobot_local`
embodiment configs already shipped. Two hardware-only entries (no sim asset,
matching the `omx` / `bi_openarm` pattern) now map `rebot_b601` and
`bi_rebot_b601` (a 6-DOF arm + gripper driven by Damiao CAN motors). Both carry
`requires_lerobot_from_source: true` because the LeRobot types are not yet in a
PyPI release within the pinned `lerobot>=0.5.0,<0.6.0` range; the conformance
guard skips from-source entries so CI stays green, and removing the flag
re-enables full checking once lerobot publishes.

### Added: DAgger correction-collection action for `lerobot_teleoperate`

`lerobot_teleoperate` could teleoperate or roll out a policy, but had no
intervention / takeover path -- no way to run a policy on the follower, let the
leader pre-empt mid-episode, and record the human correction as new dataset
episodes (DAgger), the data-collection loop behind correction-driven fine-tuning.
A `dagger` action now builds the correct nested `lerobot-rollout` CLI
(lerobot 0.5 split policy rollout out of `lerobot_record` into `lerobot-rollout`
with a native DAgger strategy) and runs it through the existing session
machinery. New params: `policy_path`, `dagger_record_autonomous`,
`dagger_input_device` (keyboard | pedal), `dagger_num_episodes`.
`--dataset.push_to_hub` is emitted explicitly (lerobot defaults it to `True`) so
an unattended correction run never auto-uploads; `policy_path` and
`dataset_repo_id` are required and the input device is enum-checked.

### Added: end-to-end VLA-on-G1 workflow example

`examples/vla_g1_workflow.py` chains record (LeRobotDataset), fine-tune
(Isaac-GR00T N1.7 via the GR00T trainer), and deploy (SONIC whole-body control
via `WBCPolicy`) on the Unitree G1 humanoid, with a `docs/training/vla_workflow.md`
page covering prerequisites and per-stage references. The default path runs in
~10s on CPU with no external services (mock policy + stub ONNX session), proving
the three components compose; `--tune` enables real fine-tuning (Docker + GPU)
and `--checkpoint` deploys an existing SONIC checkpoint directly. Closes #471.


### Fixed: `add_object` rejects unknown keyword arguments instead of silently dropping them

`Simulation.add_object` declared `**kwargs` on both the MuJoCo and Newton
backends, documented as "reserved for backend-specific extensions; currently
ignored". In practice the two signatures are identical and carry no
backend-specific parameters, so the `**kwargs` only served to swallow caller
mistakes. Passing MuJoCo's native `add_object(rgba=[1, 0, 0, 1])`, or a typo
such as `colour=`/`radius=`, silently discarded the argument and returned
`{"status": "success"}` while creating a default-grey object -- the
"success contract, wrong physical effect" failure mode the project forbids
elsewhere ("never warn-and-continue; no silent defaults"). The agent-dispatch
router (`sim(action="add_object", ...)`) reproduced the same silent drop because
it skips its "Unknown parameter" guard for `**kwargs` methods.

`add_object` now takes only its declared parameters. The agent-dispatch path
reports a structured error naming the offending parameter and listing the valid
ones (`rgba` -> use `color`); a direct Python call raises `TypeError`. The
constructor's and `create_world`'s `**kwargs` are unchanged -- those carry
genuine cross-backend forward-compat parameters (`num_envs`/`device`) that a
parallel backend consumes and MuJoCo ignores.


### Security: Hardened MuJoCo `render(output_path=...)` against path traversal, symlink, oversize, and partial-write corruption

`render(output_path=...)` is an LLM-callable tool, so its destination path is
attacker-influenced. The previous guard was a metacharacter blacklist plus a
`/`-only `..` check, which still accepted absolute paths (`/etc/cron.d/x`),
backslash traversal (`..\..\etc\passwd`), and symlinked targets, and wrote
non-atomically with no size cap. Writes are now confined to a sandbox root
(`STRANDS_ROBOTS_RENDER_ROOT`, default `~/.strands_robots/renders`); paths that
resolve outside it, use backslash separators, carry shell metacharacters, or
point at a symlink are rejected with `status=error`. Set
`STRANDS_ROBOTS_RENDER_ALLOW_ABS=1` to opt out of the sandbox for absolute
paths. PNGs larger than `STRANDS_ROBOTS_RENDER_MAX_BYTES` (default 50 MB) are
refused without writing. The write is atomic (`tempfile.mkstemp` + `os.replace`),
so a crash mid-write cannot corrupt an existing file; created files are `0o644`
and freshly created directories `0o755`.


### Security: Extended output-path hardening to `run_policy(video=...)` and `start_cameras_recording`

`render(output_path=...)` was hardened in isolation, but the sibling video sinks
that also persist an LLM-supplied filesystem path were still unguarded:
`run_policy(video={"path": ...})` and `start_cameras_recording(output_dir=...,
name=...)` did `os.path.abspath` + `os.makedirs` on the raw path (and
interpolated the raw `name` tag into the per-camera filename), so a `..`
traversal, a symlinked target, shell metacharacters, or a `name` carrying path
separators could escape the intended location. The guards are now centralized in
`strands_robots.simulation.safe_output` and shared by all three sinks: `..`
traversal segments, backslash separators, shell metacharacters, and symlinked
targets are rejected with `status=error` before any file is opened, and the
recording `name` is validated as a single path component. Unlike `render`, whose
sandbox is on by default, the video sinks preserve their historic
absolute-path contract: confinement is opt-in via `STRANDS_ROBOTS_VIDEO_ROOT`
(with `STRANDS_ROBOTS_VIDEO_ALLOW_ABS=1` to re-permit absolute paths inside that
mode). The `render` implementation now delegates to the shared helpers; its
behavior and env vars (`STRANDS_ROBOTS_RENDER_*`) are unchanged.


### Added: `BaseRLAlgo.evaluate()` - deterministic eval peer of `train()`

The from-scratch RL trainers (`PpoTrainer`, `FastSacTrainer`) could `train()` a
policy but had no first-class way to *score* it: callers had to hand-roll a
rollout loop and risk scoring the stochastic (exploration) action or an
un-frozen normalizer, neither of which matches what a deployed `policy.pt`
produces. `BaseRLAlgo.evaluate(spec=None, checkpoint_dir=None, num_episodes=10)`
now rolls out the DETERMINISTIC (mean) action with gradients disabled and
observation normalization frozen, and returns
`{num_episodes, mean_return, std_return, min_return, max_return, mean_length,
success_rate, returns}`. It works both on a live post-`train()` instance (reuses
the in-memory env + network) and on a fresh instance (builds the env from `spec`
and optionally loads `policy.pt` via the new `load_checkpoint()`); the default
`_deterministic_action()` dispatches to the actor-critic's `act_inference`, so
PPO and FastSAC share one implementation with no per-subclass override.
`success_rate` is the fraction of episodes ending on a genuine terminal (the
env's `success_fn`), not a time-out. Purely additive - no existing behaviour
changes.

### Added: `VecSimEnv` - N independent `SimEnv` presented as one `(N, D)`-batched env

The single-env `SimEnv` emits `(1, D)` tensors by design ("only the env count
changes"); `VecSimEnv` is the realisation of that promise for the CPU/MuJoCo
backend. It owns `num_envs` independent `SimEnv` (each its own engine), steps
them through one reused thread pool (MuJoCo releases the GIL during `mj_step`,
so threads give real parallelism on the physics call), and stacks the results
to `(N, D)` so the from-scratch PPO / FastSAC trainers can collect N
trajectories per step. Autoreset matches the gymnasium vector API: on a done
env the pre-reset terminal observation is captured into
`infos[i]["terminal_obs"]` before reset (load-bearing for value bootstrapping a
truncation), and the returned `obs[i]` is the fresh post-reset observation. A
construction-time homogeneity guard rejects sub-envs that disagree on
obs/action dims, and `num_envs=1` skips the thread pool entirely. Exported from
`strands_robots.training.rl`. A future GPU-batched backend can implement this
same interface as one engine driving N worlds, so trainer code written against
`VecSimEnv` does not change when the backend does.

### Added: LeKiwi is now simulatable (`Robot("lekiwi", mode="sim")`)

The `lekiwi` registry entry was hardware-only - it carried a `hardware.lerobot_type`
mapping but no `asset` block, and `robot_descriptions` ships no lekiwi module, so
`Robot("lekiwi", mode="sim")` failed with `No model found for 'lekiwi'`. The entry
now points at the Apache-2.0 [Ekumen-OS/lekiwi](https://github.com/Ekumen-OS/lekiwi)
MuJoCo description via a GitHub asset `source` (6-DOF SO-ARM arm on a 3-omniwheel
base, 9 actuators). LeKiwi auto-downloads and compiles on first sim use, steps
stably, and renders from its `front`/`wrist` cameras. The existing hardware mapping
is unchanged.

### Added: routing-degradation telemetry so a silently-degraded LeRobot rollout is machine-detectable

`LerobotLocalPolicy`'s heuristic (non-declarative) remap path keeps a rollout alive even when it cannot bind the observation to the model's inputs by name: a camera whose name matches no declared image feature is routed to a free slot positionally, and `observation.state` is composed from the observation's own scalar keys when none of `robot_state_keys` match (the generic `joint_0..N` fallback). Either makes the robot move on meaningless inputs while `run_policy` / `eval_policy` still report `status="success"` with `success_rate ~ 0`, and the only trace was a log line (the positional fallback on the preprocessor path did not even warn). Both fallbacks now flip a flag on the policy (`positional_fallback_used` / `generic_state_keys_used`) and emit a WARNING, and `run_policy` / `eval_policy` surface both flags in their JSON result block alongside the existing `action_errors` / load telemetry. A `True` flag on an otherwise-successful run is the signature of a misconfigured camera/state binding. No behaviour change for correctly-named observations (both flags stay `False`); the remap itself is unchanged.

### Fixed: LerobotLocalPolicy state-key mismatch is now loud instead of a silent zero/open-loop rollout

When `LerobotLocalPolicy` auto-generated generic state keys (`joint_0..N`, from
the model action dim) but the sim/robot reported named joints (`shoulder_pan`
...), none of the configured `robot_state_keys` matched the observation. Both
observation-to-batch paths handled this silently: the preprocessor/VLA path
(`_to_lerobot_observation`) fell back to the observation's scalar keys with no
warning, and the strands-native path (`_build_batch_from_strands_format`)
dropped `observation.state` entirely. The model then ran conditioned on a
zero/missing state - effectively open-loop - while reporting `action_errors=0`,
`success_rate=0`, and no error log. The mismatch is now surfaced through the
existing `strict_keys` knob: with `strict_keys=True` it raises a `ValueError`
that names the actual observation keys and points at `embodiment=` /
`set_robot_state_keys()`; with the default `strict_keys=False` it logs a single
WARNING with the same guidance and then falls back to the observation's own
scalar keys, so the state is populated rather than silently zeroed or dropped.


### Removed: the `strands_robots.planning` package - locomotion intent is `policy_kwargs`

The `strands_robots.planning` package (a `Planner` ABC, `PlannerCommand`,
`KinematicPlanner`, and keyboard / gamepad / agent / scripted `InputSource`s)
re-implemented a locomotion goal channel that already exists: the well-known
`policy_kwargs` keys (`target_velocity` / `target_height` / `locomotion_style`)
that `run_policy` forwards verbatim to the policy. It is removed in full, along
with the `planner=` parameter on `SimEngine.run_policy` / `PolicyRunner.run` and
the `[planning]` (pygame) extra. Locomotion intent is now expressed only through
`run_policy(policy_kwargs={...})`; a caller (including an `Agent(tools=[robot])`)
steers by re-issuing short-horizon `run_policy` calls, closing the loop at its
own cadence. The MotionBricks policy still consumes `locomotion_style` from
`policy_kwargs` (a plain string contract, never a planner import): its
planner-named symbols were renamed to the goal-kwarg framing
(`resolve_planner_style` -> `resolve_locomotion_style`,
`PLANNER_STYLE_TO_G1_CLIP` -> `LOCOMOTION_STYLE_TO_G1_CLIP`) and the accepted
style vocabulary is now owned by MotionBricks as `LOCOMOTION_STYLES`. The
`examples/planner/` scripts moved to `examples/locomotion/` as self-contained
demos that own their own goal state. The accepted `locomotion_style` vocabulary
and clip mapping are byte-for-byte unchanged.

### Fixed: unknown LeRobot policy types raise a clean ImportError instead of leaking lerobot's internal ValueError

`resolve_policy_class_by_name()` documents that it raises `ImportError` (naming
the type and the strategies it tried) when no policy class can be resolved. Its
legacy-factory rung (`lerobot.policies.factory.get_policy_class`) caught
`ImportError`, `AttributeError`, `RuntimeError`, and `TypeError` so a strategy
that is merely unavailable falls through to the clean `ImportError` -- but it
did NOT catch `ValueError`. Current lerobot ends `get_policy_class` with
`raise ValueError(f"Policy type '{name}' is not available.")` for every name it
does not recognise, so resolving an unknown type -- or one of lerobot's internal
building-block modules that live under `lerobot.policies` but are not registered
policies (e.g. `pi_gemma`, the PaliGemma layers used by pi0/pi05) -- leaked that
bare `ValueError` to the caller and broke the documented contract. `ValueError`
is now included in the rung's caught set, so resolution falls through to the
actionable `ImportError` as documented.


### Fixed: lerobot-availability probes no longer cache a transient failure

`has_lerobot_dataset()` (dataset recording) and `has_streaming_dataset()`
(streaming datasets) wrapped their `from lerobot... import ...` probe in
`functools.lru_cache`, which froze the FIRST result -- including a `False` --
for the life of the process. lerobot availability is a process capability that
can transiently fail to resolve: the probe deliberately catches `ImportError`,
`ValueError`, and `RuntimeError` precisely because a partially-installed env
(e.g. the documented JetPack/Jetson numpy-ABI mismatch) or a temporarily
shadowed `sys.modules` entry can raise mid-run. A single such failure
permanently disabled recording/streaming -- `start_recording` would return
`requires the lerobot extra` forever even after the condition cleared. Both
probes now cache only the POSITIVE result (the expensive import is still done at
most once) and re-attempt the import on the next call after a failure, restoring
recording the moment lerobot resolves again.


### Fixed: Newton recording captures camera frames when the policy skips images

`PolicyRunner` sets `skip_images = not policy.requires_images` to avoid rendering
when the policy does not consume pixels. The default `mock` policy (and any
non-VLA, proprioceptive-only policy) reports `requires_images=False`, so during a
rollout the runner asks the backend for a pixel-free observation. While a dataset
recording is active that hint must be overridden, because the recording `on_frame`
hook writes the observation's camera ndarrays into the dataset's declared video
features -- a pixel-free observation yields a dataset with correct episode counts
but no pixels (the video-modality sibling of the mega-episode corruption class).
The MuJoCo backend guards this in `get_observation`; the Newton backend wired the
recorder later (named cameras + `DatasetRecorder`) and honored `skip_images`
literally, so recording on Newton with a non-image policy silently dropped every
frame's images. `NewtonSimEngine.get_observation` now applies the same guard:
when an active recording is attached it renders cameras regardless of the
`skip_images` hint, restoring MuJoCo parity. Inference still skips rendering when
nothing is recording.


### Fixed: policy resolution no longer shadows `lerobot.policies` with a partial stub

Smart-string resolution (`create_policy("org/model")`, `grpc://`, `ws://`, ...)
calls an internal helper that registers `lerobot.policies` in `sys.modules`
without executing its heavy `__init__` (which pulls in the groot/transformers
chain that can crash on a flash-attn ABI mismatch). The helper installed a
lightweight `__path__`-only stub *unconditionally* and never removed it, so on a
healthy install the stub permanently shadowed the real package: any later
`from lerobot.policies import PreTrainedPolicy` / `get_policy_class` -- including
the imports lerobot's own `lerobot_record` / `lerobot_rollout` scripts (and the
`lerobot_teleoperate` tool that wraps them) perform -- failed with
`ImportError: cannot import name 'PreTrainedPolicy' from 'lerobot.policies'` for
the rest of the process. The stub is now a fallback: the real package is imported
first and only when its `__init__` genuinely fails do we install the stub.


### Added: `PersistentPolicy` + cache controls, `policy_resident_rss_mb` telemetry [major-perf]

A synchronous persistent worker that eliminates per-episode model-reload
overhead for multi-rollout loops. Loading a VLA / LeRobot checkpoint (a
MolmoAct2 SO-100/101 build reads ~1300 weight files into GPU memory) costs a
minute or two; a loop that rebuilds the policy per episode pays it every time
and its resident memory oscillates as the model is loaded and dropped.

- `strands_robots.policies.PersistentPolicy(provider, **config)` - a thin,
  thread-safe wrapper that builds the underlying policy ONCE and is passed to
  every `run_policy`/`eval_policy` via `policy_object=` for zero-reload reuse.
  It delegates the full `Policy` contract (chunk-shape introspection, RTC /
  control-frequency hooks, `reset`, load telemetry) so the runtime drives it
  exactly as the bare policy. Concurrent inference on one shared handle is
  serialised behind a per-call lock.
- `policies.preload(provider, **config)` warms the process-level model cache and
  reports `load_time_s`, `load_cache_hit`, `resident_rss_mb`, `rss_delta_mb`,
  plus the ready `PersistentPolicy`. `policies.list_cached()` introspects what is
  resident; `policies.evict(pretrained_name_or_path=None)` frees one checkpoint
  (or all) - `clear_model_cache` now accepts an optional checkpoint filter.
- `run_policy`/`eval_policy` JSON now also carries `policy_resident_rss_mb`
  (process RSS in MB at result time; `None` when unmeasurable). A flat value
  across episodes confirms the model stays resident rather than oscillating - the
  per-episode-reload smell, complementing the existing `policy_load_cache_hit`.

Memory note: the cache shares the SAME live `nn.Module` across instances of one
`(checkpoint, device)` key. Sequential reuse is safe (`Policy.reset()` clears
per-episode state between episodes); `PersistentPolicy`'s lock makes concurrent
reuse safe too. Opt out with `create_policy(..., cache_model=False)` for an
independent live copy.
### Added: Newton backend domain randomization + sensor-noise hooks

The Newton (GPU) backend gained `randomize()` and `set_obs_noise()`, the
sim2real pieces it was missing so datasets collected on it do not overfit to
the default physics constants. `randomize()` mirrors the MuJoCo backend's
contract (same keyword names and `randomize_physics=False` default) for the
axes Newton supports: per-shape colors (`shape_color`), directional-light
orientation, and physics - per-body mass + inertia (`body_mass` /
`body_inertia`, scaled together so Newton recomputes the inverse mass/inertia
at finalisation) and per-shape friction (`shape_material_mu`). Multipliers are
applied to the `ModelBuilder` before the immutable model is finalised and the
model is rebuilt. A fixed `seed` yields an identical multiplier sequence for a
given scene (the builder visits bodies/shapes deterministically); the applied
`mass_scales` / `friction_scales` / `light_direction` are returned in the
`json` block so callers can log or assert per-episode physics.
`randomize_positions` is not supported yet and returns an explicit error rather
than a silent no-op.

`set_obs_noise(joint_pos_std=, joint_vel_std=, camera_jitter_px=, seed=)` adds
reproducible additive Gaussian noise to `get_observation` joint positions,
`get_robot_state` positions/velocities, and rendered camera frames (integer
pixel jitter). `SimEngine` gained a `set_obs_noise` optional override (default
`NotImplementedError`) alongside `randomize`, and the Newton `describe()`
advertises both methods.

### Fixed: RTC inference now forwards `inference_delay` to lerobot's denoiser

`LerobotLocalPolicy._predict_with_rtc` passed only `prev_chunk_left_over` and
`execution_horizon` to lerobot's `predict_action_chunk`, omitting the
`inference_delay` kwarg that the RTC denoiser (pi0, pi0.5, SmolVLA) requires. It
is the `start` argument of `RTCProcessor.get_prefix_weights(start, end, total)`,
which freezes the first `d` committed actions of the new chunk to the previous
chunk's prefix and linearly blends `d..execution_horizon`. With it omitted the
denoiser received `inference_delay=None`, and `min(None, end)` raised
`TypeError: '<' not supported between instances of 'int' and 'NoneType'` the
moment a previous-chunk prefix existed (the 2nd inference onward) - so a real
flow-matching RTC policy crashed on its second chunk. The existing unit tests
missed it because they mock `predict_action_chunk` with a static tensor that
ignores its kwargs.

The wrapper now resolves the delay (the deterministic count from
`set_rtc_observed_delay`, else a wall-clock p95 estimate over PRIOR calls)
BEFORE inference and forwards it as `inference_delay` on every call, so the
prefix-attention guidance is computed with the correct freeze count and the
chunk seam blends identically in sim and on hardware. A regression test routes
the wrapper's exact kwargs through lerobot's real `RTCProcessor.denoise_step`
and fails (TypeError) on the pre-fix code.
### Fix: re-anchor the RTC chunk-seam prefix for relative-action policies

Real-Time Chunking for relative-action flow checkpoints (pi0 / pi0.5 / pi0-FAST
trained with a `RelativeActionsProcessorStep`) carried the previous chunk's
unexecuted tail (`prev_chunk_left_over`) in the coordinate frame of the
observation that produced it. Because these policies predict actions as offsets
from the current robot state, and the state moves between chunks, the stale-frame
prefix was blended into the next chunk in the wrong frame - off by the full state
drift at the seam.

`LerobotLocalPolicy` now keeps the leftover in absolute coordinates and
re-expresses it against the live robot state on every query via LeRobot's own
`reanchor_relative_rtc_prefix` helper (reading the cached state from the
preprocessor's `RelativeActionsProcessorStep`), instead of re-implementing the
rebase. Behaviour:

- Detected automatically from the loaded preprocessor pipeline - engages only
  when an enabled relative-action step is present; no flag needed.
- Absolute-action policies are unchanged: they carry the model-space leftover
  verbatim (their frame does not move).
- The deterministic step-count inference delay is untouched, so RTC stays
  bit-reproducible across fixed-seed episodes.
- Falls back to the prior behaviour with a one-time warning when the installed
  lerobot lacks a usable `reanchor_relative_rtc_prefix` - whether the symbol is
  absent (ImportError, <= 0.5.1) or `import lerobot.policies.rtc` raises at load
  time (TypeError on lerobot 0.5.1, whose rtc import chain builds a broken
  dataclass). The guard tolerates both so a relative-action policy on 0.5.1
  degrades gracefully instead of crashing.
- `ProcessorBridge.preprocessor_steps` exposes the pipeline steps so the RTC
  consumer can introspect for the relative/normalizer steps.


### Feature: full RewardModelConfig parity with lerobot (dynamic reward-type discovery)

`LerobotTrainer` reward-model training (`TrainSpec.extra["reward_model"]`) now
reaches EVERY reward model lerobot registers on its `RewardModelConfig` choice
registry, not just SARM. The valid type set and each type's configurable fields
are read live from `RewardModelConfig.get_known_choices()` and the resolved
config dataclass (the same zero-maintenance discovery Robot / Teleop / Camera /
Policy already use), so `robometer`, `topreward`, and `reward_classifier` - plus
any future or plugin-registered reward type - validate and build with no strands
change:

- The previously hardcoded reward-type list and SARM-biased friendly-key set are
  gone. Friendly `extra["reward_model"]` keys are now the chosen type's OWN
  config fields, so each type is configurable with its own knobs (e.g.
  robometer's `default_task` / `success_threshold`, the classifier's
  `num_classes`). Cross-type fields (e.g. SARM's `annotation_mode` on
  `robometer`) are rejected with the list of that type's configurable fields.
- A static fallback (SARM keys, the four known types) keeps `validate()`
  informative when `lerobot.rewards` is absent (lerobot < 0.5.2), where
  reward-model training cannot run anyway.
- Added a source-AST parity guard that scans the installed lerobot's reward
  sources for `register_subclass` sites and asserts strands' discovery matches
  exactly; it self-skips when `lerobot.rewards` is unavailable.


### Feature: Newton backend scene-discovery and per-joint state parity

The Newton GPU backend gained the discovery and state-introspection methods the
MuJoCo backend already exposes, so agents can introspect a Newton world without
guessing method names and `Robot(..., backend="newton")` reaches parity for
scene queries:

- `get_robot_state(robot_name=None)` returns per-joint `position` (from
  `joint_q`) and `velocity` (from `joint_qd`). A dedicated DOF index is now
  tracked alongside the coordinate index so velocities stay correct even when a
  free-floating object adds a quaternion coordinate (coord count != DOF count).
- `list_robots_info()`, `list_objects()`, `list_bodies(robot_name=None)` (with
  best-guess `gripper_body` resolution), and `get_features(robot_name=None)`
  return the same agent-tool dict shapes as the MuJoCo backend.
- `move_object(name, position=None, orientation=None)` repositions an object and
  rebuilds the model, preserving live joint targets.
- `list_urdfs()` / `register_urdf(data_config, urdf_path)` read/write the shared
  model registry (with path validation on register).
- `describe()` now advertises these methods plus the body labels and the single
  `default` camera so one call surfaces the full contract.


### Feature: async-RTC latency masking is the default for chunk-emitting policies

`run_policy` / `PolicyRunner.run` previously defaulted to `async_rtc=False`, so
every chunk-emitting VLA (pi0, pi0.5, pi0-FAST, SmolVLA, MolmoAct2) paid the full
inference latency at every chunk seam in sim even though the overlap pipeline
already existed - it was invisible to callers who did not know to pass the flag.

- `async_rtc` now defaults to `None` (auto-resolve). `None` reads the new
  `Policy.is_chunk_emitting()` to enable the inference/execution overlap for
  chunk-emitting policies and keep single-step policies (MockPolicy, classical
  planners) on the synchronous loop. An explicit `async_rtc=True`/`False` still
  wins. `Policy.is_chunk_emitting()` defaults to `execution_horizon > 1`;
  `LerobotLocalPolicy` also reports `True` for an RTC model or a checkpoint that
  must be driven via `predict_action_chunk` (MolmoAct2).
- The run-result `{"json": {...}}` block now carries six async-RTC telemetry
  fields - `rtc_async_enabled`, `rtc_chunks_acquired`, `rtc_prefetch_hits`,
  `rtc_prefetch_blocks`, `rtc_avg_inference_ms`, `rtc_max_inference_ms` - so
  latency masking is provable from the payload instead of from logs.
- Seam hardening: an empty prefetched chunk degrades to one synchronous
  re-query before erroring (a transient hiccup no longer kills the rollout); a
  blocking prefetch logs a starvation warning; and the new
  `rtc_inference_timeout_s` argument bounds a stuck inference, returning a
  structured `status="error"` result (with telemetry) instead of waiting for
  every remaining chunk.

Behaviour change: a chunk-emitting policy run through `run_policy` without an
explicit `async_rtc` now uses the overlap pipeline, which fires one extra
background `get_actions` for the trailing chunk. Tests that assert the exact
synchronous re-query count should pass `async_rtc=False`.


### Observability: `lerobot_local` model-load telemetry

The process-level model cache already skips the expensive `from_pretrained`
weight read when the same checkpoint is re-instantiated, but the saving was
invisible: a warm reuse and a cold reload produced identical result text, so a
caller paying a per-episode reload had no machine-checkable signal.

- `LerobotLocalPolicy` now records `load_cache_hit` (bool) and `load_time_s`
  (float) after each load, on both the generic and MolmoAct2 load paths.
- `Simulation.run_policy` and `Simulation.eval_policy` surface these as
  `policy_load_cache_hit` and `policy_load_time_s` in their `{"json": {...}}`
  result block. A `policy_load_cache_hit=False` on episode 2+ of a loop is a
  smell that the caller rebuilt the policy per episode instead of reusing one
  warm `policy_object=`. Policies without load telemetry (e.g. `MockPolicy`)
  report the honest defaults `0.0` / `False`.
- New `list_cached_models()` read-only introspection helper (exported from
  `strands_robots.policies.lerobot_local`) reports the resident cache entries
  (`namespace`, `pretrained_name_or_path`, `device`, `policy_class`) without
  exposing the private cache dict, complementing `clear_model_cache()`.

Memory note: the cache is unchanged; this is observability only. The cached
model is still one live module shared across instances with the same load key,
so concurrent (not sequential) reuse should still opt out with
`cache_model=False`.

### Fixed: hardware control loop honours the RTC contract (sim/real parity)

`HardwareRobot._execute_task_async` (the real-robot rollout loop) drove the arm
with a raw `robot_actions[:action_horizon]` slice and never told the policy its
control clock, so RTC-capable providers (pi0, pi0.5, SmolVLA, MolmoAct2) fell
back to a hardcoded assumed rate and mis-blended every chunk seam at any control
frequency except the assumed one - a policy validated in sim behaved differently
on the physical arm. The loop now mirrors the synchronous sim runner contract:

- `policy.set_control_frequency(control_frequency)` is called once before the
  rollout so latency-sensitive providers convert inference latency into the
  correct count of action steps.
- `policy.set_rtc_observed_delay(0)` is set before each inference. The hardware
  loop is synchronous (observe -> infer -> apply) and issues no servo motion
  during inference, so exactly 0 control steps elapse - a counted, deterministic
  seam offset, not a wall-clock estimate. Drivers that coast servos during
  inference would supply a non-zero counted delay here instead.
- Chunk consumption now goes through `resolve_chunk_length(policy,
  action_horizon)` instead of the raw slice: RTC policies are re-queried at
  exactly their `execution_horizon` (so cross-chunk blending engages) while
  single-step and open-loop chunked providers keep their prior
  `max(action_horizon, execution_horizon)` behaviour (a no-op for them).

Sim and hardware now share one RTC contract.

### Correctness: `run_policy` episode-count contract + `verify_dataset_episodes`

`run_policy` could silently record a single merged `episode_index=0`
mega-episode while a caller believed it had collected N distinct episodes. The
single-episode fast path returned none of the episode-count bookkeeping the
multi-episode path did, and a `status="success"` rollout was no proof of
episode-count correctness - frames buffer into the current episode and only
flush to parquet at `save_episode`/`stop_recording`. An agent that calls
`run_policy(n_episodes=1)` (the default) once but narrates "20 episodes" got a
1-episode dataset and no signal that anything was wrong.

- Both `run_policy` paths (single-episode fast path and the multi-episode
  driver) now return the episode-count truth fields in their `{"json": {...}}`
  block: `n_episodes_requested`, `n_episodes_completed`, `episodes_saved`, and
  `dataset_episode_indices` (the episode indices the active recorder reports so
  far). The fast path additionally sets `episode_flush_deferred=True` while
  recording, making explicit that its rollout buffers into one episode that is
  flushed at `stop_recording` rather than saved as a distinct boundary.
- The fast path logs a single `INFO` line at run start when a recording is
  active (`n_episodes=1, will produce 1 dataset episode ...`) so an agent driving
  the tool sees the truth in its output and self-corrects on the next call.
- New `SimEngine.verify_dataset_episodes(expected)` reads the on-disk LeRobot
  parquet (the ground truth) after `stop_recording` and returns `status="error"`
  when the recorded episode count differs from `expected`, with
  `{expected, actual, episode_indices, total_frames, total_frames_per_ep, root}`
  in the json block. Backed by the new pure-`pyarrow`
  `strands_robots.dataset_recorder.read_dataset_episode_indices(root)` helper,
  which parses `meta/episodes/**/*.parquet` without importing `lerobot`.
- The `run_policy` `n_episodes` docstring and the MuJoCo `describe()` recording
  surface now spell out the contract: pass `n_episodes=N` for N distinct
  episodes, do not loop the call, and confirm with `verify_dataset_episodes`.

No change to recorded trajectory content, video output, or the
single-rollout payload shape (the per-episode `episodes` aggregate list is still
absent on the fast path); only additive `{"json": {...}}` fields and a new
verification method.


### Refactor: unify the rclpy + RTPS hardware ROS 2 bridges under `RosTelemetryBase`

The two hardware ROS 2 transports now share a single source of truth for the
wire contract. Previously `HardwareRosBridge` (rclpy) subclassed
`RosTelemetryBridge` while the pure-RTPS bridge was a parallel implementation
that re-derived the topic names, name sanitization, and inbound `joint_command`
parsing independently - so the "byte-identical topics" guarantee depended on two
codepaths staying in sync by hand.

- New `strands_robots.ros_telemetry.RosTelemetryBase` owns the transport-agnostic
  contract: topic names (`joint_states_topic` / `image_topic` /
  `joint_command_topic`), the `_safe` segment sanitizer, robot-name resolution,
  and `joint_command` -> `send_action` dispatch. `RosTelemetryBridge` (and its
  `SimRosBridge` / `HardwareRosBridge` subclasses) and the RTPS bridge now all
  derive from it, so the rclpy and cyclonedds transports are identical on the
  ROS 2 graph by construction rather than by convention.
- The pure-RTPS bridge class is renamed `RtpsHardwareBridge` -> `HardwareRtpsBridge`
  to match its rclpy sibling `HardwareRosBridge` (both `Hardware*Bridge`). The
  module path (`strands_robots.hardware_rtps_bridge`) and the public
  top-level export are unchanged except for the class name; `Robot(...,
  ros2_transport="rtps")` selection is unaffected.

No wire behavior changes: published topics, message layouts, and the
`joint_command` contract are identical before and after.


### Feature: zero-config robot discovery from `robot_descriptions` (`registry`)

`Robot("iiwa14", mode="sim")` (and every other MJCF robot shipped by
`robot_descriptions`) now resolves without a hand-written `robots.json` entry.
Previously a standard Menagerie robot had to be re-declared in the curated
registry before the MuJoCo backend could load it, even though `robot_descriptions`
already resolves its assets canonically - so the long tail (`iiwa14`, `gen3`,
`viper`, `widow`, `so_arm101`, ...) was unreachable without a registry edit.

- New `strands_robots.registry.discovery` module. `discover_robot(name)`
  synthesizes a registry-shaped entry for any MJCF-capable `robot_descriptions`
  robot by reading the module's `MJCF_PATH` / `PACKAGE_PATH`, so the existing
  asset-download + resolution pipeline handles it unchanged. `descriptions_module`,
  `is_discoverable`, and `list_discoverable` are cheap lookups (no import, no
  network) for probing the long tail; `discover_robot` is consulted only by the
  download-capable asset resolver.
- The curated `robots.json` always wins: discovery fills the gap only for names
  unknown to the curated registry, so existing robots, joint maps, hardware
  ports, and aliases are unaffected. `robots.json` stays the place for any robot
  that needs project-specific metadata.
- `is_discoverable` / `list_discoverable` are re-exported from the top-level
  `strands_robots` package and from `strands_robots.registry`. The `Robot()`
  factory accepts a discoverable name instead of raising "Unknown robot".

### Feature: SARM reward-model training + the RA-BC production loop (`training`)

Closes the *producing* half of Reward-Aligned Behavior Cloning: `strands-robots`
could already *consume* a SARM progress parquet (RA-BC sample weighting) but
could not *produce* one. `LerobotTrainer` now trains a reward model and a new
`strands_robots.training.reward` module computes the progress weights and serves
the trained model for inference, so the full SARM -> RA-BC -> policy loop is a
sequence of strands calls. Grounded against lerobot >= 0.5.2, where SARM moved to
the `lerobot.rewards` package and sample weighting moved to a nested
`SampleWeightingConfig`.

- `LerobotTrainer` trains a reward model (SARM) when
  `TrainSpec.extra['reward_model']` is set: it builds `cfg.reward_model`
  (and leaves `cfg.policy` unset) so lerobot follows its
  `is_reward_model_training` path through the same in-process `train(cfg)`.
  The dict accepts `type` (default `sarm`), `annotation_mode`
  (`single_stage` / `dense_only` / `dual`), `image_key`, and `state_key`.
  Policy-only knobs (`sample_weighting`, `relative_actions`, non-`full`
  `method`) are rejected on a reward-model run rather than silently ignored.
- New `compute_rabc_weights(reward_model_path, dataset_root|dataset_repo_id, ...)`
  runs a trained SARM over a dataset in-process and returns the path to the
  produced `sarm_progress.parquet` - the file RA-BC sample weighting consumes.
  Running in-process avoids the always-on Hub upload of lerobot's CLI entry
  point.
- New `load_reward_model(model_path, reward_type='sarm', device=...)` +
  `reward_progress(model, batch)` expose a trained reward model for inference,
  returning a dense task-progress score in `[0, 1]` as plain floats (usable as
  an eval-time success/score signal).
- RA-BC sample weighting now targets lerobot's nested `SampleWeightingConfig`
  (`cfg.sample_weighting`) instead of the flat `use_rabc` / `rabc_*` fields,
  which lerobot >= 0.5.2 removed. Without this the `extra['sample_weighting']`
  path raised "no 'use_rabc'" against current lerobot and RA-BC was unreachable.
  The friendly keys (`type` / `progress_path` / `head_mode` / `kappa` /
  `epsilon`) map 1:1 onto the config fields; `type` accepts `rabc` or `uniform`.
- `compute_rabc_weights`, `load_reward_model`, and `reward_progress` are exported
  from `strands_robots.training`.
- Both RA-BC consumption (`extra['sample_weighting']`) and SARM reward-model
  training degrade gracefully on a lerobot older than 0.5.2: `build_config`
  raises an actionable "requires lerobot >= 0.5.2" `ValueError` instead of
  leaking a raw `ModuleNotFoundError`, and the lerobot-0.5.2-only tests skip
  (via `importorskip`) so the suite is green on the published lerobot.


### Internal Refactor: unified chunk-length rule across all policy runners (`ChunkedPolicy`)

A chunk-emitting policy (ACT, diffusion, pi0, SmolVLA, MolmoAct2) returns N
actions per `get_actions` call and expects all N consumed open-loop before a
re-query. Three consumers sized that chunk independently and had drifted: the
single-policy `PolicyRunner.run` and the multi-episode eval loop consumed
`max(action_horizon, policy.actions_per_step)`, but the synchronized
`run_multi_policy` loop truncated to `action_horizon` alone. A chunk-emitting
policy driven through `run_multi_policy` therefore had its chunk tail dropped
and was re-queried out-of-distribution (e.g. an `actions_per_step=30` policy run
with the default `action_horizon=8` re-queried every 8 steps instead of 30),
diverging from `run_policy` for the same policy.

- New `ChunkedPolicy` runtime-checkable `Protocol` in
  `strands_robots.policies.base` declares the chunk introspection contract
  (`actions_per_step: int`, `supports_rtc: bool`). Consumers can branch on
  `isinstance(policy, ChunkedPolicy)` and a type checker rejects a non-chunked
  policy where a chunked one is required.
- New `resolve_chunk_length(policy, action_horizon)` helper centralizes the one
  chunk-sizing rule (`max(action_horizon, actions_per_step)`, single-action
  policies default to 1). `PolicyRunner.run`, the multi-episode eval loop, and
  `run_multi_policy` now all route through it, so the rule can no longer drift
  per consumer.
- `run_multi_policy` now honours a policy's `actions_per_step`, matching
  `run_policy`. A chunk-emitting policy keeps its full trained chunk in the
  synchronized multi-robot loop.
- `LerobotLocalPolicy` exposes `supports_rtc` publicly (over its internal RTC
  state) so it satisfies the `ChunkedPolicy` contract; `actions_per_step` was
  already public.
- Behaviour is unchanged for single-action policies (`mock` and friends) and
  for `run_policy`, whose chunk sizing was already correct. `ChunkedPolicy` and
  `resolve_chunk_length` are exported from `strands_robots.policies`.


### Added: `async_rtc` chunk pipeline for `run_policy` (overlap inference with execution)

`run_policy` / `PolicyRunner.run` drove policies through a synchronous
chunk-then-drain loop: query the policy, fully execute the returned action
chunk, then re-query. Inference and execution never overlapped, so a
chunk-emitting VLA (pi0, SmolVLA, MolmoAct2) showed a per-seam stall in sim that
it would NOT show on real hardware, where an async controller hides inference
latency behind chunk execution. RTC's seam blending worked, but its second
benefit - latency masking - was invisible in sim, making sim timing diverge from
real-hardware timing and RTC-tuned policies look worse than they are.

- `run_policy` and `PolicyRunner.run` accept `async_rtc: bool = False`. When
  `True`, the next `get_actions` is fired on a single background worker once the
  current chunk is ~50% drained (using a fresh mid-execution observation) and
  atomically swapped in when the chunk runs out. A policy whose inference
  latency is at most the chunk's execution window then pays (almost) zero
  visible stall at the seam.
- Provider-agnostic: the runner only schedules the inference/execution overlap;
  it never touches the policy's RTC machinery, so RTC-capable policies still
  blend the seam internally via their own prev-chunk state.
- Thread-safe by construction: the policy is invoked from at most one thread at
  a time (a new prefetch is only submitted after the previous one is consumed),
  the sim is only ever touched from the calling thread, and the runner blocks on
  any in-flight inference before returning so no background thread touches the
  policy or sim after `run_policy` exits.
- Backward compatible: `async_rtc=False` (default) keeps the exact synchronous
  loop. Most useful at real-time pacing (`fast_mode=False`) with multi-step
  chunks, where there is an execution window to hide inference behind.

### Fixed: `reset()` during recording flushes the buffered rollout as an episode

A `run_policy` + `reset` data-collection loop silently merged every rollout into
a single `episode_index=0`: `meta/info.json` reported `total_episodes: 1` and the
episodes parquet held one row even after N distinct rollouts, because the
recorder buffered all rollouts together and only `stop_recording` flushed them.
Downstream training/eval that slices by episode then saw one giant episode
instead of N. `run_policy` alone does not delimit episodes, and `reset` (the
natural between-rollout boundary, since `n_episodes` is an `eval_policy` param,
not a `run_policy` one) did not either.

- `Simulation.reset()` now flushes a pending recording episode before
  re-initializing the world: when a recording is active and the recorder has
  buffered frames, `reset` calls `save_episode()` first and reports the saved
  episode in its result text. This mirrors `stop_recording`, which already
  auto-flushes the trailing episode.
- `save_episode()` remains the explicit boundary primitive; it is a no-op on an
  empty buffer, so `reset` calls that are not preceded by recorded frames (two
  resets in a row, a reset right after `start_recording`, or `eval_policy`'s
  internal per-episode resets, which do not feed the recorder) create no
  spurious empty episodes.
- To DISCARD a partial rollout instead of flushing it, call
  `clear_episode_buffer()` before `reset()`.
- Regression tests pin one-episode-per-rollout for a `run_policy` + `reset`
  loop, the empty-buffer no-op, and the unchanged non-recording reset path; they
  fail on pre-fix code (`total_episodes == 1`).


### Added: stream a Hub dataset into the LeRobot trainer (no full download)

`TrainSpec` and the `train_policy` tool gained a `streaming` flag and a
`dataset_repo_id` field so a LeRobot post-tune can pull frames on the fly from a
Hugging Face Hub dataset instead of materializing it locally first. Previously
the only data source was a local `dataset_root`, so training a large dataset
(BitRobot / HIW-500, ~50-500 GB) required downloading the whole thing before the
first forward pass - blowing disk on an edge node and wasting wall-clock.

- `TrainSpec.dataset_repo_id` (`org/name`) is an alternative data source to
  `dataset_root`; when set, the LeRobot trainer uses it as
  `DatasetConfig.repo_id` and treats `dataset_root` as an optional local cache.
- `TrainSpec.streaming=True` selects lerobot's `StreamingLeRobotDataset`
  (`--dataset.streaming=true`): bounded disk when streaming Hub shards, bounded
  RAM when streaming a local root.
- `LerobotTrainer.validate` now accepts either data source and rejects a
  malformed `dataset_repo_id` (allowlisted `org/name` format) before it reaches
  a Hub URL. Held-out `val_episodes` splitting is a no-op when streaming a Hub
  dataset with no local cache (no local `meta/info.json` to count episodes).
- Other trainers (GR00T, Cosmos3) ignore both fields per the `TrainSpec`
  tolerance rule. Regression tests pin the argv-parity helper and the typed
  `TrainPipelineConfig` for Hub-streaming, local-streaming, and local-root
  paths, and fail on pre-fix code (which hardcoded `repo_id="local"`).

### Fixed: unified "no world" guard contract across the simulation facade

Every world-touching method on the MuJoCo `Simulation` facade returns a
structured error when called before `create_world` (or after a failed
`load_scene`), but the wording had drifted into three different strings -
`"No world. Call create_world first."`, `"No world. Use action='create_world'
first."`, and `"No world. Call create_world (or load_scene) first."` - so an
agent that learned the message from one action did not recognise it from
another. Worse, four methods (`send_action`, `replace_scene_mjcf`,
`patch_scene_mjcf`, `add_robot`) guarded only on `self._world is None` (or
`_model` alone) and so accepted the *partial* world state that `load_scene`
leaves behind when its spec compile fails (`_world` set, `_model`/`_data` still
`None`), then either crashed on a `None` dereference or silently recovered.

- A single module-level `_NO_WORLD_MSG` constant is now the source of truth for
  the guard text. All 19 inline guards and the `_require_world()` helper (whose
  docstring already promised a unified message) return that exact string.
- The four weak guards now check the full `world + _model + _data` predicate,
  matching every other method, so the partial-world state is reported as the
  standard no-world error instead of slipping through.
- Regression tests parametrize all guarded methods over both the no-world and
  partial-world states and assert the exact unified message; the message-drift
  and partial-world-slip both fail on pre-fix code.


### Fixed: simulation render output is ASCII-only (no emoji)

The MuJoCo rendering tool methods embedded emoji and non-ASCII symbols in their
agent-facing text payloads: `render` / `render_depth` prefixed summaries with a
camera emoji, `render_depth` raised a degraded-accuracy warning with a warning
emoji, `render_all` labelled each camera and its empty-frame warning with the
same emoji, and `get_contacts` used a burst emoji plus `bullet`/`<->` arrow
symbols in its per-pair lines. Emoji in tool output corrupts logs and trips
downstream ASCII parsers, and the project contract is ASCII-only for code,
logs, and tool text.

- `render`, `render_depth`, `render_all`, and `get_contacts` now emit plain
  ASCII text. Image/JSON blocks and all numeric values are unchanged; only the
  decorative text prefixes/separators changed (e.g. the contact list now reads
  `ground <-> cube_geom`, the depth warning starts with `Warning:`).
- Regression tests render real frames headlessly and assert every text block of
  `render`, `render_depth` (including the ARB_clip_control warning branch),
  `render_all`, and `get_contacts` (with active contacts) is ASCII-only.


### Fixed: `get_mass_matrix` works across MuJoCo `mj_fullM` signature changes

`PhysicsMixin.get_mass_matrix()` called `mj.mj_fullM(model, M, data.qM)`, the
pre-3.10 argument order. MuJoCo 3.10 reordered the binding to
`mj_fullM(model, data, dst)` (the sparse inertia is read from `data` directly),
so the old call raised `TypeError: mj_fullM(): incompatible function arguments`
on newer MuJoCo. Because the project pins MuJoCo loosely (`>=3.2.0,<4.0.0`),
this surfaced as a hard failure once the resolver picked up 3.10+.

- A new module-level helper `_full_mass_matrix(mj, model, data)` probes the
  modern `(model, data, dst)` signature first, then falls back to the legacy
  `(model, dst, qM)` orders (1D and `[m, 1]` column variants). The `dst` buffer
  is always allocated C-contiguous/writeable to satisfy the binding contract.
- `get_mass_matrix` delegates to the helper; behaviour and JSON payload are
  unchanged on every supported MuJoCo version. Empty-DoF scenes still return a
  well-typed `(0, 0)` matrix.
- Regression tests assert the matrix is symmetric positive-definite, exercise
  the legacy-signature fallback path, and cover the zero-DoF case.

### Fixed: numpy scalars no longer dropped from recorded state/action vectors

`DatasetRecorder.add_frame` flattens each observation/action value into the
LeRobot `observation.state` / `action` vectors. The value-type dispatch handled
Python `int`/`float`, 0-dim `np.ndarray`, and list/array sequences, but NOT
numpy scalar types (`np.float32`, `np.int32`, ...). These are the element type
you get from indexing a MuJoCo `qpos`/`ctrl` array (`np.asarray(qpos)[i]`) and
from many policy action paths, so they are extremely common in real recording
loops. They are neither Python `float` nor `np.ndarray`, so they fell through
every branch and were silently omitted from the vector.

The result was a corrupted dataset with no error at write time: a column
dropped to the wrong length (or vanished entirely). On reopen, LeRobot raised a
confusing downstream `Feature mismatch` error far from the cause.

- `add_frame` now treats `np.generic` scalars (alongside 0-dim `np.ndarray`)
  as scalar values for both state and action flattening.
- Regression test feeds `np.float32`/`np.float64`/`np.int32` values and asserts
  the flattened vectors are present, correctly ordered, and full-length.


### Added: `[molmoact2]` optional-dependency extra

MolmoAct2 transformers-native VLA checkpoints (e.g. `allenai/MolmoAct2-SO100_101`)
run through `MolmoAct2Policy`, which shipped in lerobot AFTER the 0.5.1 PyPI
release (merged in lerobot PR #3604). A plain `pip install strands-robots[lerobot]`
resolves lerobot 0.5.1, which lacks it.

- New `[molmoact2]` extra layers MolmoAct2's auxiliary deps (`transformers`,
  `peft`, `scipy`) on top of `[lerobot]`, mirroring lerobot's own `[molmoact2]`
  extra. PyPI rejects direct git URLs in a published dependency table, so the
  lerobot-from-source pin stays in the documented install command (same pattern
  as `[cosmos3-diffusers]`):
  `uv pip install strands-robots[molmoact2] "lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git"`.
  Added to `[all]`.
- The fail-loud `ImportError` raised by the MolmoAct2 load path now names the
  `strands-robots[molmoact2]` extra and lerobot PR #3604, so a missing lerobot
  surfaces an actionable install hint instead of a bare import failure.
- New `molmoact2` pytest marker; README + `docs/policies/lerobot-local.md` +
  `docs/getting-started/installation.md` document the extra. Once a tagged
  lerobot >= 0.5.2 is on PyPI the extra can pin `lerobot[feetech]>=0.5.2`
  directly and the git-source step drops away.


### Fixed: `start_policy` validates the step horizon synchronously

`start_policy` runs the rollout on a background thread. A malformed step
horizon (`n_steps <= 0` or `control_frequency <= 0`) was only caught inside
`run_policy` once the future executed, so the caller received a false
`status="success"` ("Policy started") while the rollout immediately errored in
the background - and the robot was left registered as having a running policy,
wrongly gating the next `start_policy` on that robot as "already running".

- The horizon validation (`n_steps`/legacy `max_steps` -> `duration`, with the
  non-positive guards) is now a shared `SimEngine._resolve_horizon` helper used
  by both `run_policy` and the MuJoCo `start_policy` override. `start_policy`
  calls it synchronously before submitting to the executor, returning the same
  structured `status="error"` dict and leaving no future registered, so a
  subsequent well-formed `start_policy` on the same robot succeeds.

### Added: `WBCPolicy` provider (`wbc`, shorthand `sonic`) - GR00T Whole-Body-Control (SONIC)

A new non-VLA policy provider wrapping NVIDIA's GR00T Whole-Body-Control
(SONIC / decoupled-WBC) ONNX controllers for deploy-grade Unitree G1
locomotion (closes #466). Clean-room against the upstream reference
(`NVlabs/GR00T-WholeBodyControl` `decoupled_wbc/sim2mujoco`).

- In-process ONNX (no torch, no sidecar) via the new `[wbc]` extra
  (`onnxruntime` + `pyyaml` + `huggingface_hub`). No model weights bundled; the
  real `GR00T-WholeBodyControl-{Balance,Walk}.onnx` are fetched/pointed-at at
  runtime under the NVIDIA Open Model License.
- `requires_images = False`; reads the locomotion goal from the well-known
  kwargs (`target_velocity = [vx, vy, omega]`, optional `target_orientation`
  for base RPY, optional `height`). Drives the 15 leg+waist DOFs; the 14 arm
  joints are held at defaults (composing an upper-body policy on top is the job
  of a future `CompositePolicy`, #468).
- Faithful to the reference contract: 86-dim observation frame
  (`command[7] + base_ang_vel[3] + projected_gravity[3] + qj[29] + dqj[29] +
  prev_action[15]`) stacked over `obs_history_len=6` (network input 516);
  whole-body qj/dqj (all 29 joints, not just the 15 controlled); two ONNX
  sessions (main `policy` + `walk_policy`) selected by `norm(raw velocity) <=
  0.05`; PD-to-torque law exposed via `WBCPolicy.compute_torques(...)`;
  zero-warm-started history deque; quaternion math numerically identical to
  upstream.
- `WBCConfig` loads the upstream `g1_gear_wbc.yaml` directly (JSON or YAML;
  flat `*_scale` keys normalised into `obs_scales`). Joints are resolved by
  name (handles the G1's leading `floating_base_joint` + interleaved arms).
- New per-call `policy_kwargs` channel on `SimEngine.run_policy` / `start_policy`
  (and the MuJoCo overrides) forwarded verbatim to `policy.get_actions`, so the
  #300 well-known goal kwargs reach non-VLA providers (WBC, cuRobo, MoveIt2)
  through the local sim path, not just the mesh.
- `wbc` / `sonic` added to the mesh + Device Connect policy-provider allowlist
  so WBC can be driven over `tell()` / Device Connect.
- Registered as `wbc` (shorthand `sonic`) in `registry/policies.json`; docs at
  `docs/policies/wbc.md`; torque-control deploy harness +
  `simulate_rollout` at `examples/wbc_g1_torque_deploy.py` (the real weights
  produce a stable forward G1 walk).

## [0.4.0] - 2026-06-16

First tagged release since v0.3.8. Collapses the full v0.3.9 (mesh security hardening) and v0.4.0 (docs, policies, motion planners, hardware compatibility, plus the coverage and API-ergonomics hardening pass) work that had accumulated under topic-scoped `Unreleased` sections. No code changed in this release-prep step; the version is derived from the git tag via `hatch-vcs`.

### Cosmos 3 in-process diffusers backend

#### Added: `Cosmos3Policy(backend="diffusers")`

Cosmos3Policy gains a second backend that runs Cosmos 3 **in-process** via the
optional native Hugging Face `diffusers` stack (the `Cosmos3OmniPipeline`),
alongside the existing WebSocket `service` backend (the default, unchanged).

- `backend="service"` (default) - WebSocket to the Cosmos Framework RoboLab
  policy server. Zero behavioural change; all existing service output is
  byte-identical.
- `backend="diffusers"` - in-process load via native Hugging Face `diffusers`
  (the upstream `Cosmos3OmniPipeline` driven by a `CosmosActionCondition`). One
  forward pass returns the predicted world video + sound + the robot action
  chunk. The action chunk is returned through the unchanged Policy ABC contract
  (`get_actions -> list[dict]`, reusing the shared `_unpack_actions`); the world
  video/sound are surfaced on the new `Cosmos3Policy.last_rollout` attribute
  (a non-breaking auxiliary channel - the ABC return type is not changed). The
  diffusers backend emits the model's raw unified action (DROID = 9D
  end-effector pose + 1D gripper), named by the embodiment `raw_action_layout`,
  rather than the service server's post-processed `joint_pos` (8D) layout.
- Three Cosmos physics `mode`s thread through the diffusers backend: `policy`
  (default), `forward_dynamics`, `inverse_dynamics`. These do not exist in
  service mode - a non-`policy` mode under `backend="service"` raises a clear
  unsupported error (no silent no-op).
- Native `diffusers` (+ torch + transformers) is an optional dependency,
  imported lazily inside the diffusers backend. When missing it raises an
  actionable install error (the `cosmos3-diffusers` extra + the
  diffusers-from-source pin that ships `Cosmos3OmniPipeline`). The extra composes
  with `numpy>=2`, so it is co-installable with `cosmos3-service` and `lerobot`.
- New `[cosmos3-diffusers]` extra in `pyproject.toml`; NOTICE attributes
  Hugging Face diffusers (Apache-2.0).
- GPU load path hardening (surfaces only on a real `from_pretrained` + run, not
  the mocked unit tests): `Cosmos3OmniPipeline.__init__` builds a
  `CosmosSafetyChecker` that hard-raises `ImportError: cosmos_guardrail is not
  installed` unless the heavy optional `cosmos_guardrail` extra is present, so
  the backend now passes `enable_safety_checker=False` to `from_pretrained` by
  default (new `enable_safety_checker` arg opts back in when `cosmos_guardrail`
  is installed). Cosmos runs in `bfloat16`, so the output action tensor is
  `bfloat16` (or `float16`), which `np.asarray` cannot read
  (`TypeError: Got unsupported ScalarType BFloat16`); `_to_numpy` now up-casts
  half precision to `float32` before handing the chunk to NumPy.

#### Added: Cosmos 3 -> MuJoCo sim-loop bridge (de-normalize + inverse kinematics)

The diffusers backend returns the model's raw unified action **quantile-
normalized to `[-1, 1]`** and encoding a *relative end-effector pose delta* per
step - **not joint radians**. Feeding it straight into MuJoCo joint actuators is
physically meaningless (normalized columns land arbitrarily inside/outside real
joint limits; MuJoCo silently clamps and the arm does not track). A new sim-loop
bridge (`cosmos3-sim` extra: `mink` + `mujoco`) closes the loop in three honest
geometric steps, applied *after* Cosmos (the Cosmos "modes" are world-model
conditioning, not kinematics):

- **De-normalize** (`action_decode.denormalize_quantile`) - inverts the quantile
  transform with per-embodiment `q01`/`q99` stats bundled under
  `policies/cosmos3/stats/` (`0.5 * (a + 1) * (q99 - q01) + q01`, mirroring
  `cosmos_framework`'s `denormalize_action(method="quantile")`). New
  `Cosmos3Embodiment.normalization` field (`"quantile"`).
- **Decode poses** (`action_decode.decode_pose_trajectory`) - integrates the
  per-step `[translation(3), rot6d(6)]` deltas into an absolute `(T+1, 4, 4)`
  SE3 trajectory anchored at the robot's current EE pose.
- **Inverse kinematics** (`sim_ik.MinkIKBridge`) - solves each Cartesian target
  to joint angles via `mink` differential IK on the same `mujoco.MjModel`
  (`FrameTask` + `PostureTask`, warm-started). Defaults to the `daqp` QP solver
  that `mink` ships via `qpsolvers[daqp]`, so the `cosmos3-sim` extra needs no
  extra solver dependency. `decode_cosmos_chunk_to_targets` composes all three
  into `{qpos, gripper, poses, tracking_error}`.
- Verified on Thor against real `nvidia/Cosmos3-Nano` weights: a reachable EE
  trajectory tracks to **mean ~= 11.5 mm / max ~= 42.8 mm**, pinned by the
  `tests/policies/cosmos3/test_sim_ik.py` regression (off-GPU, synthetic-but-
  reachable) plus a GPU integration test exercising the path off real Cosmos
  output.
- New `[cosmos3-sim]` extra (`mink` + `mujoco`); `mink` added to the dev env.
  numpy>=2 compatible (co-installable with `cosmos3-diffusers` /
  `cosmos3-service` / `sim-mujoco` / `lerobot`). NOTICE attributes `mink` and
  MuJoCo (Apache-2.0) and the cosmos_framework-derived quantile stats.

### serial_tool ASCII output

#### Fixed: emojis in ``serial_tool`` result strings

The ``serial_tool`` agent tool emitted emojis in its result ``text`` fields
(port listings, read/send summaries, Feetech servo responses, monitor output),
violating the project's "no emojis in user-facing strings" rule -- agents read
these strings programmatically, so the glyphs are pure tokenizer noise. All
result strings are now plain ASCII (``->`` instead of the arrow glyph, ``deg``
instead of the degree sign). Also removed a dead unused inner helper
(``send_serial_data``). Behavior tests cover every action branch and pin the
ASCII-only contract.

### #385 (Mesh + IoT safety/control-surface hardening)

#### Added: mesh control-surface hardening

Defence-in-depth for the Zenoh mesh teleop and command paths:

- **Teleop lockout enforcement (C-1)** -- input frames are now dropped
  while the e-stop lockout is engaged; previously the input path bypassed
  the lockout.
- **Startup warning (H-1)** -- loud warning when `STRANDS_MESH_OVERRIDE_CODE`
  is unset (lockout becomes unrecoverable without it).
- **Teleop value + rate bound (H-2)** -- joint value clamp tightened from
  `1e6` to `4pi` (`STRANDS_MESH_INPUT_VALUE_ABS`); per-receiver apply-rate
  ceiling added (`STRANDS_MESH_INPUT_MAX_HZ`, default 100 Hz).
- **Command replay dedup (H-3)** -- `(sender, turn_id)` keyed dedup with
  TTL; read-only actions exempt.
- **Resume brute-force throttle (M-1)** -- count-keyed cooldown
  (`STRANDS_MESH_RESUME_MAX_FAILS` / `STRANDS_MESH_RESUME_BACKOFF_S`).
- **Peer registry bound (M-2)** -- `STRANDS_MESH_MAX_PEERS` (default 1024),
  evict-oldest on overflow.
- **Presence freshness validation (M-3)** -- stale/replayed heartbeats
  rejected.
- **Positive-path audit (M-5)** -- `command_executed` and sampled
  `input_stream_applied` events (`STRANDS_MESH_INPUT_AUDIT_EVERY`).

#### Added: IoT provisioning hardening

- **MQTT Last Will dead-man policy** -- `provision_robot(...,
  allow_estop_publish=False)` creates a policy that drops the estop
  Publish grant while retaining Subscribe + Receive.
- **E-stop fan-out idempotency** -- Lambda dedup per `(peer_id, t)` via
  DynamoDB conditional write, fails OPEN on store error.
  `STRANDS_ESTOP_DEDUP_TTL_S` (default 30 s) controls the window.

New env vars: `STRANDS_MESH_OVERRIDE_CODE`, `STRANDS_MESH_INPUT_VALUE_ABS`,
`STRANDS_MESH_INPUT_MAX_HZ`, `STRANDS_MESH_MAX_PEERS`,
`STRANDS_MESH_RESUME_MAX_FAILS`, `STRANDS_MESH_RESUME_BACKOFF_S`,
`STRANDS_MESH_INPUT_AUDIT_EVERY`, `STRANDS_ESTOP_DEDUP_TTL_S`.

### LeRobot 0.5.2 recording + policy pipeline hardening

#### Fixed: customer-mode E2E friction points (GH #373)

Eight first-run paper cuts found during a fresh-clone SO101 customer workflow:

- **`[lerobot]` extra now pulls `lerobot[feetech]`** so `scservo_sdk` installs
  for every Feetech-based (SO100/SO101/Koch) customer's first `mode="real"`
  run -- previously a `ModuleNotFoundError` blocker.
- **`Robot("so100")` (the `Simulation`) is now callable**:
  `robot(action="render", camera_name="topdown")` dispatches to the action
  method instead of raising `TypeError: object is not callable`, matching the
  README contract.
- **Pre-0.5 SO-family calibration files auto-migrate.** lerobot 0.5.1 unified
  `so100_follower/`/`so101_follower/` into `so_follower/`; `HardwareRobot`
  now copies a single legacy calibration JSON to the new path at init so
  existing calibrations Just Work (no more confusing `RuntimeError` on the
  first `get_observation()`).
- **README param-name aliases accepted.** The action dispatcher now treats
  `camera_names=` -> `cameras=` and `joint_positions=` -> `positions=` as
  aliases so copy-pasted older docs don't raise "unexpected keyword argument".
- **`STRANDS_MESH_LOCAL_DEV=1`** is a one-variable localhost mesh preset:
  defaults auth to `none` AND satisfies the insecure-acknowledgement second
  factor by itself (no separate `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1`).
  An explicit `STRANDS_MESH_AUTH_MODE=mtls` still wins.
- **`mesh.peers_by_id` dict + `mesh.get_peer(peer_id)` helper** added
  alongside the existing `mesh.peers` list, so dict-style peer lookup
  (`mesh.peers_by_id[peer_id]`) no longer raises `TypeError`.
- **README sweep**: clarified `Robot()` auto-creates the world (don't call
  `create_world()` again), fixed the callable usage example, and documented
  the new mesh env vars in the Configuration table.

#### Fixed: realistic sim rendering + wrist cameras (GH #373 follow-up)

- **Dimmed the MuJoCo headlight.** The default camera-tracking headlight
  (diffuse 0.4, specular 0.5, always on) stacked additively on the two
  explicit scene lights, washing out renders and flattening shadow contrast --
  and looking nothing like real camera footage. `SpecBuilder.build` now sets
  the headlight to a low, shadow-free term (diffuse 0.2, specular 0) so the
  explicit directional lights do the work. More realistic sim data.
- **Body-mounted (wrist/gripper) cameras.** `add_camera` gained a
  `parent_body` parameter: pass a body name (e.g. `"so101/gripper"`) and the
  camera mounts ON that body and tracks it as the arm moves -- matching the
  physical wrist camera on a real SO101/SO100. `position`/`target` are then
  interpreted in the body's local frame. Omitting `parent_body` keeps the
  prior world-fixed behaviour. An unknown `parent_body` returns a structured
  error listing the available (namespaced) body names.


#### Changed (breaking): ``panda`` embodiment split into joint-space vs EEF

The ``panda`` embodiment previously aliased to ``panda_libero``, conflating a
joint-space configuration with an end-effector/task-space one. These are now
two distinct entries:

- ``data_config='panda'`` -> **joint-space** (7 arm joints + gripper).
- ``data_config='panda_libero'`` -> **EEF/task-space** (LIBERO convention).

**Migration:** any caller passing ``data_config='panda'`` that actually
expected the EEF/task-space schema (the old aliased behaviour) must switch to
``data_config='panda_libero'``. Left unchanged, such a policy now receives
joint-space observations/actions and will silently misbehave. Callers wanting
plain joint-space need no change.

#### Added: synchronized multi-robot recording (``run_multi_policy``)

Drives N robots in one synchronized control loop and records all robots into a
single merged frame per timestep (prefixed ``<robot>__<key>`` state/action +
all cameras), stepping physics exactly once per loop iteration. Replaces the
earlier two-thread approach that interleaved single-robot frames into a corrupt
dataset. ``action_horizon`` accepts an ``int`` (all robots) or a
``{robot: horizon}`` mapping; a policy is re-queried only when its per-robot
action queue drains (open-loop chunk execution), so expensive VLA inference
amortizes over the horizon instead of running every step.

Note: LeRobot stores one task string per frame. Supplying distinct per-robot
instructions logs a ``WARNING`` and records only the first robot's task;
per-robot task columns are not yet supported.

#### Added: multi-episode recording append (``DatasetRecorder.resume``)

``start_recording(overwrite=False)`` on an existing dataset previously crashed
with ``FileExistsError`` (it always called ``LeRobotDataset.create()``). It now
routes to a new append-capable ``DatasetRecorder.resume()`` so multiple
episodes accumulate into one dataset. This replaces a hard crash, so no caller
could have depended on the prior behaviour.

#### Fixed: camera recorder returned success before the first frame

``start_cameras_recording`` now blocks until the recorder thread's
(thread-bound) EGL context is warm and the capture loop has begun, so a
caller that stops shortly after start no longer races the warmup and gets an
empty buffer / no MP4.

#### Fixed: embodiment + registry correctness

- Embodiment coverage 4 -> 33 configs grounded in lerobot drivers + MuJoCo XMLs.
- ``aloha`` had empty state/action keys (silent no-op) -> 16 bimanual joints.
- ``so100``/``so101`` decoupled (distinct sim joint names).
- Registry: ``tiago_dual`` (``++`` module-name regex) and ``unitree_a1``
  (``xml/`` asset subdir) now load; all 57 menagerie-asset robots resolve.
- Policy-config registration walks every ``lerobot.policies`` subpackage
  (incl. PEP-420 namespace packages), so newly shipped policies (e.g.
  ``molmoact2``) register without a hand-maintained import list.

### #320 (MuJoCo robot-scene ground-plane z-fighting)

#### Fixed: broken floor render when a robot asset ships its own ground plane

Robots whose asset MJCF includes its own ground/floor plane (e.g.
``franka_emika_panda/scene.xml`` ships ``<geom name="floor" type="plane"/>``)
produced a **severely broken floor** - a flickering checkerboard/triangle mess
- when added to a world created with ``ground_plane=True`` (the default). Two
coplanar infinite ground planes at z=0 with different checker materials
(``grid_mat`` vs the robot's ``groundplane``) caused depth-buffer Z-fighting.
The artifact corrupted rendered videos, camera observations fed to policies,
and demos, with no error raised.

``SpecBuilder.attach_robot`` now strips plane geoms from the robot scene MJCF
before attaching it, so exactly one world-owned ``ground`` plane survives. The
world ``ground`` plane (configurable via ``create_world(ground_plane=...)``)
is the single source of truth; robots contribute only their own
bodies/joints/actuators/sensors.

### #273 (estop lockout concurrency pin)

#### Added (tests): concurrent-estop lockout race regression pins

Pinned the issue #273 invariant that the e-stop lockout check-then-set
(`_estop_lockout.set()` + `_last_estop_ts` / `_last_estop_mono` writes)
stays inside `Mesh._estop_replay_lock`. Two concurrent e-stops from
distinct issuers now provably yield exactly one `remote_estop_engaged`
plus one `remote_estop_redundant` audit event (never two engages).
`tests/mesh/test_estop_lockout_race.py` adds a deterministic forced-
interleave race test plus source-text pins guarding lock containment
and timestamp-pair atomicity against future refactors. Code already
fixed on main; this locks it.

### #228 (AWS IoT provisioning hardening)

#### Changed: default presigned-URL TTL for camera offload

``CameraOffloader.presign_ttl`` default is now **60 seconds** (was 3600s).
A 1-hour ceiling (``MAX_PRESIGN_TTL_SECONDS``) is enforced; values above
the cap are clamped with a ``WARNING``. The change shrinks the replay
window for a captured ``strands/<thing>/camera/<cam>/ref`` MQTT message
from one hour to one minute.

Migration: deployments whose downstream consumers (review UIs,
recording pipelines that fetch on a delay) need >60 seconds of validity
should opt in explicitly:

```bash
export STRANDS_MESH_CAMERA_PRESIGN_TTL=3600   # legacy 1h
```

or pass ``presign_ttl=3600`` to ``CameraOffloader(...)`` / ``enable_for_mesh(...)``.

#### Added: AWS IoT provisioning hardening

Applies to ``strands_robots.mesh.iot.provision`` and
``strands_robots.mesh.iot.camera_offload``:

- **CA pinning** - ``AmazonRootCA1.pem`` is verified against an
  in-tree pin tuple (``_AMAZON_ROOT_CA1_PINS``) at download AND on
  every on-disk re-use. Defeats CA-substitution MITM. Operators can
  add additional pins via ``STRANDS_MESH_CA_PINS`` (comma-separated
  64-char lowercase hex). The break-glass ``STRANDS_MESH_DISABLE_CA_PIN=true``
  (case-insensitive) writes a ``.unverified`` sidecar marker (mode
  ``0o600``) for audit traceability.
- **Strict thing-name regex** (``^[a-zA-Z0-9_-]{1,128}$``,
  ``re.fullmatch``) applied symmetrically across ``provision_robot``,
  ``provision_operator``, and ``teardown_thing``. Rejects path
  separators, dots, spaces, NUL, non-ASCII, and trailing
  ``\n``/``\r``/``\t``. Pre-existing AWS IoT Things containing ``:``
  must be renamed (we deliberately reject ``:`` due to NTFS / classic
  Mac filesystem semantics).
- **IoT policy scope** - robot/operator policies use explicit
  per-thing topic prefixes; no ``Resource: '*'`` on Receive.
  ``OperatorPublishToFleet``'s ``*/cmd`` wildcard is documented and
  pinned as a deliberate design choice (``test_publish_to_fleet_wildcard_is_deliberate``).
- **Per-recv TLS timeout bound** via custom ``HTTPSHandler`` (defeats
  malicious-broker connection-stalling).
- **``teardown_thing(cert_dir=...)`` kwarg** for parity with
  ``provision_robot``/``provision_operator`` (closes stale-credential
  leak on non-default ``cert_dir`` deployments).

New env vars (documented in README Configuration matrix):
``STRANDS_MESH_CA_PINS``, ``STRANDS_MESH_DISABLE_CA_PIN``,
``STRANDS_MESH_CAMERA_PRESIGN_TTL``.

Known follow-ups: #249 (camera privacy kill-switch + S3 ACL),
#251 (chunked-read parity in ``_ensure_ca``), #259 (kwarg negative-TTL
WARNING symmetry), #260 (warn on re-use of break-glass-written CA).

### #178 (LiberoOffScreenRenderEngine retired)

#### Removed: ``LiberoOffScreenRenderEngine`` simulation backend (BREAKING)

After PR #184 made ``MuJoCoSimEngine`` byte-equivalent to upstream LIBERO
(model-level inertias, ``mj_step`` divergence 0 over 200+ substeps, mean
``success_rate=0.92`` vs offscreen ``0.72`` on libero-10/SCENE5),
``LiberoOffScreenRenderEngine`` has no functional reason to exist. It is
deleted entirely.

What is gone:
- **Deleted**: ``strands_robots/simulation/libero_offscreen_render/``
  (entire package, ~700 LoC).
- **Deleted**: ``"libero_offscreen_render"`` registry entry in
  ``strands_robots.simulation.factory`` and its aliases
  ``"libero_offscreen"`` and ``"libero_osr"``.
- **Deleted**: ``LiberoAdapter._on_episode_start_offscreen`` and the
  ``hasattr(sim, "setup_libero_task")`` dispatch branch in
  ``LiberoAdapter.on_episode_start``. The unified ``MuJoCoSimEngine``
  path is the only path now.
- **Deleted**: ``LiberoAdapter.is_success`` no longer delegates to
  ``env.check_success`` on ``OffScreenRenderEnv``-backed engines (no
  such engines exist anymore). It now always evaluates the BDDL
  predicate tree, hardened in #170 / #173 / #175 to match upstream's
  ``check_ontop`` / ``check_contact`` semantics.
- **Deleted**: ``STRANDS_LIBERO_PREDICATE_LOG`` and
  ``STRANDS_LIBERO_PREDICATE_LOG_MAX`` env vars (the BDDL ↔
  ``env.check_success`` disagreement diagnostic; no offscreen env
  to compare against). The ``_walk_predicate_tree`` helper is kept
  for any future BDDL-evaluator debugging.
- **Deleted**: ``tests/simulation/libero_offscreen_render/`` (3 unit
  test files).
- **Rewrote**: ``tests_integ/benchmarks/libero/test_upstream_state_parity.py``'s
  ``test_state_observation_byte_equivalent_at_canonical_init`` to
  compare ``MuJoCoSimEngine`` directly against upstream's raw
  ``OffScreenRenderEnv`` (skipping the intermediate engine wrapper).
  Same coverage, less indirection.

Migration: rename the backend in any ``create_simulation()`` call.

```python
# Before
sim = create_simulation("libero_offscreen_render", ...)
# (also "libero_offscreen", "libero_osr")

# After
sim = create_simulation("mujoco", ...)
```

The ``mujoco`` backend now reaches ``success_rate >= 0.92`` on
libero-10/SCENE5 (vs ``0.72`` for the offscreen engine), so this is
strictly an upgrade for benchmark eval consumers.

Out of scope: ``examples/libero_mujoco.py`` in
``strands-labs/robots-sim`` still has an ``--engine={mujoco,libero_offscreen_render}``
switch. A follow-up issue tracks updating it once this PR lands.

### PR #85 (MuJoCo backend remediation)

#### MJCF builder refactor: string-concat -> MjSpec AST (closes #121, #122-#126)

The ``MJCFBuilder`` string-concat path and the ``scene_ops`` XML-round-trip
machinery (~700 lines total) are replaced by direct manipulation of
``mujoco.MjSpec`` - the editable MJCF AST shipped with MuJoCo 3.2+.

What changed under the hood:
- **New module** ``strands_robots/simulation/mujoco/spec_builder.py``. The
  ``SpecBuilder`` class owns scene construction + mutation (``build``,
  ``add_object``, ``remove_body``, ``add_camera``, ``remove_camera``,
  ``attach_robot``, ``from_mjcf_string``, ``from_file``).
- **Deleted**: ``strands_robots/simulation/mujoco/mjcf_builder.py`` (273
  lines of f-string MJCF and the ``_camera_xyaxes_from_target`` helper).
- **Rewrote**: ``strands_robots/simulation/mujoco/scene_ops.py`` from
  ~980 lines of tmpdir + ``mj_saveLastXML`` + ``ElementTree`` round-trips
  down to ~295 lines that go through ``spec.recompile(model, data)``.
- **Bumped**: ``mujoco>=3.0.0`` -> ``>=3.2.0`` in ``pyproject.toml`` so
  ``MjSpec`` is always available. Current hatch env runs 3.8.0.

Agent-visible wins:
- **New action** ``patch_scene_mjcf(ops=[...])`` - apply a list of
  structured ops (add_body, add_geom, add_site, set_body_pos,
  set_body_quat, delete_body) to the live spec atomically. Whole batch
  is rolled back from an XML snapshot if any op fails; one
  ``spec.recompile()`` for the whole batch, so qpos/qvel for unchanged
  joints are preserved. Narrower surface than ``replace_scene_mjcf``
  but much cheaper for surgical edits (no full-scene XML round-trip).
- **New action** ``replace_scene_mjcf(xml=...)`` - atomically replace the
  whole scene with agent-authored MJCF. Validated by actually compiling
  it, so ``<tendon>``, ``<equality>``, ``<pair>``, custom solref/solimp,
  sites, hfield, etc. all work without needing new ``SimObject`` shape
  vocabulary. On malformed XML returns a clean error dict (no process
  abort).
- **``ellipsoid`` shape** now works in ``add_object`` - it's a free
  bonus MuJoCo geom type the string-concat builder rejected.
- **Camera orientation** uses ``quat`` (computed via
  ``mujoco.mju_mat2Quat``) instead of a hand-rolled ``xyaxes`` string.
  Compiled ``cam_mat0`` is numerically identical within ~4e-7.
- **``spec.recompile(model, data)``** preserves existing joint qpos/qvel
  for unchanged joints automatically - no manual "copy state by name"
  loop. Object freejoints added post-compile get initialised to the
  body's ``pos``/``quat``.
- **No more XML injection surface**: names go straight into MjSpec which
  validates them itself, so the old ``_sanitize_name`` regex gate +
  fuzz test are no longer needed.

Downstream API is unchanged: ``add_object``, ``add_robot``, ``remove_object``,
``remove_robot``, ``add_camera``, ``remove_camera``, ``load_scene`` all keep
their tool-action signatures. Tests that asserted on exact XML strings
were rewritten to assert on compiled ``MjModel`` properties (``cam_mat0``,
``mj_name2id``) so they are representation-agnostic.

Known constraint: ``remove_robot`` now rebuilds the scene from scratch
(drops joint qpos state) rather than going through ``spec.delete()`` on
attached bodies. This sidesteps a MuJoCo 3.8 double-free bug where
``spec.delete(attached_body)`` + interpreter shutdown crashes. Trade-off
is documented in ``scene_ops.eject_robot_from_scene``.

#### Breaking

These changes tighten the MuJoCo AgentTool contract. Legacy callers that
silently worked by accident will now receive a clear error instead:

- **Router input validation**: The ``_dispatch_action`` router rejects any
  top-level parameter that isn't declared on the target method. Passing
  ``step(num_steps=5)`` (wrong name) or ``set_gravity(device="mps")``
  (stray kwarg) now errors with *"Unknown parameter X for action Y.
  Valid: [...]"* instead of silently dropping the value. Methods whose
  Python signature includes ``**kwargs`` (e.g. ``add_object``) keep their
  pass-through semantics.
- **Missing required args**: produce *"Action X requires parameter Y."*
  instead of a raw Python ``TypeError``.
- **Vector dimension validation**: ``position``, ``target``, ``origin``,
  ``force``, ``torque``, ``gravity``, ``direction``, ``point``, ``orientation``
  (quaternion), and ``color`` (rgba) all validated for length + numeric
  dtype before reaching numpy/MuJoCo.
- **Camera orientation**: ``add_camera(target=[x,y,z])`` is now honoured
  by baking ``xyaxes`` into the MJCF ``<camera>``. Previously the target
  was silently dropped and every custom camera rendered a default view.
  Degenerate case (``target == position``) errors.
- **Render camera validation**: ``render(camera_name="missing")`` errors
  with *"Camera 'missing' not found."* instead of silently falling back
  to the free camera while claiming to render from the named one.
- **Raycast zero-direction guard**: ``raycast(direction=[0,0,0])`` now
  errors with *"direction vector is zero-length"*. Previously MuJoCo's
  C-level ``mj_ray`` would abort the Python process.
- **apply_force requires a non-zero vector**: passing neither ``force``
  nor ``torque`` (or both zero) errors. Previously the call silently
  succeeded with no effect.
- **step(n_steps<0)** rejected (previously it corrupted ``step_count``).
- **Negative mass / timestep / size** rejected per shape; previously
  ``set_body_properties(mass=-1)`` and ``set_timestep(-0.01)`` silently
  succeeded.
- **Plane objects auto-static**: ``add_object(shape="plane")`` now forces
  ``is_static=True`` (planes are infinite in MuJoCo). Explicit
  ``is_static=False`` on a plane is a hard error.
- **Duplicate camera name** rejected. Previously a second ``add_camera``
  with an existing name silently overwrote the registry entry while
  leaving the old camera in the XML - ghost behaviour. Use
  ``remove_camera`` + ``add_camera`` to replace.
- **stop_policy(robot_name='')** errors with *"stop_policy requires
  'robot_name'."* instead of silently matching the first robot.
- **eval_policy** requires an explicit ``robot_name``. Default
  ``n_episodes`` lowered from 10 to 1.
- **register_urdf** validates the path: file must exist, be a file, and
  be readable. Previously bad paths were cached and blew up later.

#### Recording backend split

- ``start_recording`` (LeRobotDataset: parquet + per-camera MP4) still
  requires the ``[lerobot]`` extra. Its error message when lerobot is
  missing now points callers at ``start_cameras_recording`` for plain
  MP4 (which runs under ``[sim-mujoco]`` alone via imageio-ffmpeg).
- No API change - the fix is informational.

#### Resource hygiene

- ``destroy()`` and ``cleanup()`` now close renderers on the main thread
  and empty the TLS cache. Previously each ``create_world/destroy``
  cycle leaked one ``mujoco.Renderer`` + its GL context (~33 MB per
  cycle measured). Worker-thread renderers still release themselves on
  thread teardown (we avoid cross-thread ``close()`` to prevent
  ``cgl.free()`` SIGSEGVs on macOS).
- ``get_mass_matrix`` and ``get_contacts`` run ``mj_forward`` first so
  values are valid immediately after a ``reset`` or ``add_robot``
  (previously returned stale / uninitialised memory).

#### Concurrency guards

Write-mutations are now refused while a policy is running on any robot
in the world. Previously these could race the policy worker thread and
produce undefined behaviour or SIGSEGV:

    reset, set_gravity, set_timestep, set_joint_positions,
    set_joint_velocities, apply_force, set_body_properties,
    set_geom_properties, load_state, randomize, move_object

The error now lists *which* robot(s) are active so the LLM can
``stop_policy`` on each without guessing: *"Cannot 'X' while a policy
is running on 'armA', 'armB'. Stop it first: action='stop_policy'."*

#### Concurrent per-robot policies (GH #114)

Multiple ``start_policy`` calls on *different* robots now run
concurrently. MuJoCo physics is still serialized via ``self._lock``
(``mj_step`` and ``ctrl[]`` writes are not thread-safe for concurrent
mutation), but each policy owns a disjoint slice of ``data.ctrl[]`` so
two VLA arms can operate in the same scene without semantic conflict.

- ``start_policy("armA")`` + ``start_policy("armB")`` both succeed.
  Second call no longer hits a global "policy already running" gate.
- ``start_policy`` on the *same* robot while its policy is active
  still errors (unchanged).
- ``remove_robot("X")`` now gracefully stops X's own policy before
  removing, instead of requiring a prior ``stop_policy("X")``. Still
  errors if a *different* robot has an active policy (XML round-trip
  invalidates cached IDs everywhere).
- New action ``list_policies_running`` returns the names of robots
  with live policies. Prunes completed Futures as a side-effect.
- Completed policy Futures are no longer retained forever in
  ``_policy_threads`` (GH #120 companion fix).

#### Policy-hook robustness (GH #117)

``PolicyRunner.run`` previously caught *all* ``on_frame`` exceptions at
WARN level and kept iterating. A recording hook with a typo'd observation
key would log 500 lines and produce an empty dataset. Now we count
*consecutive* failures and abort the episode after a threshold (default
5, tunable via new ``max_onframe_failures`` kwarg).

- A single transient failure still logs + continues; counter resets on
  the next successful call.
- ``N`` consecutive failures raise ``RuntimeError`` so ``run()`` returns
  ``status='error'`` with a clear message, preventing silent dataset
  corruption.

#### Cleanup graceful shutdown (GH #116)

``Simulation.cleanup()`` no longer races the policy worker. Previously
cleanup set ``self._world = None`` and called ``executor.shutdown(wait=False)``
nearly simultaneously - a policy still inside ``mj_step`` segfaulted on
freed arrays. Now cleanup:

1. Signals every live policy to stop (``policy_running = False``).
2. Awaits each outstanding Future with a bounded timeout (default 5s,
   overridable via new ``cleanup(policy_stop_timeout=...)`` kwarg).
3. Only AFTER workers unwind do we null ``self._world`` and tear down
   renderers / viewer / executor.

Wedged workers that don't stop in time get logged as a warning - cleanup
proceeds rather than hanging the host process on exit.

#### Error message consistency

- All "no world" paths return the same string:
  *"No world. Call create_world (or load_scene) first."*
- Unknown-name errors use a uniform ``<Kind> 'X' not found.`` shape
  (Robot / Object / Body / Geom / Joint / Sensor / Camera / Checkpoint).
- ``stop_recording``, ``stop_cameras_recording``, ``stop_policy``,
  ``close_viewer`` are now **idempotent**: calling them when nothing
  is running returns ``status="success"`` with a *"Was not ..."* message
  so callers can invoke them unconditionally.
- ``get_recording_status`` returns success in every lifecycle state
  (no world / not recording / recording).

#### Deprecations

- **add_robot name-as-registry fallback**: passing ``name="my_bot"``
  without ``urdf_path`` or ``data_config`` used to resolve ``my_bot`` in
  the model registry. This now fires a ``DeprecationWarning``. Use
  ``add_robot(name="...", data_config="<registry_key>")`` instead. Will
  be removed next major release.

#### New / extended actions

- ``forward_kinematics(body_name="X")`` filters to a single body.
- ``get_features(robot_name="X")`` filters to a single robot's joints
  and actuators.
- ``set_geom_properties(geom_name="X")`` accepts the bare object name
  as an alias for the injected ``"{name}_geom"``.
- ``render_all`` flags cameras whose frame has near-zero pixel variance
  (``"⚠️ camera 'X': image appears empty (variance < 1)"``).
- ``render_depth`` surfaces MuJoCo's one-time ``ARB_clip_control``
  warning in the response text on macOS, so the LLM knows when depth
  accuracy is reduced.
- ``render`` / ``render_depth``: width/height validated up front;
  oversized requests get a plain-English message naming the actual
  framebuffer cap (``<global offwidth=...>``) instead of MuJoCo's raw
  error.
- ``run_policy`` / ``start_policy``: accept optional ``n_steps``
  (primary) or legacy ``max_steps`` as an alternative to
  ``duration``+``control_frequency``. ``duration = n_steps /
  control_frequency`` when ``n_steps`` is set.
- **New ``list_policies_running``** action returns the names of robots
  with a live policy - pairs with the new concurrent-policy support
  (see *Concurrent per-robot policies* above).
- ``randomize(randomize_physics=True)`` now reports per-body mass scales
  and per-geom friction scales in the response (not just range
  endpoints).
- ``get_contacts`` resolves unnamed geoms to
  ``"<body_name>/geom_<id>"`` so contact pairs are always human-readable.
- ``get_sensor_data(sensor_name="X")`` on a model with no sensors now
  distinguishes *"Sensor 'X' not found. Model has no sensors."* from
  the generic "no sensors in model" success.

#### Tests

- New: ``tests/simulation/mujoco/test_agenttool_contract.py`` - ~50
  tests that lock in router validation, tool_spec ↔ method parity,
  unified error messages, idempotent stop family, ``mj_forward`` before
  reads, render-dim validation, feature filters, camera duplicate
  policy, plane auto-static, policy horizon unification, and more.
- New: ``tests/simulation/mujoco/test_renderer_hygiene.py`` - 4 tests
  asserting TLS cache is emptied on ``destroy``, renderer reuse works
  for identical ``(w,h)``, and ``create_world`` after ``destroy``
  rebuilds cleanly.
- New: ``tests/simulation/mujoco/test_recording_backends.py`` - 2 tests
  (one skipped when ``lerobot`` IS installed) pinning the
  MP4-without-lerobot backend.
- New: ``tests/simulation/mujoco/test_input_validation.py`` - 11 tests
  for step/raycast/apply_force validation.
- New: ``tests_integ/test_resource_hygiene.py`` - 3 integration tests
  (require ``psutil``): 50 create/destroy cycles grow RSS < 50 MB; 500
  renders at fixed dims grow RSS < 100 MB; TLS cache cleared on destroy.

Test count: **256 → 362** (+106 new regression tests), zero
regressions. ``hatch run lint`` (ruff + mypy) clean across 102 source
files.
