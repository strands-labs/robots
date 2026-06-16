# CHANGELOG

All notable behavioural changes to `strands-robots` are logged here. Follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

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
