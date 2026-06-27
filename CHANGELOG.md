# CHANGELOG

All notable behavioural changes to `strands-robots` are logged here. Follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

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
