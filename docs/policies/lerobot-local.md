---
description: HuggingFace LeRobot direct inference - ACT, Pi0, SmolVLA, Diffusion Policy, MolmoAct2. RTC + processor bridge.
---

# LeRobot Local

```bash
uv pip install "strands-robots[lerobot]"
export STRANDS_TRUST_REMOTE_CODE=1        # required; raises UntrustedRemoteCodeError otherwise
```

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",   # HF model_id or local path
    device="cuda",
)
```

## Parameters

```python
LerobotLocalPolicy(
    pretrained_name_or_path="",          # HF model_id or local checkpoint dir (required)
    policy_type=None,                    # override auto-detected class
    device=None,                         # "cuda" | "cpu" | "mps"
    actions_per_step=1,                   # auto-set from config.n_action_steps if left at 1
    use_processor=True,                  # observation/action processor bridge
    processor_overrides=None,
    tokenizer_max_length=48,
    tokenizer_padding_side="right",
    rtc_enabled=None,                    # Real-Time Chunk smoothing (NOT rtc=)
    rtc_execution_horizon=None,
    rtc_max_guidance_weight=None,
    inference_kwargs=None,
    embodiment=None,
    norm_tag=None,                       # MolmoAct2 normalisation tag
    image_keys=None,                     # MolmoAct2 camera key override
    inference_action_mode="continuous",  # "continuous" | "discrete"
    camera_key_map=None,                 # {robot_cam_name: policy_image_key}
    obs_rename_override=None,            # {runtime_obs_key: "observation.images.*"} merged over embodiment.obs_rename (value None/"" DROPS that key)
    strict_keys=False,                   # raise instead of positional camera fallback
    cache_model=True,                    # reuse a process-cached model across instances
    revision=None,                       # pin a HF Hub revision (branch/tag/commit SHA)
)
```

### Pinning a Hub revision

Pass `revision=` to pin a checkpoint to a reproducible Hub version - a
branch name, tag, or commit SHA. It is threaded to lerobot's
`PreTrainedPolicy.from_pretrained(..., revision=...)` (and to the config
resolution that auto-detects `policy_type`), so the exact weights are
loaded regardless of later pushes to the repo's default branch:

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/smolvla_base",
    revision="v1.0",   # or a 40-char commit SHA
)
```

Two revisions of the same repo are cached independently (the revision is
part of the model-cache key), so pinning never collides with an unpinned
load. Revision pinning is not supported for transformers-native MolmoAct2
checkpoints, which load weights via `checkpoint_path` rather than
`from_pretrained`; passing `revision=` with one raises `ValueError`. Pin
those by downloading the revision locally and pointing at the directory.

## Model caching

Loading a checkpoint reads its weights from disk and uploads them to the
device. For a large VLA (MolmoAct2 SO-100/101 ships 1295 weight files) that is
on the order of a minute or more per load. Eval/rollout drivers that build a
fresh policy per call - e.g. ``create_policy("lerobot_local", ...)`` inside a
per-episode loop - would otherwise pay that full load cost every time.

By default (`cache_model=True`) the loaded underlying model is cached at
process level, keyed by `(pretrained_name_or_path, policy_type, device)` (plus
the MolmoAct2 normalisation/processor knobs). Re-instantiating the policy for
the same checkpoint reuses the resident model and skips the weight load:

```python
from strands_robots.policies import create_policy
from strands_robots.policies.lerobot_local import clear_model_cache

# First build loads the weights; subsequent builds for the same checkpoint
# reuse the resident model (no reload).
p1 = create_policy("lerobot_local", pretrained_name_or_path="allenai/MolmoAct2-SO100_101", device="cuda")
p2 = create_policy("lerobot_local", pretrained_name_or_path="allenai/MolmoAct2-SO100_101", device="cuda")

clear_model_cache()  # evict cached models and free their GPU/CPU memory
```

The cached object is the same live module shared by every instance with the
same key. LeRobot policies carry per-episode state (action queue, temporal
ensemble buffers) that `Policy.reset()` clears between episodes, so sequential
reuse - including `Simulation.eval_policy` and per-rollout drivers - is safe.
Pass `cache_model=False` to force a private load when driving two rollouts of
the same checkpoint+device concurrently, and call `clear_model_cache()` to
release the held memory.

For multi-episode evaluation, prefer a single `Simulation.eval_policy(..., 
n_episodes=N)` (or `run_policy(..., policy_object=loaded)`) call, which loads
the policy once and reuses it across episodes regardless of this cache.

### Load telemetry

Every `LerobotLocalPolicy` records two attributes after construction so the
saving from the cache is observable instead of guessed:

- `load_cache_hit` (`bool`): `True` when the heavy `from_pretrained` weight
  read was skipped because the process cache already held this checkpoint.
- `load_time_s` (`float`): wall time the load took (near `0.0` on a cache hit).

`Simulation.run_policy` and `Simulation.eval_policy` surface these in their
`{"json": {...}}` result block as `policy_load_cache_hit` and
`policy_load_time_s`. In a multi-episode loop, a `policy_load_cache_hit=False`
on episode 2+ is a smell that the caller rebuilt the policy per episode instead
of reusing one warm `policy_object=`; an agent can read that field and
self-correct. Policies that expose no load telemetry (e.g. `MockPolicy`) report
the honest defaults `0.0` / `False`.

```python
from strands_robots.policies.lerobot_local import list_cached_models

# Inspect what is resident without touching private state.
for entry in list_cached_models():
    print(entry["namespace"], entry["pretrained_name_or_path"], entry["device"])
```

`list_cached_models()` returns one read-only dict per cached entry
(`namespace`, `pretrained_name_or_path`, `device`, `policy_class`); pair it with
`clear_model_cache()` to decide when to evict before loading a different
checkpoint.

## Supported models

`policy_type` accepts any type the installed lerobot can resolve - the strings
below mirror lerobot's own policy registry. It is auto-detected from a
checkpoint's config when omitted; pass `policy_type=` to override. The set
tracks the installed lerobot, so enumerate it at runtime rather than trusting a
static list (see [Discovering supported policy types](#discovering-supported-policy-types)).

| `policy_type` | Model |
|---------------|-------|
| `act` | Action Chunking Transformer |
| `diffusion` | Diffusion Policy (visuomotor) |
| `vqbet` | VQ-BeT - discrete action tokenisation |
| `tdmpc` | TD-MPC model-based control |
| `smolvla` | SmolVLA - HuggingFace small VLA |
| `pi0` / `pi05` / `pi0_fast` | Physical Intelligence VLA family |
| `groot` | NVIDIA GR00T |
| `molmoact2` | transformers-native SO100/SO101 VLA; `pip install 'strands-robots[molmoact2]'` (see below) |
| `eo1` | EO-1 VLA |
| `xvla` | X-VLA |
| `wall_x` | Wall-X VLA |
| `vla_jepa` | VLA-JEPA |
| `multi_task_dit` | Multi-task Diffusion Transformer |
| `gaussian_actor` | Gaussian actor |

### Discovering supported policy types

Enumerate the resolvable `policy_type` strings programmatically instead of
guessing. `list_policy_types` is the discovery peer of `list_providers` (the
follow-up to "which provider?" is "which `policy_type` does it take?"), so it
is re-exported at the package root and on `strands_robots.policies` alongside
`list_providers` -- no reach into the `lerobot_local` submodule required:

```python
from strands_robots import list_policy_types  # or: from strands_robots.policies import list_policy_types

list_policy_types()
# ['act', 'diffusion', 'eo1', 'gaussian_actor', 'groot', 'molmoact2',
#  'multi_task_dit', 'pi0', 'pi05', 'pi0_fast', 'smolvla', 'tdmpc',
#  'vla_jepa', 'vqbet', 'wall_x', 'xvla']
```

The submodule path `from strands_robots.policies.lerobot_local import
list_policy_types` keeps working; the top-level re-export is lazy, so reaching
it does not make a bare `import strands_robots` pull in torch.

The list reflects the *installed* lerobot (sourced from its policy registry),
so a newer lerobot reports more entries and a slimmer one fewer; it returns
`[]` when lerobot is not installed. Passing an unknown `policy_type` to
inference now raises an error that names these valid choices, so a typo is a
one-line fix instead of a dead end.

## MolmoAct2

MolmoAct2 ships in lerobot **>= 0.6** - its `MolmoAct2Policy` was merged in
lerobot PR #3604 and first released in 0.6.0 - so it resolves straight from
PyPI with no git-from-source install. The `[molmoact2]` extra layers the
auxiliary deps MolmoAct2's modeling and processor code needs
(`transformers>=5.4.0,<5.6.0`, `peft`, `scipy`) on top of
`strands-robots[lerobot]` (which pins `lerobot>=0.6.0,<0.7.0`):

```bash
uv pip install "strands-robots[molmoact2]"
```

MolmoAct2 then works:

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="your-org/molmoact2-so101",
    device="cuda",
    norm_tag="so101",
    image_keys=["wrist_camera", "front_camera"],
    inference_action_mode="continuous",
    # actions_per_step is auto-set from config.n_action_steps (30 for the
    # SO-100/101 checkpoints) when left at the default 1 - so the full
    # 30-step chunk the model was trained to replay open-loop is consumed
    # before re-querying vision. Pass an explicit value to override.
)
# see examples/molmoact2_so101_pickplace.py
```

MolmoAct2 SO-100/101 was trained for **30-step open-loop chunk replay**
(`n_action_steps = 30`). Run it through the sim with an `action_horizon` that
does not truncate the chunk - the runner clamps the effective horizon up to the
policy's `actions_per_step`, so passing `action_horizon=8` (or the default) is
safe, but you can also pin it explicitly:

```python
sim.run_policy(
    robot_name="so101_follower",
    policy_provider="lerobot_local",
    policy_config={
        "pretrained_name_or_path": "your-org/molmoact2-so101",
        "norm_tag": "so101",
        "inference_action_mode": "continuous",
        "actions_per_step": 30,   # explicit; matches the trained chunk size
    },
    instruction="pick up the cube",
    action_horizon=30,            # do not truncate the 30-step chunk
)
```

This requirement will go away once HuggingFace publishes lerobot >= 0.5.2 to PyPI
(which will include MolmoAct2 natively). At that point the `[molmoact2]` extra can
pin `lerobot[feetech]>=0.5.2` directly and the git-source step drops away --
`pip install strands-robots[molmoact2]` alone will suffice.

## Processor bridge and normalization

`use_processor=True` (default) wraps the policy in a processor bridge that
normalizes observations going into the model and unnormalizes actions coming
back out, so the robot sees commands in physical joint units.

The bridge loads the model's own pipeline configs in priority order:

1. `policy_preprocessor.json` / `policy_postprocessor.json` - LeRobot's standard
   saved pipelines (most lerobot-native checkpoints).
2. **`norm_stats.json` fallback** - checkpoints that ship only a stats file (no
   standard pipeline configs), such as the MolmoAct2 SO-100/101 family. The
   bridge detects the `molmoact2_norm_stats.v1` schema and builds the
   normalizers itself.

Without the fallback in (2), a stats-only checkpoint would silently pass data
through un-normalized: state reaches the policy in raw degrees and predicted
actions reach the motors still in the model's normalized space, producing
off-policy / micro-motion trajectories.

The `norm_stats.json` fallback builds a minimal pipeline (the normalizer
step only) - it has no `AddBatchDimension` or device step, unlike a standard
`policy_preprocessor.json`. The runtime batches and device-moves the
preprocessed observation itself, so a stats-only checkpoint runs on the
declarative `embodiment=` path with the same batched-tensor contract as a
standard pipeline; the model never sees an unbatched `observation.state`
alongside a batched image.

The fallback supports the `q01_q99`, `q10_q90`, `min_max` and `mean_std`
normalization modes declared by `norm_mode`. For `q01_q99`:

```
state_norm  = clip(2 * (state - q01) / (q99 - q01) - 1, -1, 1)
action_unnorm = (clip(action, -1, 1) + 1) * (q99 - q01) / 2 + q01
```

When a stats file declares multiple embodiment tags, pass `norm_tag=` to select
one; a single-tag file is auto-detected.

### Device-pinned checkpoints

A checkpoint trained on GPU bakes `device_processor.device = "cuda"` into its
`policy_preprocessor.json` / `policy_postprocessor.json`. Loaded on a host
without that device (CPU-only edge box, or a CUDA build on a machine whose
driver predates the wheel's CUDA version), LeRobot asserts the device is
available and the `device_processor` step fails to instantiate -- which surfaces
as an error indistinguishable from "no pipeline config present".

The bridge already moves every tensor onto the `device` you pass to
`create_policy` (auto-detected when `None`), so it reconciles the pinned step
onto that resolved device and retries the load once, rather than dropping the
pipeline. Without this, normalization would be silently disabled: state reaches
the policy in raw units and actions reach the motors in normalized space,
producing off-policy / micro-motion trajectories. An explicit
`processor_overrides={"device_processor": {"device": ...}}` is still honored
as-is and takes precedence over the automatic reconciliation.

## Camera routing

Robot/sim observations use bare camera names (`top`, `wrist`, `side`); the policy
declares image inputs under its own keys (`observation.images.top`, ...). The
policy routes each camera to a declared image slot by, in order:

1. an explicit `camera_key_map` (`{robot_cam: policy_image_key}`) when provided;
2. exact name match (`top` -> `observation.images.top`);
3. positional fallback into remaining slots, with a WARNING so a mismatched
   wiring is loud rather than silent. Pass `strict_keys=True` to raise a
   `ValueError` (listing the unmatched cameras vs available image keys)
   instead of falling back positionally; it defaults to `False` and is a
   no-op when `camera_key_map` or exact names already resolve every camera.

The declared order follows the model config's `image_keys` list when present
(e.g. MolmoAct2), otherwise the order of the model's image input features. If
the robot supplies fewer cameras than the policy requires, a `ValueError` is
raised instead of feeding the model a missing or wrong view.

Image input slots are identified by their declared `FeatureType.VISUAL`, not
by a substring match on the feature name. A policy may declare image keys that
do not follow the `observation.images.*` convention (for example MolmoAct2's
bare `base`/`wrist` keys); such cameras are still routed to their VISUAL slots
rather than dropped, so the preprocessor never fails with a misleading
"image_keys missing from observation".

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="your-org/molmoact2-so101",
    camera_key_map={"front": "observation.images.top", "hand": "observation.images.wrist"},
)
```

### Embodiment `obs_rename` and the pre-flight check

When you pass an `embodiment` (e.g. `embodiment="so101"`), camera routing is
configured declaratively from the embodiment's `obs_rename` map
(`{runtime_camera_name: "observation.images.*"}`) instead of the heuristic
above. The runtime observation MUST therefore contain the camera names the
embodiment declares as rename sources. For `so101` those are `front` and
`wrist`:

```json
"obs_rename": {"front": "observation.images.image", "wrist": "observation.images.wrist_image"}
```

A camera-name mismatch (e.g. you added `realsense_top` / `realsense_side`
because a model card said "top + side") used to surface only deep in the
preprocessor AFTER the multi-minute weight download, as a confusing
`image_keys missing from observation` failure. `run_policy` / `eval_policy` now
run a cheap pre-flight check (`Policy.preflight`) BEFORE `create_policy`
downloads anything, and return a `status=error` naming the expected source
keys:

```
Embodiment 'so101' cannot route cameras to the model's image feature(s)
['observation.images.image', 'observation.images.wrist_image']: none of the
expected source key(s) ['front', 'wrist'] are in the runtime observation, which
provides [...]. Either: (a) rename your sim cameras to one of ['front', 'wrist']
..., or (b) pass policy_config={'obs_rename_override': {...}} ...
```

Two ways to fix it:

1. Rename your cameras to the expected source keys
   (`sim.add_camera(name="front", ...)`, `sim.add_camera(name="wrist", ...)`).
2. Keep your custom names and pass `obs_rename_override`, which merges OVER the
   embodiment's `obs_rename` so your names route onto the model's image
   features without renaming cameras:

   ```python
   sim.run_policy(
       robot_name="so101",
       policy_provider="lerobot_local",
       policy_config={
           "pretrained_name_or_path": "allenai/MolmoAct2-SO100_101",
           "embodiment": "so101",
           "obs_rename_override": {
               "realsense_top": "observation.images.image",
               "realsense_side": "observation.images.wrist_image",
           },
       },
   )
   ```

3. Drop a camera the embodiment declares but your checkpoint does not. The
   built-in SO embodiments declare both `front` and `wrist`; a single-camera
   checkpoint declares only one image feature, so the unmatched `wrist` rename
   targets a feature the model never declares and fails validation. Map the
   stale source key to `None` (or `""`) in `obs_rename_override` to remove it,
   and route your real camera onto the model's feature:

   ```python
   create_policy(
       "lerobot_local",
       pretrained_name_or_path="your-org/so101-single-cam-act",
       embodiment="so_real",
       obs_rename_override={
           "front": "observation.images.front",  # route the one camera you have
           "wrist": None,                         # drop the camera you do not
       },
   )
   ```

### Single camera with no embodiment

You do not need an embodiment at all for a single-camera checkpoint. Declare the
robot's joint names with `set_robot_state_keys([...])` and the policy synthesizes
a state-only embodiment that routes each declared `observation.images.*` feature
from its short name (`observation.images.front` <- `front`) and composes
`observation.state` in your joint order. A bare camera key (`front`) is
canonicalized to CHW float and renamed onto the model feature, and the state is
batched alongside it, so a single-camera ACT checkpoint runs on the declarative
path without manual key wiring:

```python
policy = create_policy("lerobot_local", pretrained_name_or_path="your-org/so101-single-cam-act")
policy.set_robot_state_keys(
    ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
     "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
)
# obs={"front": <HWC uint8 frame>, "shoulder_pan.pos": ..., ...}
actions = policy.get_actions_sync(obs, "pick up the cube")
```

Pass `camera_key_map={"my_cam": "observation.images.front"}` if your runtime
camera name differs from the feature's short name.

See [camera naming](camera-naming.md) for the model-card -> embodiment
translation table.

## RTC

```python
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/pi0_so100",
                        rtc_enabled=True, rtc_execution_horizon=16, rtc_max_guidance_weight=1.0)
```

Real-Time Chunking does two things. First, **seam blending**: each new action
chunk is denoised conditioned on the still-unexecuted tail of the previous
chunk (`rtc_execution_horizon`), so the trajectory has no discontinuity where
one chunk hands off to the next. Second, on real hardware, it lets **inference
overlap execution**: the controller fires the next inference while the current
chunk is still being executed, so the model's latency is hidden behind motion
rather than appearing as a stall at the seam.

### Re-query interval: `execution_horizon`

The SIM consumes `policy.execution_horizon` actions from each chunk before
re-querying - the single source of truth for the re-query rate. For an RTC
policy this is `rtc_execution_horizon` (default 10), **not** the trained chunk
length (`actions_per_step`, auto-detected from the model, e.g. 50). Re-querying
mid-chunk is what lets the policy receive its previous chunk's unexecuted tail
(`prev_chunk_left_over`) and blend the seam; draining the full trained chunk
first leaves that tail empty and silently degrades RTC to open-loop replay. The
policy decides this interval - a caller-supplied `action_horizon` is ignored for
RTC policies (it cannot stretch the interval and break blending). For non-RTC
policies `execution_horizon == actions_per_step` and the consumer still takes
`max(action_horizon, actions_per_step)` so the trained chunk is never truncated.

### Relative-action policies: prefix re-anchoring

Some flow-matching checkpoints (pi0 / pi0.5 / pi0-FAST trained with a
`RelativeActionsProcessorStep`) predict actions as offsets from the current
robot state rather than absolute joint targets. The unexecuted tail carried into
the next chunk (`prev_chunk_left_over`) is therefore only valid in the
coordinate frame of the observation that produced it. Because the robot state
moves between chunks, feeding that tail back verbatim would blend a STALE-frame
prefix into the next chunk and corrupt the seam.

For these policies the provider keeps the leftover in absolute coordinates and
re-expresses it against the live robot state every query via LeRobot's
`reanchor_relative_rtc_prefix` (reading the cached state from the preprocessor's
`RelativeActionsProcessorStep`), so the model always receives a correctly
anchored prefix. This is detected automatically from the loaded preprocessor
pipeline - no flag is needed - and only engages when an enabled relative-action
step is present. Absolute-action policies carry the model-space leftover
verbatim (their frame does not move). The deterministic step-count delay below
is untouched, so re-anchoring preserves bit-reproducibility.

### Synchronous vs async chunk execution in sim

`run_policy` / `PolicyRunner.run` accept an `async_rtc` flag controlling which
of those two RTC benefits the sim reproduces:

| `async_rtc` | Behaviour | Use when |
| --- | --- | --- |
| `None` (default) | Auto-resolve from `policy.is_chunk_emitting()`: chunk-emitting policies get the async overlap, single-step policies stay synchronous. An explicit `True`/`False` always wins. | The common case - let the policy decide. |
| `False` | Query the policy, drain `execution_horizon` actions (the full chunk for non-RTC; `rtc_execution_horizon` for RTC), then re-query. Seam blending works because the RTC policy is re-queried mid-chunk, but inference and execution do **not** overlap. | Single-step policies, deterministic regression runs, or any policy whose `get_actions` reads live sim state. |
| `True` | While the current chunk drains, fire the next `get_actions` on a single background worker once the chunk is ~50% consumed (using a fresh mid-execution observation), then atomically swap it in. A policy whose inference latency is at most the chunk's execution window pays (almost) zero visible stall at the seam. | Chunk-emitting VLA / flow-matching policies (pi0, pi0.5, pi0-FAST, SmolVLA, MolmoAct2) where you want sim per-step timing to track real-hardware behaviour, or to benchmark a streaming controller. |

Because MolmoAct2, pi0, pi0.5, pi0-FAST and SmolVLA all self-report as chunk-emitting, the default `async_rtc=None` enables latency masking for them automatically - no flag needed. Each `run_policy` result carries `rtc_*` telemetry (`rtc_async_enabled`, `rtc_prefetch_hits`, `rtc_prefetch_blocks`, `rtc_avg_inference_ms`, ...) so you can confirm the masking worked; see [Simulation -> Async-RTC chunk pipeline](../simulation/overview.md#async-rtc-chunk-pipeline-latency-masking). Pass `rtc_inference_timeout_s=` to abort cleanly on a stuck inference instead of stalling the whole rollout.

```python
# Default (async_rtc=None): pi0 self-reports as chunk-emitting, so the async
# overlap is enabled automatically - inference latency is masked.
sim.run_policy(robot_name="so101", policy_provider="lerobot_local",
               policy_config={"pretrained_name_or_path": "lerobot/pi0_so100", "rtc_enabled": True},
               action_horizon=8)

# Force the synchronous chunk-then-drain loop (inference shows up as a per-seam
# stall in sim) - e.g. for a deterministic regression run.
sim.run_policy(robot_name="so101", policy_provider="lerobot_local",
               policy_config={"pretrained_name_or_path": "lerobot/pi0_so100", "rtc_enabled": True},
               action_horizon=8, async_rtc=False)
```

`async_rtc` is provider-agnostic: it only schedules the inference/execution
overlap and never touches the policy's RTC machinery, so RTC-capable policies
still blend the seam internally. The runner invokes the policy from at most one
thread at a time and blocks on any in-flight inference before returning, so it
introduces no data race. Masking only helps when there is execution to hide
behind, so the benefit is largest at real-time pacing (`fast_mode=False`) with
multi-step chunks; with `fast_mode=True` and near-instant physics there is
little execution window to overlap.

### Deterministic inference delay

RTC has to know how many control steps the robot executed *while inference was
running* - that offset is where it slices the next chunk so the seam lines up.
Estimating it from wall-clock latency is non-reproducible: the measured latency
warms up over the first few inferences of an episode and jitters run-to-run, so
two fixed-seed episodes drift apart at the seam (the "seeds fixed but trajectory
varies" symptom in multi-episode evals).

`PolicyRunner` instead tells the policy the **exact** step count via
`policy.set_rtc_observed_delay(steps)` immediately before each query:

- Synchronous loop (`async_rtc=False`): the world is paused during inference, so
  exactly `0` steps elapse.
- Async pipeline (`async_rtc=True`): the prefetched chunk first applies after the
  remaining steps of the chunk currently executing drain - a known integer,
  independent of how long inference actually takes (a slow inference just stalls
  the loop; the arm does not advance past the chunk end while it waits).

So an eval driven by the runner is bit-reproducible across episodes regardless of
machine load. When a policy is driven directly without a runner (e.g. on async
real hardware where the arm genuinely keeps moving during inference), leave the
override unset (`None`) and the policy falls back to the wall-clock p95 estimate,
which is the right proxy there.

## See also

- [MolmoAct2 (SO-100/101)](molmoact2.md) - action contract, units, and motion diagnostics
- [Policy providers](../policies/overview.md)
- [Training](../training/overview.md)
- [GR00T](groot.md)
- [cuRobo](curobo.md)
- [LeRobot project](https://github.com/huggingface/lerobot)
