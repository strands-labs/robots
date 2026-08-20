---
description: Post-tune any policy natively with the Trainer abstraction - one interface over LeRobot, Isaac-GR00T, and Cosmos3 pipelines.
---

# Training

`strands-robots` post-tunes policies **natively** through the `Trainer`
abstraction - the training-side peer of [`Policy`](../policies/overview.md)
(inference). One interface wraps three genuinely different upstream pipelines,
selected by the **same provider name** you use for inference:

```python
from strands_robots.training import create_trainer, TrainSpec

trainer = create_trainer("lerobot_local")   # same name as create_policy(...)
spec = TrainSpec(
    dataset_root="/tmp/my_dataset",          # what Robot.stop_recording() writes
    base_model="lerobot/act_aloha_sim",
    output_dir="/tmp/ft_out",
    steps=20000,
)
result = trainer.train(spec)                 # -> launches lerobot_train
# result.checkpoint_dir loads straight back into create_policy(...)
```

## Why an abstraction (not just `lerobot train`)

Not everything is LeRobot. Each backend ships its own post-training pipeline,
and a single `--policy.type` flag can't express them:

| Provider | Upstream entry point | Config surface | Launcher | HW floor |
|----------|---------------------|----------------|----------|----------|
| `lerobot_local` | `lerobot.scripts.lerobot_train` | draccus `--dotted.flags` | `python` / `accelerate launch` | CPU for a toy run; 1 consumer GPU in practice |
| `groot` | Isaac-GR00T `launch_finetune.py` | `FinetuneConfig` (tyro) + `tune_*` flags | `python` / `torchrun` | 1 modern GPU |
| `cosmos3` | `cosmos_framework.scripts.train` | TOML recipe + Hydra overrides; **DCP convert** + **safetensors export** | `torchrun` (HSDP) | 8×H100 80GB |
| `sagemaker` | none - the container image's own trainer | the same `TrainSpec`, as job hyperparameters | `CreateTrainingJob` (managed) | none locally; the job brings its own |

Those three local backends import a training library and drive it in-process.
`sagemaker` is the other shape: pure transport, importing no training library and
submitting the spec to a managed runner whose image packages one of the local
paths. The difference is visible in one place a caller has to handle - a local
`train()` blocks and returns a terminal result, while a submitted run can outlive
the call and come back as `running` with a `job_id` to poll via `status()`, and
no checkpoint yet. Branch on all three `TrainResult.status` values, not on
"not `error`".

The `Trainer` ABC hides all of that behind one lifecycle:

```
validate()  ->  prepare()  ->  train()  ->  export()
                   ▲                           ▲
            (cosmos: DCP convert,        (cosmos: DCP -> safetensors;
             groot: modality cfg)         lerobot/groot: passthrough)
```

plus `status()` for a "RUNNING ≠ learning" verdict on an in-flight job.

## The data loop, end to end

```python
from strands_robots import Robot, MockPolicy, create_policy
from strands_robots.training import create_trainer, TrainSpec

# 1. RECORD - one episode is enough to smoke-test the loop
sim = Robot("so100", mesh=False)
sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])
sim.start_recording(repo_id="local/demo", root="/tmp/demo_ds",
                    # fps must equal the rollout's control_frequency (default 50.0)
                    fps=50, task="pick up the red cube", overwrite=True)
sim.run_policy(robot_name="so100", policy_object=MockPolicy(),
               instruction="pick up the red cube", n_steps=60)
sim.stop_recording()        # writes a LeRobotDataset v3 at /tmp/demo_ds

# 2. TRAIN - thin wrapper over lerobot_train; ACT from scratch on CPU
trainer = create_trainer("lerobot_local", device="cpu")
spec = TrainSpec(dataset_root="/tmp/demo_ds", base_model="",
                 output_dir="/tmp/demo_ft", steps=2, save_freq=2,
                 global_batch_size=2, extra={"policy_type": "act"})
result = trainer.train(spec)

# 3. EXPORT - loadable artifact (HF-native passthrough for lerobot/groot)
ckpt = trainer.export(spec, result.checkpoint_dir)

# 4. DEPLOY - load the freshly-trained checkpoint back as a Policy
policy = create_policy(ckpt, device="cpu")
sim.run_policy(robot_name="so100", policy_object=policy,
               instruction="pick up the red cube", n_steps=15)
```

Swap `create_trainer("lerobot_local")` → `"groot"` or `"cosmos3"` and **only the
provider string changes** - exactly how `Robot("so100", mode="real")` swaps
sim↔hardware.

## TrainSpec - one spec, many backends

`TrainSpec` carries provider-agnostic fields; each trainer reads what it
supports and **ignores the rest** (the same tolerance rule as
`Policy.get_actions(**kwargs)`). Backend-specific knobs go in `extra`:

| Field | Meaning | Notes |
|-------|---------|-------|
| `dataset_root` | LeRobotDataset v3 root | a data source; has `meta/info.json` (optional when `dataset_repo_id` is set) |
| `dataset_repo_id` | Hub dataset id `org/name` | alternative data source; train from the Hub (lerobot) |
| `streaming` | stream frames, no full materialize | lerobot `StreamingLeRobotDataset`; bounded disk (Hub) / RAM (local); mutually exclusive with `val_episodes` |
| `base_model` | HF id / local ckpt to tune from | required for GR00T & Cosmos |
| `steps` / `global_batch_size` | the run size: optimizer steps x batch | each must be a positive integer; `validate()` refuses `0`, a fractional or non-finite value, and a `bool` (`True` would read as a silent one-step run) before anything is loaded |
| `method` | `full` \| `lora` \| `expert_only` \| `frozen_backbone` | `lora`+`expert_only` are mutually exclusive |
| `tune` | `{llm,visual,projector,diffusion}` | GR00T only |
| `val_episodes` | hold out the LAST N episodes | deterministic split; must be a positive integer below the dataset's episode count, and that count must be readable from a local `meta/info.json` (see the Hub-source note below). `validate()` refuses `0` or a negative (they produced no split and no eval cadence at all - the run trained on everything and logged no validation loss), a `bool`, and a fractional value (`2.7` reserved 3 episodes, `0.5` reserved none while still evaluating); mutually exclusive with `streaming` |
| `num_gpus` / `num_nodes` | multi-GPU / multi-node | selects the launcher |
| `seed` | reproducibility seed | must be a non-negative integer; `validate()` refuses a negative (`torch.manual_seed` would take it modulo `2**64`, so `-1` silently becomes `2**64 - 1`), a fractional or non-finite value, and a `bool`. `None` uses the backend's own default |
| `extra["policy_type"]` | lerobot `--policy.type` | act/diffusion/smolvla/pi0/pi05/... |
| `extra["groot_root"]` | Isaac-GR00T checkout | GR00T |
| `extra["sft_toml"]` / `extra["cosmos_root"]` | recipe + checkout | Cosmos |
| `extra["relative_actions"]` | train pi0-family with delta actions | lerobot `--policy.use_relative_actions=true` (pi0/pi05/pi0_fast) |
| `extra["sample_weighting"]` | RA-BC per-sample loss weighting dict | lerobot `cfg.sample_weighting` (`--sample_weighting.*`) |
| `extra["reward_model"]` | train a reward model (`sarm` / `robometer` / `topreward` / `reward_classifier`) instead of a policy | lerobot `cfg.reward_model` (`--reward_model.*`); requires lerobot >= 0.5.2 |

## From an agent (natural language)

The `train_policy` tool exposes the abstraction to a Strands Agent:

```python
from strands import Agent
from strands_robots import Robot
from strands_robots.tools import train_policy

agent = Agent(tools=[Robot("so100", mesh=False), train_policy])
agent("Record 50 cube-pick episodes, then post-tune lerobot ACT on the dataset "
      "at /tmp/demo_ds into /tmp/demo_ft, and tell me if it's actually learning.")
```

`train_policy` actions: `train`, `validate`, `status`, `export`, `list`.

`train` reports the step that fits the run it got. A finished run names the
checkpoint to load; a run that has not finished - the managed-job backend
whose job outlives the submitting process, which returns `running` once its
local poll budget expires - names the `status` poll for its `job_id` instead,
and a run that wrote no discoverable checkpoint says so. The tool never offers
`create_policy(<checkpoint_dir>)` for a result whose `checkpoint_dir` is
`None`, because that renders as `create_policy('None')` and raises
`Unknown policy provider: 'None'`. The run's own status always travels verbatim
in the result's `{"json": ...}` block.

## Provider-specific knobs

### LeRobot (`lerobot_local`)

```python
TrainSpec(..., method="lora", lora_r=16, extra={"policy_type": "pi05"})
# -> lerobot_train --peft.method_type=LORA --peft.r=16 --policy.type=pi05
```

#### Passing lerobot's own flags through `extra`

Any key in `extra` that names a real field of lerobot's config tree is applied to
it, dotted for a sub-config (`policy.*`, `dataset.*`, `wandb.*`). A key that
matches no field is ignored with a warning, so a typo can never become an
arbitrary flag.

A value may be written either as the field's own Python type or as text. Text is
read with lerobot's own draccus decoder - the same one behind `--key=value` - so
one spec means one run whether the backend builds the config in-process or shells
out:

```python
# these two are the same request
extra={"policy.freeze_vision_encoder": False}
extra={"policy.freeze_vision_encoder": "false"}
```

For a boolean field that decoder accepts `false`/`no`/`off` and
`true`/`yes`/`on` in any case. `0` and `1` are integers to it, not booleans, and
are refused - as they are on the command line. Text that does not decode to the
field's declared type is refused naming the field and its type, rather than
stored as-is: a stored string is truthy, so `"false"` would otherwise read as
"true" and silently invert the flag.

Some policies freeze most of themselves by default, which `method="full"` does
not override - it selects strands' tuning strategy, not lerobot's per-policy
defaults. SmolVLA, for instance, ships `freeze_vision_encoder=True` and
`train_expert_only=True`, so full-model finetuning means asking for both:

```python
TrainSpec(
    dataset_repo_id="org/tictactoe",
    base_model="lerobot/smolvla_base",
    output_dir="/tmp/ft_out",
    steps=20000,
    method="full",
    extra={"policy_type": "smolvla",
           "policy.freeze_vision_encoder": False,
           "policy.train_expert_only": False},
)
```

Check what a policy freezes by default before assuming `method="full"` trains
all of it: `dataclasses.fields()` on its lerobot config class lists every knob
and its default.

#### RA-BC sample weighting (reward-aligned behavior cloning)

Reward-Aligned Behavior Cloning reweights the per-sample loss so high-progress
demonstration frames dominate - the technique behind the strongest
behavior-cloning ablations on long-horizon manipulation. lerobot >= 0.5.2 drives
it from a nested `SampleWeightingConfig` on `TrainPipelineConfig`
(`cfg.sample_weighting`, with fields `type` / `progress_path` / `head_mode` /
`kappa` / `epsilon`). Surface it through `extra` with a friendly dict whose keys
match those fields 1:1:

```python
TrainSpec(
    dataset_root="/data/folding_v3",
    base_model="lerobot/pi05_base",
    output_dir="/tmp/ft_out",
    extra={
        "policy_type": "pi05",
        "sample_weighting": {
            "type": "rabc",          # scheme: "rabc" or "uniform"
            "kappa": 0.01,           # high-progress threshold
            "head_mode": "sparse",   # SARM progress head ("sparse"/"dense")
            "progress_path": "/tmp/ft_out/sarm_progress.parquet",
        },
    },
)
# -> lerobot_train --sample_weighting.type=rabc --sample_weighting.kappa=0.01 \
#                  --sample_weighting.head_mode=sparse \
#                  --sample_weighting.progress_path=/tmp/ft_out/sarm_progress.parquet ...
```

The friendly keys are forwarded verbatim into `SampleWeightingConfig`. An
unknown key, an unsupported `type` (lerobot ships `rabc` and `uniform`), or a
lerobot too old to expose `cfg.sample_weighting` each raise an actionable error.
Omit `sample_weighting` entirely for standard (uniform) behavior cloning.

The `progress_path` parquet is produced from a trained SARM reward model - see
the SARM production loop below.

#### SARM reward model + the RA-BC production loop

RA-BC needs a per-frame *progress* signal (`sarm_progress.parquet`). SARM
(Stage-Aware Reward Model) learns that signal from demonstrations; lerobot
>= 0.5.2 trains it through the SAME `train(cfg)` entry point as a policy, but on
`cfg.reward_model` instead of `cfg.policy`. The full producing loop is three
strands calls:

```python
from strands_robots.training import (
    create_trainer, TrainSpec, compute_rabc_weights,
)

trainer = create_trainer("lerobot_local")

# 1. TRAIN a SARM reward model (single_stage needs no annotations).
trainer.train(TrainSpec(
    dataset_root="/data/folding_v3",
    output_dir="/tmp/sarm_out",
    steps=5000,
    extra={"reward_model": {
        "type": "sarm",
        "annotation_mode": "single_stage",
        "image_key": "observation.images.base",
    }},
))

# 2. COMPUTE per-frame RA-BC progress weights from the trained SARM.
progress = compute_rabc_weights(
    reward_model_path=trainer.latest_checkpoint("/tmp/sarm_out"),
    dataset_root="/data/folding_v3",
)

# 3. TRAIN the policy with RA-BC pointed at the produced parquet.
trainer.train(TrainSpec(
    dataset_root="/data/folding_v3",
    base_model="lerobot/pi05_base",
    output_dir="/tmp/ft_out",
    steps=20000,
    extra={"policy_type": "pi05",
           "sample_weighting": {"type": "rabc", "progress_path": progress}},
))
```

`extra["reward_model"]` works for every reward model lerobot registers on its
`RewardModelConfig` choice registry - `sarm` (default), `robometer`, `topreward`,
and `reward_classifier` today, plus any new type a future lerobot or a plugin
adds, with no strands change needed. Besides `type`, the dict accepts that
type's OWN config fields: e.g. SARM's `annotation_mode`
(`single_stage` / `dense_only` / `dual`), `image_key`, `state_key`; robometer /
topreward's `default_task`, `success_threshold`, `max_frames`; the classifier's
`num_classes`, `hidden_dim`. Fields that do not belong to the chosen type (e.g.
SARM's `annotation_mode` on `robometer`) are rejected with the list of that
type's configurable fields. The policy-only knobs (`sample_weighting`,
`relative_actions`, non-`full` `method`) are rejected on a reward-model run
rather than silently ignored.

A trained SARM can also be queried for a dense task-progress score in `[0, 1]`
(e.g. as an eval-time signal):

```python
from strands_robots.training import load_reward_model, reward_progress

model = load_reward_model("/tmp/sarm_out/checkpoints/last/pretrained_model")
scores = reward_progress(model, batch)   # list[float], one per batch element
```

#### Relative (delta) actions

Predicting actions as deltas from the current robot state - rather than
absolute targets - is part of the strongest manipulation ablations. lerobot
implements it as a matched processor pair built from
`config.use_relative_actions`: a `RelativeActionsProcessorStep` encodes
target->delta at train time, and the inverse `AbsoluteActionsProcessorStep`
decodes delta->target at inference. Both are saved into the checkpoint's
pre/post processors, so deployment via `lerobot_local` (which loads the saved
processor pipeline) restores the inverse decode automatically - no separate
inference-side wiring is needed.

```python
TrainSpec(
    dataset_root="/data/folding_v3",
    base_model="lerobot/pi05_base",
    output_dir="/tmp/ft_out",
    extra={"policy_type": "pi05", "relative_actions": True},
)
# -> lerobot_train --policy.type=pi05 --policy.use_relative_actions=true ...
```

Only `pi0` / `pi05` / `pi0_fast` expose `use_relative_actions`; the flag is
rejected (not silently ignored) for any other policy type.

#### Quantile normalization (molmoact2, pi05)

Some policies normalize `STATE`/`ACTION` with `NormalizationMode.QUANTILES`
(currently `molmoact2` and `pi05`) rather than mean/std or min/max. Quantile
normalization reads the dataset stats' quantile keys (`q01`..`q99`); a dataset
recorded *before* quantile stats existed carries only mean/std/min/max, so
lerobot either raises or silently mis-normalizes deep inside its stats plumbing
at train time. `validate()` catches this at spec time: when the resolved policy
normalizes with quantiles and a local `meta/stats.json` lacks the quantile keys,
it returns an actionable problem naming lerobot's remedy:

```bash
python -m lerobot.scripts.augment_dataset_quantile_stats \
    --repo-id=<your-dataset-repo-id> --root=/data/my_v3_dataset
```

Datasets recorded by `Robot.start_recording()` / `DatasetRecorder` on current
lerobot already include quantile stats (lerobot's `compute_episode_stats`
computes them by default), so they train `molmoact2` / `pi05` with no manual
stats surgery. The check is conservative: a Hub dataset with no local cache is
left unflagged (its quantiles are verified by lerobot when the shards load).

#### Dataset format version (`codebase_version`)

Every LeRobotDataset declares the format version it was written in as
`codebase_version` in `meta/info.json`. lerobot compares it against its own
`CODEBASE_VERSION` and refuses a dataset whose MAJOR is older - an older *minor*
loads with a warning. Only a `v2.1` root gets a message naming the dataset and
the converter; for any other older major lerobot raises
`NotImplementedError: Contact the maintainer on [Discord](...)` from inside the
exception constructor, naming neither the dataset nor the version.

The version sits in the same `meta/info.json` the trainer already reads for the
episode count, so `validate()` decides this offline and returns an actionable
problem instead:

```
dataset_root '/data/old_dataset' declares codebase_version 'v2.1' in
meta/info.json, which lerobot 0.6.2 cannot read (it loads 'v3.0' and refuses an
older major). Convert it with lerobot's own converter: python -m
lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=<your-dataset-repo-id>
```

lerobot ships exactly one conversion (`v2.1` -> `v3.0`), so a root older than
`v2.1` is told so plainly rather than handed a command that would fail. Datasets
recorded by `Robot.start_recording()` / `DatasetRecorder` are written in the
installed lerobot's own format, so they pass cleanly.

Like the quantile-stats check, this one is conservative: it flags only a
definite mismatch. A Hub dataset with no local cache is left unflagged -
`validate()` does not reach the network, so Hub-side metadata (the format
version, and the git tag lerobot resolves the revision from) is verified by
lerobot when the dataset loads.

#### Streaming a large Hub dataset (no full download)

Real datasets (BitRobot / HIW-500, ~50-500 GB) do not fit on a single edge node.
Point the trainer at a Hub dataset id and stream it - lerobot pulls shards on
the fly via `StreamingLeRobotDataset`, so disk stays bounded and the first
forward pass starts without waiting for a full download:

```python
TrainSpec(
    dataset_repo_id="org/hiw_500",   # train from the Hub, not a local root
    streaming=True,                  # -> --dataset.streaming=true
    base_model="lerobot/act_aloha_sim",
    output_dir="/tmp/ft_out",
    extra={"policy_type": "act"},
)
# -> lerobot_train --dataset.repo_id=org/hiw_500 --dataset.streaming=true ...
```

`dataset_root` is optional here - if given it is used as a local cache root.
`streaming=True` also works with a local `dataset_root` (streams from disk with
bounded RAM).

Held-out `val_episodes` splitting needs a local `meta/info.json` to count
episodes, because lerobot's split is a FRACTION and the count is what turns an
episode number into one. `validate()` therefore refuses `val_episodes` whenever
that count is unreadable - a Hub source with no `dataset_root`, or a
`dataset_root` cache directory nothing has been downloaded into yet - rather
than launch a run that trains on every episode and logs no validation loss. The
refusal names the two ways to get the split: point `dataset_root` at a populated
local copy of the dataset, or pass lerobot's own knobs directly with
`extra={"dataset.eval_split": 0.1, "eval_steps": 1000}`.

`streaming` and `val_episodes` cannot both be honored, so `validate()` refuses
the pair. A non-zero `eval_split` routes lerobot into `make_train_eval_datasets`,
which rebuilds BOTH splits as map-style `LeRobotDataset` objects without
consulting `dataset.streaming` - the streaming dataset it opened first is
discarded. The run therefore materializes the whole dataset, which is exactly
what `streaming` exists to avoid, and nothing reports it: an annulled stream is
indistinguishable from `streaming=False`. Set `streaming=False` to keep the
validation split, or `val_episodes=None` to keep the stream.

### GR00T (`groot`)

```python
TrainSpec(..., embodiment="GR1",
          tune={"llm": False, "visual": False, "projector": True, "diffusion": True},
          extra={"groot_root": "/path/to/Isaac-GR00T"})
# -> launch_finetune.py --embodiment_tag=GR1 --tune_projector=true ...
```

### Cosmos3 (`cosmos3`)

```python
TrainSpec(..., num_gpus=8,
          extra={"cosmos_root": "/path/to/cosmos-framework",
                 "sft_toml": "examples/toml/sft_config/action_policy_droid_repro.toml"})
# prepare(): convert_model_to_dcp ; train(): torchrun ... --sft-toml=... ;
# export(): DCP -> safetensors
```

## Dependencies & extras (per provider)

**Every `lerobot_local` row below also needs `lerobot[training]`, on CPU as well
as GPU.** LeRobot's `train()` calls
`require_package("accelerate", extra="training")` *before* it branches on
device, so "no GPU" does not mean "no extra" - and nothing on this path pulls
`accelerate` in: the `[lerobot]` extra is exactly `lerobot[feetech,dataset]`. The
`strands-robots` extras that declare `accelerate` belong to other providers
(`kimodo`, `cosmos3-diffusers`), so an install only has it by accident - `[all]`
does, because it pulls `[kimodo]`; `[lerobot]` alone does not.

```bash
pip install "lerobot[training]"
```

`validate()` reports an absent `accelerate` - and an absent `peft` when
`method="lora"` - as a preflight problem, so `train()` fails closed on it and
leaves `output_dir` untouched rather than clearing it on the way to a run that
cannot start. Either way the cause arrives in the `TrainResult` rather than as a
raise, so check `result.status` and surface `result.message` - it names the
package and the `lerobot[...]` extra that supplies it. An unchecked call hands
whatever consumes `checkpoint_dir` a `None` instead.

The base `strands-robots[lerobot]` extra covers **recording, streaming, and
loading a trained checkpoint**, and with `lerobot[training]` it covers **ACT /
diffusion from scratch**; VLA post-tunes pull in policy-specific stacks on top.
Install the extra that matches your `extra["policy_type"]` / provider — verified
on an L40S GPU:

| Provider / policy | Install | Notes |
|---|---|---|
| `lerobot_local` + ACT / diffusion | `pip install 'strands-robots[lerobot]' 'lerobot[training]'` | `[lerobot]` supplies torch + torchcodec + datasets; it does **not** supply `accelerate` |
| `lerobot_local` + `smolvla` | `pip install 'strands-robots[lerobot]' 'lerobot[training]' 'lerobot[smolvla]'` | lerobot 0.6's `[smolvla]` extra layers `transformers>=5.4.0,<5.6.0` + num2words on top. Do **not** pin `transformers==5.3.0` - it conflicts with lerobot 0.6's transformers floor. |
| `lerobot_local` + `pi0` / `pi05` | `pip install 'strands-robots[lerobot]' 'lerobot[training]' 'lerobot[pi]'` | lerobot 0.6's `[pi]` extra (same `transformers>=5.4.0,<5.6.0` range + scipy) |
| `groot` | Isaac-GR00T checkout, installed with `pip install -e` into the **same** environment as `strands_robots` (it pulls `omegaconf`, `tyro`, …); point `extra["groot_root"]` / `GR00T_ROOT` at the checkout | `gr00t` is imported in the calling interpreter, so it has to be importable there; `GR00T_ROOT` resolves relative configs, not the interpreter |
| `cosmos3` | cosmos-framework checkout (`uv sync --group=cu130-train`), installed into the **same** environment as `strands_robots`; point `extra["cosmos_root"]` / `COSMOS_ROOT` at the checkout | `cosmos_framework` is imported in the calling interpreter; multi-GPU goes through torch's programmatic `elastic_launch`, not a `torchrun` binary |

> **torchcodec / torch ABI:** the lerobot training dataloader decodes video via
> `torchcodec`, whose compiled `.so` must match the **exact** installed torch
> build. A torch *nightly* (e.g. `2.12.0.dev`) load-fails a stable-built
> torchcodec with `undefined symbol: ...MessageLogger` even when ffmpeg is
> present — and lerobot silently swallows the per-shard decode error, so
> training fails with a generic non-zero exit. Pin `torch` + `torchcodec`
> together (verified-good combo: `torch==2.10.0+cu128` + `torchcodec==0.10.0`).

> **One interpreter:** `LerobotTrainer` / `Gr00tTrainer` / `Cosmos3Trainer` call their
> backend as a library in the **same** interpreter that imports `strands_robots`, so
> there is no second interpreter to point them at. There is no `python_executable=`
> argument either, and because each constructor absorbs unknown keywords, passing one
> is silently a no-op rather than an error. Install the provider's deps into the
> environment your agent process runs in; otherwise `train()` reports
> `<package> is not importable from this interpreter` in `TrainResult.message`.
> `GR00T_ROOT` / `COSMOS_ROOT` (and `extra["groot_root"]` / `extra["cosmos_root"]`)
> resolve the checkout so relative configs load - they are not interpreter paths.

## See also

- [Recording](../recording.md) - produce the dataset.
- [Policy Providers](../policies/overview.md) - the inference peer of `Trainer`.
- [`examples/07_post_tune_any_policy.py`](https://github.com/strands-labs/robots/blob/main/examples/07_post_tune_any_policy.py) - the full loop in one script.
