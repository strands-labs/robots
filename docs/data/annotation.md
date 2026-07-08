---
description: Add steerable language annotations (language_persistent / language_events) to a LeRobot dataset with lerobot-annotate, then train a language-conditioned VLA.
---

# Steerable dataset annotation

Steerable policies accept a language and preference conditioning signal at
inference time (subtasks, interjections, VQA-style grounding) on top of the
usual task string. Training one needs a dataset whose frames carry those
language signals. LeRobot 0.6 ships the pipeline that produces them,
`lerobot-annotate`, and `strands-robots` datasets are ordinary LeRobot v3
datasets, so the two compose directly:

```
record (strands-robots)  ->  annotate (lerobot-annotate)  ->  train steerable VLA
```

`strands-robots` does not reimplement the annotation pipeline. It is a
GPU-heavy, fast-moving research module in LeRobot; mirroring it would duplicate
a moving target. This page documents how to drive the upstream tool against a
dataset you recorded with `strands-robots` and what it writes.

## What `lerobot-annotate` actually is

It is an **automated Vision-Language-Model labeling pipeline**, not a
human-in-the-loop UI. It reads each episode's frames, sends sampled
contact-sheets to a Qwen-VL model served over an OpenAI-compatible endpoint
(vLLM, auto-spawned by default), and rewrites the dataset's parquet shards in
place with two new language columns. There is no interactive editor and no
manual review step; a human only sets the config and inspects the result.

The pipeline runs six phases in dependency order:

1. **plan** module - subtasks, plan, memory, optional task augmentation
2. **interjections** module - interjections plus paired speech
3. **plan update** - re-emit plan rows at each interjection timestamp
4. **vqa** module - general visual-question-answer pairs
5. **validator** - schema and coverage checks on the staged output
6. **writer** - rewrite `data/chunk-*/file-*.parquet` and update `meta/info.json`

## Requirements

- `lerobot>=0.6` (already pinned by `strands-robots`) with the annotation deps
  (`datasets`, `pyarrow`, `av`/`torchcodec`, `openai`).
- A Qwen-VL model served on an OpenAI-compatible endpoint. `lerobot-annotate`
  auto-spawns a local vLLM server by default (`--vlm.auto_serve=true`), which
  needs a GPU; point `--vlm.api_base` at an existing server to reuse one.
- For dataset-scale runs, distribute with Hugging Face Jobs - see
  `examples/annotations/run_hf_job.py` in the LeRobot repository.

## Running it on a strands-robots dataset

Record a dataset the usual way (see [Recording & datasets](../recording.md)):

```python
from strands_robots import Robot

sim = Robot("so100")
sim.start_recording(repo_id="user/pick_place", task="pick up the cube", fps=30)
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", duration=10.0)
sim.stop_recording()
```

Then annotate it in place with the LeRobot console script:

```bash
lerobot-annotate \
    --root=~/.strands_robots/datasets/user/pick_place \
    --vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct
```

Common flags:

- `--repo_id=user/pick_place` - download the source from the Hub instead of `--root`.
- `--new_repo_id=user/pick_place_annotated` - write to a separate target.
- `--push_to_hub=true` - upload the annotated dataset when finished.
- `--vlm.api_base=http://host:8000/v1` with `--vlm.auto_serve=false` - reuse a
  running vLLM server instead of spawning one.
- `--only_episodes=0,1,2` - annotate a subset while iterating.
- toggle modules with `--plan.enabled`, `--interjections.enabled`, `--vqa.enabled`.

## What it writes: the language columns

Two columns are added to every frame row (and advertised in `meta/info.json`
via `language_feature_info()`, so non-streaming loads keep working):

| Column | Shape | Meaning |
| --- | --- | --- |
| `language_persistent` | list of rows, each with a `timestamp` | A state that becomes active at a moment and stays active until superseded (subtasks, plan, memory, motion, task_aug). |
| `language_events` | list of rows, no `timestamp` | An instantaneous event stored on the frame whose timestamp is its firing time (interjection, vqa, trace). |

Row fields:

- persistent row: `role`, `content`, `style`, `timestamp` (float32), `camera`, `tool_calls`
- event row: `role`, `content`, `style`, `camera`, `tool_calls`

Styles are drawn from a fixed registry:

- persistent styles: `subtask`, `plan`, `memory`, `motion`, `task_aug`
- event-only styles: `interjection`, `vqa`, `trace`
- view-dependent styles (`camera` must reference an `observation.images.*`
  key): `vqa`, `trace`. Every other style carries `camera=None`.

The pipeline also appends a canonical `say` tool schema (`SAY_TOOL_SCHEMA`) to
`meta/info.json`'s `tools` so speech interjections have a declared call surface.

## Feeding a steerable VLA

An annotated dataset is a superset of the original: policies that ignore the
language columns train unchanged, while a language-conditioned VLA consumes
`language_persistent`/`language_events` as extra conditioning. Point the
existing training workflow (see [VLA-on-G1 Workflow](../training/vla_workflow.md))
at the annotated `repo_id`; no `strands-robots` code changes are needed to carry
the columns through recording, since they are added after recording by the
annotation step.

## References

- LeRobot annotation pipeline: `lerobot/annotations/steerable_pipeline/`
- CLI: `lerobot-annotate` (`lerobot/scripts/lerobot_annotate.py`)
- Column schema: `lerobot/datasets/language.py`
- Distributed example: `examples/annotations/run_hf_job.py`
