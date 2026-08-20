---
description: Label recorded episodes with a VLM judge layered on deterministic predicate verdicts - quality grades, failure-mode tags, and dispute annotations in a schema-versioned sidecar, then train on the filtered subset.
---

# Episode labels (judge on top of predicates)

Episode validation with `evaluate_benchmark` is deterministic-only: predicates
over simulator state score success and failure per the task definition. That
covers the objective checks, but not what a human reviewer looks at in
recorded episodes - smooth vs jerky motion, near-misses, camera occlusion,
wrong-but-lucky successes. For automated episode farming, that labeling step
is the remaining human bottleneck.

`strands_robots.episode_labels` closes it with a **two-stage verdict**, the
same doctrine as the safety and dispatch layers:

1. **Deterministic predicates run first and are authoritative** for what they
   can measure. Their per-episode verdicts (from `evaluate_benchmark`, or from
   `run_policy(stop_when=...)` per-episode predicate stops) are recorded with
   `record_deterministic_verdicts`.
2. **A judge annotates on top** - a quality grade, a failure-mode tag from a
   fixed taxonomy, a free-text note. The judge can never overturn a
   deterministic verdict: `annotate_episode` (and the `write_label` tool)
   writes only the `judge` block, refuses an episode with no deterministic
   verdict, and records a disagreeing `success_opinion` as
   `disputes_verdict: true` while the verdict stands. Pinned by
   `tests/test_episode_labels.py` and `tests/tools/test_episode_judge.py`.

End-to-end walkthrough: `examples/17_judge_recorded_episodes.py` (record ->
verdicts -> judge -> agreement measurement -> filtered re-training).

## The sidecar

Labels live in `episode_labels.json` at the dataset root, next to LeRobot's
own `meta/` / `data/` / `videos/` - episode-level metadata, so training can
filter without rewriting a single parquet shard. The file is schema-versioned
(`schema_version: 1`); a version this build does not know is refused on read
rather than misread. Writes are two-phase (temp file + atomic rename).

```json
{
  "schema_version": 1,
  "benchmark": "so100_pan_reach",
  "episodes": {
    "0": {
      "episode_index": 0,
      "deterministic": {
        "success": false,
        "failure": true,
        "steps": 15
      },
      "judge": {
        "quality": "low",
        "failure_mode": "incomplete",
        "note": "budget exhausted at 15 steps without reaching the target",
        "success_opinion": true,
        "disputes_verdict": true,
        "model": "scripted-heuristic",
        "labeled_at": 1755590400.0
      }
    }
  }
}
```

Field domains:

| field | domain |
|---|---|
| `episode_index` | a non-negative whole number, on the shared domain every surface applies. Holds in each spelling the index arrives in: the `episode` argument of `deterministic_verdict` / `annotate_episode` and the judge tools, an `episodes[i]["episode"]` entry handed to `record_deterministic_verdicts`, and a key of `measure_agreement`'s holdout mapping. A value outside it selects a *different* episode rather than failing slowly - `True` is `1` to an index - so it is refused and named |
| `quality` | `low` / `medium` / `high` (ordered; filters compare rank). An *execution* grade, orthogonal to the outcome - see below |
| `failure_mode` | `null` or one of `jerky_motion`, `near_miss`, `camera_occlusion`, `wrong_but_lucky`, `drift`, `collision`, `incomplete`, `other` |
| `success_opinion` | `null` (no opinion) or a boolean |
| `disputes_verdict` | derived: opinion present and different from the deterministic `success` |
| `model` | free-form provenance (`"human"`, a model id, an endpoint) |

A `failure_mode` is deliberately legal on a deterministically *successful*
episode: `near_miss` and `wrong_but_lucky` are exactly the annotations that
make a success worth excluding from training data.

`quality` grades the **execution visible in the recording** - smoothness,
directness, control - never the outcome. The deterministic verdict already
carries success/failure, and `filter_episodes` gates on that verdict, so a
grade that re-derives the outcome carries no information in the one place it
is consulted: with `require_success=True` it only ever discriminates among
successes, and a jerky or lucky success graded `high` for succeeding is
exactly the episode the grade exists to exclude. This is not a hypothetical
drift: measured on a graded five-recording ladder with exact ground truth
(PR #2486 review), an unsteered VLM's grade tracked the outcome - `low` for
every failure, `high` only for the success - so `JUDGE_SYSTEM_PROMPT` states
the contract where the model reads it (a clean failure can be `medium` or
`high`; a jerky or lucky success can be `low`), and calibration
(`measure_agreement`) is where to confirm a given judge honours it.

## The judge agent

Four tools drive the labeling, assembled by
`strands_robots.tools.episode_judge.create_judge_agent` - model-provider
agnostic (any strands model object; a local OpenAI-compatible VLM endpoint
works, no cloud dependency required):

- `load_episode` - frame count, features, whether a verdict/label exists.
- `sample_frames` - evenly spaced state vectors plus a motion summary:
  `rms_state_jerk` (rms third difference of the state series - jerk is the
  third derivative, so this is the field that grounds `jerky_motion` from
  state alone) and `max_state_delta` (peak per-step delta, for spotting
  discontinuities); `include_images=True` decodes the camera frames into
  image blocks for a multimodal judge (needs the `lerobot` extra). Every
  recorded camera is included - one block per camera per sampled position,
  position-major, cameras in sorted order, with the count and grouping
  stated in the leading text block. That is deliberate, not a missing
  simplification: the same world motion can be well above a judge's
  legibility threshold in one view and below it in another (a 185 mm slide
  measured as 84 px of travel in one camera and 22 px in the other, with
  the verdict lost on the weaker view alone - PR #2486 review), so
  sampling one canonical camera would drop verdicts the interleave keeps.
- `read_predicate_verdict` - the authoritative deterministic verdict.
- `write_label` - the annotation; structurally unable to touch the verdict.

Two failure modes lean on judge capability rather than on a payload field,
so calibrate before trusting them: `jerky_motion` is grounded for a
text-only judge by `rms_state_jerk`, but `camera_occlusion` is inherently a
claim about *one* view that the payload's unlabelled image blocks cannot
name, and the open-weights VLM measured on PR #2486 did not tag even a
total single-camera occlusion (0 visible object pixels in every sampled
frame of that view) from any presentation. Expect `camera_occlusion` from a
human or a stronger multimodal judge, and treat any direction phrase in a
free-text `note` as a statement about a camera frame, not about the world.
The `note` is for humans and is never parsed by anything downstream - the
filterable channels are the closed vocabularies, and that split is measured,
not stylistic: on a frozen-arm control clip (arm silhouette travelling ~1 px
over the whole episode) 11 of 12 sampled free-text descriptions from the same
VLM said the arm moves toward the object, while the closed-vocabulary pick on
the identical clip was right at mass 0.95 (PR #2486 review).

```python
from strands.models import BedrockModel  # or any strands model provider
from strands_robots.tools.episode_judge import create_judge_agent

judge = create_judge_agent(model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-5-v1:0"))
judge("Label every episode of the dataset at /data/pick_place. "
      "Use sample_frames with include_images=True to look at the recording.")
```

## Calibration before trust

Measure judge/human agreement on a small human-labeled holdout before letting
the judge filter training data - the measurement ships with the pipeline
(`measure_agreement`), not as a promise:

```python
from strands_robots.episode_labels import measure_agreement

report = measure_agreement("/data/pick_place", {
    3: {"quality": "high", "failure_mode": None},
    7: {"quality": "low", "failure_mode": "jerky_motion"},
})
print(report["quality_agreement"], report["disagreements"])
```

## Filtering and re-training

`filter_episodes` selects on the deterministic verdict (authoritative - a
judge opinion never admits a failed episode) plus the judge's quality grade:

```python
from strands_robots.episode_labels import filter_episodes
from strands_robots.training import TrainSpec, create_trainer

chosen = filter_episodes("/data/pick_place", require_success=True, min_quality="medium")

trainer = create_trainer("lerobot_local", device="cpu")
spec = TrainSpec(
    dataset_root="/data/pick_place",
    base_model="",
    output_dir="/data/pick_place_ft",
    steps=2000,
    extra={"policy_type": "act", "dataset.episodes": chosen},
)
result = trainer.train(spec)
```

The subset reaches lerobot as `DatasetConfig.episodes` through the typed
`extra` passthrough; the dataset itself is untouched. The same list feeds the
read side: `stream_dataset(..., episodes=chosen)`.

## Relation to steerable annotation

[Steerable annotation](annotation.md) (`lerobot-annotate`) writes *frame-level
language columns* into the dataset's parquet shards for training
language-conditioned policies. Episode labels are the complementary layer:
*episode-level* quality/failure metadata in a sidecar, for deciding **which**
episodes to train on. The two compose - annotate the episodes the judge kept.
