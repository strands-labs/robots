---
description: Predicate verdicts on every episode, a judge agent's grade and failure tag on top, and a filter for the episodes worth training on.
---

# Label and judge

Every episode gets a verdict from the simulator's predicates, a judge's quality grade and failure-mode tag on top, and a filter for the episodes worth training on. The judge can annotate a verdict, never overturn it.

Runs without lerobot, on the toy dataset from [verify](verify.md):

```python
import json, pathlib
import pyarrow as pa, pyarrow.parquet as pq
from strands_robots.episode_labels import (
    annotate_episode, filter_episodes, labels_path, record_deterministic_verdicts,
)

root = pathlib.Path("/tmp/toy_dataset")                 # the dataset the verify page builds
(root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
pq.write_table(pa.table({"episode_index": [0, 1, 2], "length": [90, 90, 90]}),
               root / "meta/episodes/chunk-000/file-000.parquet")
(root / "meta/info.json").write_text(json.dumps({"total_episodes": 3, "total_frames": 270, "fps": 30, "features": {}}))
(root / "episode_labels.json").unlink(missing_ok=True)
verdicts = [   # the per-episode list evaluate_benchmark returns
    {"episode": 0, "success": True,  "failure": False, "steps": 90, "cumulative_reward": 1.0, "seed": 0},
    {"episode": 1, "success": True,  "failure": False, "steps": 90, "cumulative_reward": 1.0, "seed": 1},
    {"episode": 2, "success": False, "failure": True,  "steps": 90, "cumulative_reward": 0.0, "seed": 2},
]
record_deterministic_verdicts(root, verdicts, benchmark="so101_reach")         # stage one
annotate_episode(root, 0, quality="high", note="clean approach", model="judge-vlm")
annotate_episode(root, 1, quality="medium", failure_mode="near_miss", success_opinion=False, model="judge-vlm")

print(filter_episodes(root, require_success=True, min_quality="medium"))                          # [0, 1]
print(filter_episodes(root, require_success=True, min_quality="medium", exclude_disputed=True))   # [0]
print(labels_path(root))                                                                          # /tmp/toy_dataset/episode_labels.json
```

## Two stages, one precedence

| stage | who | writes | can it change the other block |
|---|---|---|---|
| deterministic | the benchmark's predicates, from simulator state (`evaluate_benchmark`) | the `deterministic` block: `success`, `failure`, `steps`, `cumulative_reward`, `seed` | no |
| judge | a VLM agent or decider reading the recording | the `judge` block: `quality`, `failure_mode`, `note`, `success_opinion`, `disputes_verdict`, `model`, `labeled_at` | no |

A recorded frame is the state before its action, so a `stop_when` episode ends one step short of the state that fired it. `annotate_episode` refuses an episode with no deterministic verdict yet. A `success_opinion` that contradicts the predicate is recorded as `disputes_verdict: true` for a human to review; the `deterministic` block stays byte-identical.

## The sidecar

Labels live in `episode_labels.json` at the dataset root, next to LeRobot's `meta/`, `data/` and `videos/`, so training can filter episodes without rewriting the dataset. `schema_version` is 1; `read_labels(root)` returns the document, `deterministic_verdict(root, episode)` one verdict.

Vocabulary is fixed so filters match on identity:

| field | values |
|---|---|
| `quality` | `low`, `medium`, `high` |
| `failure_mode` | `jerky_motion`, `near_miss`, `camera_occlusion`, `wrong_but_lucky`, `drift`, `collision`, `incomplete`, `other` |

`near_miss` and `wrong_but_lucky` are legal on a deterministically successful episode on purpose: they mark a success worth excluding from training data.

## The judge agent

Four `@tool`s in `strands_robots.tools.episode_judge` drive a judge; `create_judge_agent(model=None)` assembles them with a system prompt carrying the two-stage doctrine. Pass any strands model object, or none for the default:

```python title="sketch"
from strands_robots.tools.episode_judge import create_judge_agent

judge = create_judge_agent()
judge(f"Label every episode of the dataset at {root}. Sample four frames each, with images.")
```

| tool | returns |
|---|---|
| `load_episode(root, episode)` | frame count, features, whether a verdict and a label exist yet |
| `sample_frames(root, episode, n_frames=4, include_images=False)` | evenly spaced frames: `observation.state` and timestamps, a decoded image per camera when asked, and `rms_state_jerk` to ground `jerky_motion` without images |
| `read_predicate_verdict(root, episode)` | the authoritative deterministic verdict |
| `write_label(root, episode, quality, failure_mode=None, note="", success_opinion=None, judge_model="")` | the judge block, through `annotate_episode` |

Every tool returns the `{"status", "content"}` envelope and never raises. `sample_frames` with images needs the `[lerobot]` extra to decode video.

## A decision-model judge

`DeciderJudge` asks a running [Strands Decider](https://github.com/strands-labs/strands-decider) (`serve --vision`) for typed answers per episode and writes them through `write_label` only above `min_confidence`; the rest are reported `deferred`, for the agent judge or a person.

```python title="sketch"
from strands_robots.tools.decider_judge import DeciderJudge

DeciderJudge("http://127.0.0.1:8000").judge_dataset(str(root), task="reach the target")
```

## Filtering for training

`filter_episodes(root, require_success=True, min_quality="medium", exclude_disputed=False)` returns the episode indices that clear the bar. Unlabeled and deferred episodes are excluded: with no `judge` block there is no quality to compare. Pass the list to lerobot's `--dataset.episodes` or to `StreamingDatasetReader.open(episodes=...)` ([stream and sync](stream-and-sync.md)).

## Checking the judge

`measure_agreement(root, human_labels)` compares the judge's labels with a human-labeled holdout and reports agreement per field against a constant-answer baseline, so a judge that always says `medium` cannot look accurate.
