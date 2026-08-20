### Added: VLM judge agent for recorded-episode labeling, layered on deterministic predicate verdicts

`strands_robots.episode_labels` records per-episode deterministic benchmark
verdicts and judge annotations (quality grade, fixed failure-mode taxonomy,
free-text note) in a schema-versioned JSON sidecar (`episode_labels.json`)
next to the dataset, so training can filter episodes without rewriting them.
The precedence is structural: the judge surface writes only the `judge` block,
refuses an episode with no deterministic verdict, and records a disagreeing
`success_opinion` as a `disputes_verdict` annotation while the verdict stands.
Four agent tools (`load_episode`, `sample_frames` - with optional camera-frame
decoding for a multimodal judge - `read_predicate_verdict`, `write_label`) and
`create_judge_agent` assemble a model-provider-agnostic strands judge agent.
`sample_frames`' motion summary carries `rms_state_jerk` (rms third difference
of the state series, so `jerky_motion` is groundable from state alone - a max
first difference is a peak-velocity statistic that cannot see a superimposed
jitter) beside `max_state_delta`. With images requested, every recorded camera
is decoded (one block per camera per sampled position - a single canonical
view measurably drops verdicts the interleave keeps), and the leading text
block states the block count and the position-major, sorted-camera grouping.
`examples/17_judge_recorded_episodes.py` runs the pipeline end to end: record
with per-episode predicate stops, verdicts, scripted judge, judge/human
agreement on a holdout, and a filtered ACT re-training run on the
judge-approved subset (`TrainSpec.extra={"dataset.episodes": [...]}`).
Documented in `docs/data/episode-labels.md`.
