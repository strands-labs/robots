---
description: DatasetRecorder - LeRobot v3 dataset writer used by both Simulation and HardwareRobot.
---

# Recording & datasets

```python
from strands_robots import Robot

sim = Robot("so100")
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", duration=10.0)
sim.stop_recording()
# LeRobot v3 dataset written to ~/.strands_robots/datasets/
```

`start_recording` requires `[lerobot]`. Without it, use `start_cameras_recording` for plain MP4.

## Selecting which cameras to record

By default every camera in the scene is recorded into the dataset - including
the implicit `default` free camera that exists even before you call
`add_camera`. A policy that declares a fixed set of image features (e.g. SmolVLA
expects exactly `observation.images.camera1/camera2/camera3`) then trains
against a dataset that carries a stray `observation.images.default` view it
never asked for, and the extra MP4 stream bloats every episode.

Pass `cameras=` to record exactly the views the policy expects:

```python
sim.add_camera(name="camera1", ...)
sim.add_camera(name="camera2", ...)
sim.add_camera(name="camera3", ...)
sim.start_recording(
    repo_id="user/my_dataset", task="pick up the cube", fps=30,
    cameras=["camera1", "camera2", "camera3"],   # drops the implicit 'default'
)
```

The dataset schema then declares only those three image features. Names may be
given in raw MuJoCo form (`arm0/wrist_cam`) or schema-safe form
(`arm0__wrist_cam`); an unknown name fails loudly and lists the available
cameras rather than silently recording the wrong set. Omit `cameras=` to keep
the legacy behavior of recording every camera - in that case a one-time
warning is logged when the implicit `default` overview camera is swept in
alongside your real sensor cameras, so the stray view is never recorded
silently.

### Where the dataset is written (`root` / `overwrite`)

`root` is the on-disk directory for the dataset (defaults to the LeRobot cache
under `repo_id` when omitted). Passing an existing **empty** directory - for
example one returned by `tempfile.mkdtemp()` - is accepted and recorded into:

```python
import tempfile
root = tempfile.mkdtemp()                       # existing empty dir
sim.start_recording(repo_id="user/my_dataset", root=root, fps=30)   # records here
```

When `root` already contains a LeRobotDataset (a `meta/` directory),
`start_recording` **resumes** it and appends new episodes unless
`overwrite=True`, which wipes and recreates it. A `root` that exists, is not a
LeRobotDataset, and is **not empty** is left untouched and reported as an error
rather than clobbered - pass `overwrite=True` or choose a new/empty `root`.

When you drive recording through the `run_policy` tool (which owns the
`start_recording` -> rollout -> `stop_recording` cycle), forward the same
subset with `dataset_cameras=`:

```python
from strands_robots.tools.run_policy import run_policy

run_policy(
    simulation=sim,
    robot_name="so101",
    policy_provider="lerobot_local",
    instruction="pick up the cube",
    n_episodes=1,
    dataset_root="/tmp/my_dataset",
    dataset_cameras=["camera1", "camera2", "camera3"],  # drops the implicit 'default'
)
```

When set, `dataset_cameras` is forwarded as `start_recording(cameras=...)`,
which both the MuJoCo and Newton backends support, so the subset is applied
identically on either engine. Omit it (the default `None`) to record every
scene camera - the default path forwards no `cameras` kwarg at all.

To also capture a rollout MP4 (the visual artifact for review or VLM
defect-checking), pass the same `video={...}` config the
`Simulation.run_policy` facade accepts - the tool forwards it per episode:

```python
run_policy(
    simulation=sim,
    robot_name="so101",
    policy_provider="lerobot_local",
    instruction="pick up the cube",
    n_episodes=3,
    dataset_root="/tmp/my_dataset",
    video={"path": "/tmp/rollout.mp4", "fps": 30, "camera": "camera1",
           "width": 640, "height": 480},
)
```

`video["path"]` is required to enable recording; a falsy/absent path disables
it. For `n_episodes > 1` an `_ep{i}` suffix is inserted before the extension
(`/tmp/rollout_ep0.mp4`, `_ep1`, ...) so episodes do not overwrite one another -
matching the facade's own multi-episode naming. The returned `{"json": {...}}`
block carries `video_paths`, the list of MP4s that actually landed on disk
(only existing files are reported, never the requested paths).

![run_policy tool rollout video (SmolVLA on a simulated SO-101, MuJoCo headless)](assets/run_policy_video_demo.gif)

*A SmolVLA-on-SO-101 rollout recorded through the `run_policy` tool's `video=`
config (MuJoCo headless, `MUJOCO_GL=egl`).*

## Multi-episode recording

A recording session is one dataset. The simplest way to collect N episodes in
one session is `run_policy(n_episodes=N)` - it runs N rollouts back-to-back,
flushes a dataset episode boundary after each, and resets the sim between
episodes for you:

```python
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", n_steps=60, n_episodes=20)
sim.stop_recording()
# -> 20 episodes, each with its own episode_index / length / from_index / to_index
```

`n_steps` (or `duration`) is the per-episode horizon. `reset_between=False`
chains episodes from the previous end state instead of resetting. When a `seed`
is given it is offset per episode (`seed + i`) for reproducible-yet-distinct
rollouts, and a `video={...}` config is written per episode to a path with
`_ep{i}` inserted before the extension so episodes do not overwrite one another.
The aggregate result carries `n_episodes_completed`, `episodes_saved`,
`total_steps`, and a per-episode list in its `{"json": {...}}` block.

If you need full control over each rollout (different instructions, custom
randomization, conditional logic between episodes), drive the loop yourself and
call `save_episode()` after each rollout to flush it as its own episode:

```python
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
for _ in range(20):
    sim.reset()
    sim.run_policy(robot_name="so100", instruction="pick up the cube",
                   policy_provider="mock", n_steps=60)
    sim.save_episode()        # flush this rollout as one episode
sim.stop_recording()          # flushes any trailing rollout automatically
```

`save_episode` is idempotent on an empty buffer, so it is safe to call
unconditionally inside a loop. LeRobot computes `stats.json` per episode and then
aggregates, so per-rollout boundaries keep dataset statistics correct across the
`reset()` teleport between rollouts.

`reset()` is itself an episode boundary during a recording: if frames are
buffered when you call it, `reset()` flushes them as their own episode before
re-initializing the world (it reports the saved episode in its result text).
This means a bare `run_policy` + `reset` collection loop already produces one
episode per rollout - the explicit `save_episode()` is optional when you reset
between rollouts:

```python
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
for _ in range(20):
    sim.run_policy(robot_name="so100", instruction="pick up the cube",
                   policy_provider="mock", n_steps=60)
    sim.reset()               # flushes this rollout as one episode, then resets
sim.stop_recording()          # flushes any trailing rollout automatically
# -> 20 episodes
```

Without `n_episodes`, an explicit `save_episode()`, or a `reset()` between rollouts, all
20 rollouts append to the same buffer and `stop_recording` flushes them as a
single `episode_index=0` (1200 steps in one episode). To DISCARD a partial
rollout instead of flushing it on the next `reset()`, call
`clear_episode_buffer()` first.

## Verifying episode count

An LLM agent narrating "20 episodes recorded" is not proof: a single
`run_policy(n_episodes=1)` (or 20 looped tool calls into one open buffer)
produces one merged `episode_index=0` mega-episode while the agent believes it
recorded 20. Never trust agent narration for dataset structure - verify against
the on-disk metadata. After `stop_recording`, call `verify_dataset_episodes`:

```python
sim.stop_recording()
result = sim.verify_dataset_episodes(expected=20)
assert result["status"] == "success"   # else MISMATCH, fail loud
```

It checks two independent sources of truth and requires them to AGREE:

* the parquet under `meta/episodes/**/*.parquet` (the distinct `episode_index`
  set - the ground truth), and
* the `total_episodes` header in `meta/info.json`.

`status` is `"error"` when the parquet count differs from `expected` OR when the
parquet disagrees with `info.json` (an internally inconsistent dataset, e.g. an
interrupted finalize - `sources_agree` is then `False`), so a dataset that
happens to match `expected` on one source but not the other still fails. The
`{"json": {...}}` block carries `expected`, `actual`, `info_total_episodes`,
`sources_agree`, `episode_indices`, and `total_frames` for programmatic CI
gating. The pure-pyarrow `read_dataset_episode_indices(root)` exposes the same
facts without instantiating a `LeRobotDataset`.

The same check runs from the shell against any LeRobot dataset on disk, with an
exit code suitable for CI:

```bash
strands-robots verify-dataset /path/to/dataset --expected 20   # exit 0 pass, 1 fail
strands-robots verify-dataset /path/to/dataset --json          # machine-readable report
strands-robots verify-dataset /path/to/dataset --no-check-videos  # skip the per-episode MP4 checks
```

`verify-dataset` reuses the same pure-pyarrow `read_dataset_episode_indices`
helper (no `lerobot` import) and flags four failure modes: the mega-episode
(fewer distinct episodes than `--expected`), `meta/info.json` `total_episodes` /
`total_frames` drifting from the parquet ground truth (caught even without
`--expected`), any episode below `--min-frames` (default 1), and - unless
`--no-check-videos` is passed - any per-episode video file that is missing or
empty on disk. The last check is the video-modality sibling of the
mega-episode class: a dataset can carry the right episode count yet have no
pixels because the recorder's video encoder failed or the MP4 streams were
never written. It resolves each camera's MP4 from `meta/info.json`'s
`video_path` template and the episode parquet's `videos/<key>/chunk_index` /
`file_index` columns, and reports the count it checked in
`video_files_checked`. The programmatic form is
`strands_robots.verify_dataset.verify_dataset(root, expected=None, min_frames=1, check_videos=True)`,
which returns the same report dict.

## Recording paths

| Method | Extra needed | Output |
|--------|-------------|--------|
| `start_recording` / `stop_recording` | `[lerobot]` | LeRobot v3 (parquet + MP4) |
| `save_episode` | `[lerobot]` | Close current rollout as one episode (call once per `run_policy` for N episodes) |
| `start_cameras_recording` / `stop_cameras_recording` | `[sim-mujoco]` alone | Plain MP4, no parquet |

## Video codec (H.264 default, AV1 opt-in)

`start_recording` (and `DatasetRecorder.create`/`resume`) default to
`vcodec="h264"`. H.264 is universally decodable - including by OpenCV's
`cv2.VideoCapture`, which is what many downstream VLM video readers use to
*watch* a recorded episode. The recorded per-camera MP4s therefore reopen and
decode everywhere without a transcode step:

```python
import cv2
cap = cv2.VideoCapture(".../videos/observation.images.base/chunk-000/file-000.mp4")
ok, frame = cap.read()   # H.264: ok is True, frame is a real (H, W, 3) array
```

Pass `vcodec="libsvtav1"` to opt into AV1 for smaller files in
storage-constrained training pipelines. AV1 reads back fine through LeRobot's
own loader (`torchcodec`/`pyav`), but OpenCV wheels commonly lack an AV1 decoder
and silently yield **0 frames** (`VideoCapture.read()` returns `False`
immediately even though the stream has frames) - so avoid AV1 if anything in
your pipeline reads the videos through OpenCV.

The flat `vcodec` value accepts either codec names (`h264`, `hevc`,
`libsvtav1`) or ffmpeg encoder names (`libx264`, `libx265`); it is normalized
to whatever the installed LeRobot version's encoder config expects. An
unsupported codec fails loudly rather than silently reverting to the default.

## DatasetRecorder direct API

```python
from strands_robots.dataset_recorder import DatasetRecorder

recorder = DatasetRecorder.create(
    repo_id="user/my_dataset",
    fps=30,
    robot_type="so100",
    # When recording from a real LeRobot hardware robot pass the schema dicts
    # straight through:
    #   robot_features=robot.observation_features,
    #   action_features=robot.action_features,
    # When recording from a sim Robot (no `observation_features` attr), pass
    # `joint_names=[...]` instead - the recorder builds the schema for you.
    camera_keys=["default"],
    joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    task="pick up the red cube",
    # root=None → ~/.strands_robots/datasets/
    # vcodec="h264", streaming_encoding=True, image_writer_threads=4
)

for step in control_loop:
    recorder.add_frame(observation, action, task="pick up the red cube")
recorder.save_episode()
recorder.finalize()
recorder.push_to_hub(tags=["so100", "sim"], private=False)
```

Append to existing dataset (requires `lerobot>=0.5.2`):

```python
recorder = DatasetRecorder.resume(repo_id="user/my_dataset", task="pick up the blue cube")
recorder.add_frame(observation, action)
recorder.save_episode()
recorder.finalize()
```

## Instance methods

| Method | What |
|--------|------|
| `add_frame(observation, action, task=None, camera_keys=None)` | Append one timestep |
| `save_episode()` | Flush buffer as a new episode |
| `clear_episode_buffer()` | Discard current episode |
| `finalize()` | Write metadata, stats, close writers |
| `push_to_hub(tags=None, private=False)` | Upload to a versioned HF dataset repo |
| `sync_to_bucket(bucket, run_id=None, private=True)` | Sync to a mutable HF Storage Bucket (`hf://buckets/...`) — Xet-deduped collection target; needs the `hf` CLI. `bucket` (`name` or `org/name`) and `run_id` (single segment) are allowlist-validated (`[A-Za-z0-9._-]`, no traversal) before the sync |

## Read back

Fully materialized (downloads everything):

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="user/my_dataset", root="/tmp/my_dataset")
print(len(ds), ds[0].keys())
```

## Stream back (no full download)

`sim.stream_dataset()` is the in-process counterpart to `start_recording` /
`stop_recording` — it reads frames lazily from the Hub (or a local `root`) via
LeRobot's `StreamingLeRobotDataset`. Camera frames are decoded on the fly from
the MP4 shards; state/action come from the parquet shards.

```python
from strands_robots import Robot

sim = Robot("so100")
reader = sim.stream_dataset(
    "user/my_dataset",                 # or a local repo_id + root=
    root="/tmp/my_dataset",
    delta_timestamps={                 # optional: stacked time windows + *_is_pad masks
        "observation.state": [-0.0667, -0.0333, 0.0],
        "action": [0.0, 0.0333, 0.0667],
    },
    shuffle=False,                     # chronological for replay/eval
)
print(reader.num_episodes, reader.num_frames, reader.fps)
for frame in reader:
    ...

# torch DataLoader (shuffles INTERNALLY — do not pass shuffle=True):
for batch in reader.dataloader(batch_size=64, num_workers=4):
    ...
```

Equivalently, the standalone reader: `from strands_robots import StreamingDatasetReader`.

Useful kwargs (forwarded to `StreamingLeRobotDataset`, version-tolerant):
`episodes=[...]` (subset without download), `buffer_size`, `max_num_shards`,
`return_uint8=True` (default; halves frame bandwidth), and
`drop_videos=True` (proprio-only — skips video decode entirely, so it works on
edge devices without a torchcodec wheel).

For **training**, the upstream trainer uses the same engine:

```bash
python -m lerobot.scripts.train policy=act \
  dataset.repo_id=user/my_dataset dataset.streaming=true num_workers=4
```

> **macOS:** video streaming needs Homebrew ffmpeg on the dyld path. `import
> strands_robots` auto-fixes this (zero-touch); disable with
> `STRANDS_ROBOTS_NO_DYLD_SHIM=1`. See the README "Recording & streaming
> datasets" section.

## See also

- [Training](training/overview.md) - what to do with the data.
- [Steerable annotation](data/annotation.md) - add language conditioning columns to a recorded dataset.
- [LeRobot dataset docs](https://huggingface.co/docs/lerobot) - upstream spec.
