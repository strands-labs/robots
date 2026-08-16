---
description: DatasetRecorder - LeRobot v3 dataset writer used by both Simulation and HardwareRobot.
---

# Recording & datasets

```python
from strands_robots import Robot

sim = Robot("so100")
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=50)
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", duration=10.0)
sim.stop_recording()
# LeRobot v3 dataset written to $HF_LEROBOT_HOME/user/my_dataset
```

`start_recording` requires `[lerobot]`. Without it, use `start_cameras_recording` for plain MP4.

## `fps` must equal the rollout's `control_frequency`

The recorder captures **one frame per control step and never decimates**, so the
rate frames arrive at is the rollout's `control_frequency` - and LeRobot derives
every timestamp from the dataset's declared `fps` positionally
(`timestamp = frame_index / fps`). A differing `fps` therefore cannot be
honored, only mislabelled, so it is refused:

```python
sim.start_recording(repo_id="user/my_dataset", task="t", fps=30)
sim.run_policy(robot_name="so100", policy_provider="mock")   # default 50.0 Hz
# -> "run_policy: the active recording declares 30 fps but this rollout captures
#     at control_frequency=50 Hz. [...] a 1.667x distortion of the episode
#     duration [...] Align the two rates: pass control_frequency=30 to
#     run_policy(), or record at the rollout's rate"
```

The refusal lands before any frame is written, so nothing is lost - pass either
rate. It matters beyond the label: that per-frame interval is the control period
a policy trains on, and `replay_episode` derives its per-frame physics budget
from the dataset rate, so a mislabelled episode also replays at the wrong speed.
To record at a lower rate than you control at, run the rollout at that rate -
there is no decimating recorder.

`PolicyRunner` is drivable directly, and the engine's check is not on that path,
so `PolicyRunner.run` / `PolicyRunner.evaluate` apply the same rule themselves.
They raise `ValueError` rather than returning an error dict, because a direct
caller has no tool envelope to read:

```python
runner = PolicyRunner(sim)
runner.run("so100", policy, control_frequency=50.0, on_frame=hook)
# -> ValueError: PolicyRunner.run: the active recording declares 30 fps but this
#    rollout captures at control_frequency=50 Hz. [...] pass control_frequency=30
#    to PolicyRunner.run()
```

The rule holds whichever call comes first. `start_policy` returns while its
rollout keeps running, so a recording can be opened against a rollout already in
flight - and on the defaults (`fps=30` against `control_frequency=50.0`) that
recorded a 1.667x mislabelled episode with every call reporting success.
`start_recording` refuses the same disagreement, before creating the dataset:

```python
sim.start_policy(robot_name="so100", policy_provider="mock")   # default 50.0 Hz
sim.start_recording(repo_id="user/my_dataset", task="t", fps=30)
# -> "start_recording: this recording would declare 30 fps but a rollout is
#     already running ('so100' at 50 Hz). [...] Align the two rates: record at
#     the rollout's rate (start_recording(fps=50)), or restart the rollout at the
#     recording's rate (stop_policy(robot_name='so100'), then
#     start_policy(..., control_frequency=30))"
```

Rollouts running at *different* rates are refused outright, even when `fps`
matches one of them: their frames interleave into one episode whose single
declared rate cannot describe both, so there is no `fps` to pass instead. Stop
all but one, or restart them at a common `control_frequency`.

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
    repo_id="user/my_dataset", task="pick up the cube", fps=50,
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

A robot's own cameras are offered under two names: the short name from its
MJCF (`wrist`) and the namespaced name the compiled scene uses
(`arm0/wrist`). Both address the same physical camera and record the same
frames, so pick one per dataset - listing both records the same view twice
under two columns. Every camera key carries the view of the camera it names:
a key the scene can no longer render (for example after `replace_scene_mjcf`,
which swaps the compiled scene but leaves the camera registry untouched) is
absent from the observation rather than filled in with the
overview, so a column is never quietly populated from the wrong camera.

`cameras=` is a list of **distinct** camera names, and every surface that accepts
one - `start_recording`, `render_all`, and the plain-MP4
`start_cameras_recording` / `start_cameras_recording_synchronous` - enforces that
shape up front. Two mistakes are refused rather than guessed at, because neither
can be honored as written:

- **A single name passed as a bare string.** `cameras="wrist"` is iterable per
  character, so it would be read as five cameras, one per letter. Wrap it in a
  list: `cameras=["wrist"]`.
- **A repeated name.** `cameras=["wrist", "wrist"]` cannot mean one camera and
  cannot mean two. In a dataset schema it collapses to a single column, so the
  recording declares fewer views than were asked for; in `render_all` it renders
  the same view twice; and in a plain-MP4 recording it opens a second encoder on
  the one output path, so the camera is rendered and appended twice per capture
  tick and the artifact ledger reports two files where one exists.

A `Mapping` is refused for the same reason - it is iterable over its keys, so its
values would be silently discarded. `cameras=None` keeps its "every camera"
meaning.


### Where the dataset is written (`root` / `overwrite`)

`root` is the on-disk directory for the dataset, used verbatim. When omitted,
the directory is derived from `repo_id`: an `owner/name` id records into
`$HF_LEROBOT_HOME/{repo_id}` (default `~/.cache/huggingface/lerobot`), and a
`repo_id` that is itself a path is taken as the directory. The home is read from
LeRobot's own `HF_LEROBOT_HOME` constant, so exporting it moves both the
recording and where `LeRobotDataset` later reads it back from -
`resolve_dataset_dir` is the one owner of those rules and every backend's
`start_recording` applies it.

Passing an existing **empty** directory - for example one returned by
`tempfile.mkdtemp()` - is accepted and recorded into:

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

`overwrite` and `push_to_hub` select a **posture**, so both must be booleans and
are checked before anything is created, resumed, wiped or published. Neither is
read by truthiness: every non-empty string is truthy, so `overwrite="false"` -
the spelling an operator reaches for when opting out - used to reach the wipe
branch and delete the dataset it was meant to append to, and
`push_to_hub="false"` used to publish it at `stop_recording`. Both now report the
flag instead:

```python
sim.start_recording(repo_id="user/my_dataset", root=root, fps=30, overwrite="false")
# -> error: "start_recording: overwrite must be a boolean, got 'false'. ..."
```

A resume inherits the existing dataset's schema, so `fps` must match the rate it
was created at - a resume cannot change it. Requesting a different rate is
refused, naming the on-disk value, rather than appending frames timestamped on
the dataset's timebase instead of the one they were captured at:

```python
sim.start_recording(repo_id="user/my_dataset", root=root, fps=60)  # dataset is 30 fps
# -> error: "dataset fps differs: on-disk=30 vs requested=60
#            (a resumed dataset keeps its on-disk rate; pass fps=30 to append at it)"
```

Pass `fps=30` to append at the dataset's rate, or `overwrite=True` to record a
fresh dataset at 60.

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
it. Only the keys above are accepted: any other key (`filename`, `resolution`,
...) is rejected with an error listing the accepted set, so a mistyped option
cannot silently produce a rollout with no video or a video at the wrong
resolution. For `n_episodes > 1` an `_ep{i}` suffix is inserted before the extension
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
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=50)
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
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=50)
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
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=50)
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

`--expected` and `--min-frames` are both non-negative integers, and each has a
meaningful `0`: `--expected 0` asks that a dataset be empty, and `--min-frames 0`
skips the per-episode length check (useful when the writer omits the `length`
column). Anything else - a negative threshold, a fraction, a non-finite value -
is reported as a problem and exits non-zero, rather than being applied. That
matters for `--min-frames` specifically: the length check runs only when the
threshold is above zero, so a value that is not a usable count would otherwise
switch the check off and certify a dataset holding a zero-length episode. The
same domain backs `verify_dataset_episodes(expected=...)`, so neither surface
accepts an episode count the other refuses.

`verify-dataset` always produces a report - it never crashes on the corruption
it exists to flag. A corrupt or foreign `meta/episodes` parquet, a non-v3
`video_path` template, or a truncated / unreadable MP4 is reported as a problem
string in the report (and a non-zero exit code), not surfaced as a raw
traceback.

Corruption confined to SOME episode parquet shards (the usual outcome of an
interrupted rsync or hub download) is localised rather than fatal: each
unreadable shard is named as a problem, the readable shards still supply
`total_episodes` / `frames_per_episode`, and the info.json, video, and
dead-column checks still run against them. Only a `meta/episodes` tree with no
readable shard at all reports zero episodes. The same holds for
`verify_dataset_episodes`, which additionally refuses to certify a dataset with
unreadable shards even when the readable count matches `expected` - the count is
then only a lower bound, reported in the `unreadable_files` diagnostics.

## Recording paths

| Method | Extra needed | Output |
|--------|-------------|--------|
| `start_recording` / `stop_recording` | `[lerobot]` | LeRobot v3 (parquet + MP4) |
| `save_episode` | `[lerobot]` | Close current rollout as one episode (call once per `run_policy` for N episodes) |
| `start_cameras_recording` / `stop_cameras_recording` | `[sim-mujoco]` alone | Plain MP4, no parquet |

`fps`, `width`, `height` and `max_frames_per_camera` on the plain-MP4 recorders
must be positive whole numbers - the same domain `run_policy(video={...})`,
`start_recording(fps=...)` and the shared encoder
`strands_robots.rendering.encode_clip` enforce. An unusable value (`fps=0`, a
negative frame cap) is a structured error naming the parameter rather than a
recording that reports success and writes no file. Omit `width`/`height` to use
each camera's configured resolution.

Encoding frames directly through `encode_clip(frames, path, fps=...)` follows the
same rule and raises `ValueError` for a rate it cannot honor - a fractional or
non-positive `fps` previously produced a clip at some other rate (the GIF writer
clamped it to 1 fps, ffmpeg substituted its own default) or no file at all, with
the output path still returned. `encode_clip` also raises `RuntimeError` when the
encoder wrote no clip despite accepting the frames, so a returned path always
names a clip that exists.

`encode_clip`'s `quality` is a finite number in `[1, 10]` (higher is better), and
that domain holds for both containers even though only the MP4 writer reads the
value - so one call does not become valid by changing the output extension. The
bound is the encoder's own: `0` is refused despite older documentation offering
`0-10`, and `True` is refused rather than acting as a silent quality of `1`, the
lowest offered. A NumPy real such as `np.int64(8)` is accepted and converted,
since the ffmpeg writer only recognises a plain `int`/`float`. The refusal is a
`ValueError` from `encode_clip` rather than the encoder's own `assert`, so it
does not disappear when Python runs with `-O`:

```python
from strands_robots.rendering import encode_clip

encode_clip(frames, "clip.mp4", fps=30, quality=9)   # a usable quality
encode_clip(frames, "clip.mp4", fps=30, quality=0)   # ValueError: quality must be between 1 and 10
```

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
    # root=None → $HF_LEROBOT_HOME/user/my_dataset
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

### Schema column names must be distinct

`camera_keys`, `joint_names` and `action_names` each declare the recorded
dataset's **column names**, so each must be a list of distinct, non-blank names.
`create()` refuses anything else before it touches the on-disk target, so a
refused `overwrite=True` call leaves an existing dataset intact:

```python
DatasetRecorder.create(repo_id="user/d", joint_names="gripper")     # ValueError
DatasetRecorder.create(repo_id="user/d", camera_keys=["front", "front"])  # ValueError
```

Both mistakes used to be accepted and only surfaced in the recorded data:

- A single name passed as a **bare string** is iterable per character, so
  `joint_names="gripper"` declared seven columns (`g`, `r`, `i`, `p`, `p`, `e`,
  `r`). `add_frame` reads each declared name out of the observation, and none of
  those names is in it, so every column recorded `0.0` for the whole episode -
  `create()`, `add_frame()`, `save_episode()` and `finalize()` all succeeded.
  Wrap a single name in a list: `joint_names=["gripper"]`.
- A **repeated** name collapses where it keys a dict and doubles where it indexes
  a position. `camera_keys=["front", "front"]` declared one camera column for the
  two the caller asked for; `joint_names=["j1", "j2", "j2"]` recorded `j2` twice
  and the joint the caller meant not at all.

`None` and `[]` still mean "not supplied" - the schema is then derived from
`robot_features` / `action_features`, or from `joint_names` for the action
columns.

### Camera frame shape must be a usable pixel count

`camera_dims` and the `video_width` / `video_height` pair are one quantity in two
spellings: the recorder declares each camera at
`camera_dims.get(camera, (video_height, video_width))`, so the mapping sets the
shape of the cameras it covers and the pair sets the shape of every other one.
Note the order - `camera_dims` is `(height, width)`, the reverse of the pair.

It is a **declaration, not a resize** - the recorder rescales nothing - so
whatever is given goes straight into the LeRobot feature as `(3, height, width)`.
`create()` refuses a shape it cannot honor, on the same shared domain and in the
same place as the column names above:

```python
DatasetRecorder.create(repo_id="user/d", camera_keys=["image"],
                       camera_dims={"imagee": (240, 320)})   # ValueError: not a declared camera
DatasetRecorder.create(repo_id="user/d", camera_keys=["image"],
                       camera_dims={"image": (240,)})        # ValueError: not a (height, width) pair
DatasetRecorder.create(repo_id="user/d", camera_keys=["image"],
                       video_width=0)                        # ValueError: must be a positive integer
```

Three mistakes used to be accepted, and none was reported near the parameter that
caused it:

- A key `camera_keys` does not declare is **never looked up**, so the camera it
  was meant for silently took the global pair instead. A camera streaming
  240x320 declared as `camera_dims={"imagee": (240, 320)}` against
  `camera_keys=["image"]` was declared `(3, 480, 640)` from the defaults -
  nothing logged, dataset created, and the mismatch surfacing later against
  `add_frame`.
- A component that is **not a positive integer** was written in as given, so the
  schema declared `(3, 480, nan)` or `(3, 480, '640')` and no frame could match
  it. A pixel count is written into `meta/info.json`, so it has to be a true
  `int`: an integral float would be declared as `480.0`, and a NumPy integer is
  not JSON-serializable.
- A value that is **not a two-element sequence** unpacked as a bare `TypeError`,
  and a non-mapping `camera_dims` as a bare `AttributeError` from the lookup.

`camera_dims=None` and `{}` still mean "not supplied" - every camera then takes
the global pair. Passing the shape as a list rather than a tuple is accepted.

### The recording rate must be a usable frame count

`fps` is the rate the dataset is **declared** at - it is written into
`meta/info.json` and every timestamp is derived from it positionally - so
`create()` holds it to the same positive-whole-number domain every backend's
`start_recording(fps=...)` already applies to the rate it forwards here:

```python
DatasetRecorder.create(repo_id="user/d", fps=2.7)    # ValueError: fps must be a positive whole number
DatasetRecorder.create(repo_id="user/d", fps=float("nan"))   # same refusal
DatasetRecorder.create(repo_id="user/d", fps=True)   # refused, not read as 1 fps
```

LeRobot rejects only `fps <= 0`, so everything above used to be accepted by the
direct API and cost the caller the episode without reporting anything: a
fractional `2.7`, a `nan` or an `inf` created the dataset and then saved **zero
frames**, with `create`, `add_frame`, `save_episode` and `finalize` all returning
normally. `fps=True` recorded a 1 fps dataset (an `int` subclass acting as a 1),
and `fps="30"` dead-ended in a bare `TypeError` naming neither the parameter nor
the method.

The rule is the facades' rule by construction - one shared domain, reached from
both - so a rate `start_recording` accepts cannot be refused deeper. A rate that
disagrees with a rollout's `control_frequency` is a separate check that only the
facades can make (see [`fps` must equal the rollout's
`control_frequency`](#fps-must-equal-the-rollouts-control_frequency)); `create()`
judges the rate on its own terms.

### Re-recording into an existing `repo_id`

`DatasetRecorder.create()` builds a **fresh** dataset. If the resolved dataset
directory already exists, LeRobot's `create()` would raise a bare
`FileExistsError` (its `mkdir` uses `exist_ok=False`). `create()` resolves this
up front, matching the `start_recording` facade:

- `overwrite=True` wipes the existing directory and creates a fresh dataset.
- `overwrite=False` (default) on an existing dataset (a dir containing `meta/`)
  raises a clear `FileExistsError` naming `overwrite=True` (fresh) and
  `resume()` (append) - not a cryptic LeRobot error, and never a silent clobber.
- An existing **empty** directory (e.g. from `tempfile.mkdtemp()`) is cleared so
  `create()` does not dead-end on its own existence guard.
- A non-empty **non-dataset** directory raises `ValueError` rather than deleting
  unrelated files.

```python
# Re-run a capture script into the same repo_id, replacing the old dataset:
recorder = DatasetRecorder.create(
    repo_id="user/my_dataset", fps=30, joint_names=[...], overwrite=True,
)
```

Use `resume()` (not `create(overwrite=True)`) when you want to **append**
episodes to the existing dataset instead of replacing it.

### A frame the recorder cannot write fails the rollout

`DatasetRecorder` is fail-fast by default (`strict=True`): if the underlying
`LeRobotDataset` write fails, it raises
`strands_robots.dataset_recorder.RecordingFrameError` instead of continuing. When
the recorder is driven by `run_policy`, that ends the rollout with
`status="error"` naming the frame the recording stopped being complete at:

```
Policy failed: dataset add_frame failed after 7 frame(s) written; the recording
is incomplete from this frame on (strict=True, so it is not dropped silently):
<the underlying error>
```

Continuing past a lost frame is not a smaller failure. Timestamps are derived
positionally from the declared `fps`, so the surviving frames are re-stamped into
a shorter span than they were captured over - a rollout that loses every other
frame at 50 Hz produces an episode labelled at 2x speed, with no gap to detect.

Construct the recorder with `strict=False` to trade that for best-effort
recording: a failed write is counted in `dropped_frame_count`, warned about at
`WARNING` (on the 1st, 2nd, 4th, 8th ... failure so a 50 Hz loop cannot flood the
log), and the rollout continues.

## Instance methods

| Method | What |
|--------|------|
| `add_frame(observation, action, task=None, camera_keys=None)` | Append one timestep |
| `save_episode()` | Flush buffer as a new episode |
| `clear_episode_buffer()` | Discard current episode |
| `finalize()` | Write metadata, stats, close writers |
| `push_to_hub(tags=None, private=False)` | Upload to a versioned HF dataset repo. `private` selects the published repo's visibility, so it must be a boolean — a truthy spelling of off such as `"false"` would otherwise select the opposite posture |
| `sync_to_bucket(bucket, run_id=None, private=True)` | Sync to a mutable HF Storage Bucket (`hf://buckets/...`) — Xet-deduped collection target; needs the `hf` CLI. `bucket` (`name` or `org/name`) and `run_id` (single segment) are allowlist-validated (`[A-Za-z0-9._-]`, no traversal) before the sync, and `create` / `private` / `delete` must each be a boolean — `delete` mirror-deletes remote files absent locally, so a truthy `"false"` must not select it |

## Read back

Fully materialized (downloads everything):

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="user/my_dataset", root="/tmp/my_dataset")
print(len(ds), ds[0].keys())
```

## Replay an episode

`sim.replay_episode(repo_id, robot_name=..., episode=0, root=None, speed=1.0)`
plays a recorded episode back through the sim: each recorded frame is one
control step, applied via `send_action` and integrated for a full control period
derived from the dataset fps, so a position-servo robot reproduces the recorded
trajectory. `speed` scales only the wall-clock playback rate.

Each recorded action index is bound to an action key. By default those are
`robot_action_keys(robot_name)` — the robot's **actuator** keys, which is the
ordering the recorder writes the `action` column in. Pass `action_key_map` only
when the dataset's action ordering differs:

```python
sim.replay_episode(
    "user/my_dataset",
    robot_name="so101",
    root="/tmp/my_dataset",
    action_key_map=["1", "2", "3", "4", "5", "6"],  # one key per action index
)
```

`action_key_map` must be a non-empty list/tuple of unique strings whose length
equals the recorded action vector's width. A bare string (consumed one key per
character), a non-string entry, a duplicate key, or a width mismatch is rejected
with an actionable error before the dataset is fetched — never truncated to fit.

A `"success"` status means **every** frame reached the actuators. If a recorded
action cannot be applied — e.g. the mapped keys resolve to no actuator on this
robot — the replay aborts at that frame and returns `status="error"` with the
frame index, how many frames were applied, and the unresolved keys:

```python
result = sim.replay_episode("user/my_dataset", robot_name="so101", action_key_map=["wrong"] * 6)
result["status"]                                  # "error"
result["content"][1]["json"]["unresolved_keys"]   # ['wrong', ...]
result["content"][1]["json"]["frames_applied"]    # 0
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
edge devices without a torchcodec wheel; requires `delta_timestamps` with at
least one non-video key, otherwise `open()` raises `ValueError` rather than
silently streaming video anyway).

The four numeric kwargs (`tolerance_s`, `buffer_size`, `max_num_shards`,
`seed`) are checked before the lerobot import: `StreamingLeRobotDataset`
validates only `repo_type` and stores the rest verbatim, so every consumer of
them is downstream of a call that already returned. `open()` raises `ValueError`
naming the parameter instead - a shard count of `0` used to open successfully
and then stream **zero frames**, a `buffer_size` of `0` raised out of NumPy
part-way through iteration, and a `tolerance_s` of `inf` switched off the
delta-grid check below. `tolerance_s=0` is accepted and means "require an exact
grid match"; `seed=0` is accepted and is simply a seed.

One kwarg is **not** tolerant-forwarded because its absence changes semantics:
`repo_type="bucket"` requires `lerobot>=0.6.1`, which the `[lerobot]` extra
floors — so a resolver-conformant install always has it. On an environment
carrying an older lerobot, `open()` raises `RuntimeError` naming the upgrade
instead of silently streaming from the versioned dataset namespace (a different
storage system).

For **training**, the upstream trainer uses the same engine:

```bash
python -m lerobot.scripts.lerobot_train --policy.type=act \
  --dataset.repo_id=user/my_dataset --dataset.streaming=true --num_workers=4
```

> **macOS:** video streaming needs Homebrew ffmpeg on the dyld path. `import
> strands_robots` auto-fixes this (zero-touch); disable with
> `STRANDS_ROBOTS_NO_DYLD_SHIM=1`. See the README "Recording & streaming
> datasets" section.

## See also

- [Training](training/overview.md) - what to do with the data.
- [Steerable annotation](data/annotation.md) - add language conditioning columns to a recorded dataset.
- [LeRobot dataset docs](https://huggingface.co/docs/lerobot) - upstream spec.
