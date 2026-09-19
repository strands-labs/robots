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

`start_recording` requires `[lerobot]`; without it, `start_cameras_recording` writes plain MP4. The
writer itself is [DatasetRecorder](data/dataset-recorder.md), grading a finished dataset is
[Verify a dataset](data/verifying-datasets.md), and reading one back is
[Read back & replay](data/reading-back.md).

## What fills an episode

| Source | Frames it records |
|---|---|
| `run_policy` / `start_policy` | one per control step, on every backend |
| `step` | one per `1/fps` of sim time (MuJoCo) |
| `move_to`, `set_gripper`, other primitives | none - drive them, then `step` to capture |

While a recording is open, `step` turns scripted motion into a demonstration:

```python
sim.start_recording(repo_id="user/three_poses", task="three poses", fps=10)
for pose in (0.3, -0.3, 0.0):
    sim.set_joint_positions(positions={"1": pose}, robot_name="so101", hold=True)
    sim.step(n_steps=250)          # 0.5 s -> "recorded 5 frames" (6 on the first call: the opening frame too)
sim.stop_recording()               # 16 frames, 1 episode
```

Each frame carries the same state and cameras a `run_policy` frame does; its *action* is the
position-servo target in force at that instant (`data.ctrl`, keyed like `robot_action_keys`) and its
`task` is the session's. A `step` shorter than one frame period records nothing and says when the
next frame is due.

## `fps` must equal the rollout's `control_frequency`

The recorder captures **one frame per control step and never decimates**, and LeRobot derives every
timestamp from the declared `fps` positionally (`timestamp = frame_index / fps`). A differing `fps`
cannot be honored, only mislabelled, so it is refused before a frame is written - to record at a
lower rate, control at that rate.

| Ordering | Neither rate passed | Rates passed and disagreeing |
|---|---|---|
| recording open, then rollout | the rollout adopts the recording's `fps` | `run_policy` refuses |
| rollout in flight, then recording | the recording opens at the rollout's rate | `start_recording` refuses, before the dataset is created |
| `run_policy` tool (`dataset_fps` + `control_frequency`) | settled up front, before anything opens | refused with the dataset at `dataset_root` untouched |

```python
sim.start_recording(repo_id="user/my_dataset", task="t", fps=30)
sim.run_policy(robot_name="so100", policy_provider="mock", control_frequency=50.0)
# -> "run_policy: the active recording declares 30 fps but this rollout captures
#     at control_frequency=50 Hz. [...] a 1.667x distortion of the episode
#     duration [...] Align the two rates: pass control_frequency=30 to
#     run_policy(), or record at the rollout's rate"
```

A rate the caller passed is a decision, so a mismatch is refused; a rate nobody passed is a default,
and the reply names the one adopted. `PolicyRunner.run` / `PolicyRunner.evaluate` raise `ValueError`
instead, a direct caller having no envelope to read. Rollouts at *different* rates are refused
outright - one declared rate cannot describe both. `dataset_fps` is ignored when `dataset_root` is
omitted.

## Selecting which cameras to record

Every scene camera is recorded by default, including the implicit `default` free camera. Pass
`cameras=` to record exactly the views a policy declares (SmolVLA expects
`observation.images.camera1/camera2/camera3`, and a stray `default` column bloats every episode):

```python
sim.add_camera(name="camera1", ...)
sim.add_camera(name="camera2", ...)
sim.add_camera(name="camera3", ...)
sim.start_recording(
    repo_id="user/my_dataset", task="pick up the cube", fps=50,
    cameras=["camera1", "camera2", "camera3"],   # drops the implicit 'default'
)
```

The dataset then declares only those image features. Names may be raw MuJoCo form
(`arm0/wrist_cam`) or schema-safe form (`arm0__wrist_cam`); an unknown one is refused with the
available list, before any dataset is created, resumed or wiped. Omitting `cameras=` records
everything and warns once when the implicit `default` is swept in. A robot's own cameras are offered
under both the short MJCF name (`wrist`) and the scene's namespaced name (`arm0/wrist`) - one
physical camera, so listing both records one view twice. A key the scene can no longer render (after
`replace_scene_mjcf`) is absent from the observation rather than filled from the overview camera.

A camera name is also the key its frames travel under - the mesh topic, the S3 object key, the
`observation.images.<name>` column - so it is a bare token of letters, digits, `_` or `-` opening on
a letter or digit, optionally scoped to one robot as `<robot>/<camera>`: `wrist`, `front_cam`,
`cam-2`, `arm0/wrist_cam`. One scope level, the namespace `add_robot` gives what it spawns. Anything
else (`a b`, `wrist.rgb`, `*`, `..`, `sub/../etc`, `a//b`) is refused at `add_camera` rather than
registered and then misrouted.

| Refused shape | Why it cannot be honored |
|---|---|
| two cameras collapsing to one column (`arm0/wrist` and `arm0__wrist`) | `/` -> `__` is not injective: two cameras, one column. Rename one; the plain-MP4 sinks owe the same collapse, writing `clip__arm0__wrist.mp4` |
| a bare string (`cameras="wrist"`) or a `Mapping` | a string is iterable per character, a mapping over its keys. Pass `cameras=["wrist"]` |
| a repeated name (`cameras=["wrist", "wrist"]`) | one column in a schema, two renders in `render_all`, two encoders on one MP4 path |

`cameras=None` keeps its "every camera" meaning, and every surface taking the list -
`start_recording`, `render_all`, `start_cameras_recording` and its synchronous twin - enforces the
shape up front.

### Where the dataset is written (`root` / `overwrite`)

`root` is the on-disk directory, used verbatim. Omit it and `resolve_dataset_dir` - one owner,
applied by every backend - derives it from `repo_id`:

| `repo_id` | Directory | Read back |
|---|---|---|
| `owner/name` | `$HF_LEROBOT_HOME/{repo_id}` (default `~/.cache/huggingface/lerobot`) | LeRobot's revision-safe snapshot cache, which is that directory |
| a path (`sim_recording`, `./data/run1`) | the path itself | the same directory, no `root` restated |

The resolved directory reaches `LeRobotDataset.create` as an explicit `root`, so the directory
inspected (and, under `overwrite=True`, deleted) is the one written into; `resume` forwards it too,
so the `repo_id` that created a dataset reopens it.

| State of `root` | `start_recording` |
|---|---|
| absent, or an existing empty directory (`tempfile.mkdtemp()`) | records into it |
| holds a LeRobotDataset (`meta/`) | **resumes**, naming what is already on disk; `overwrite=True` wipes and recreates instead |
| exists, non-empty, not a dataset | error, left untouched - pass `overwrite=True` or choose another `root` |

```python
sim.start_recording(repo_id="user/my_dataset", root=root, fps=30)
# -> "Recording to LeRobotDataset: user/my_dataset
#     Resuming the existing dataset (1 episode(s), 19 frames); this session's
#     episodes are appended. Pass overwrite=True to record from scratch instead."
sim.stop_recording()
# -> "user/my_dataset -- 37 frames, 2 episode(s) (+18 frames, +1 episode(s) this session)"
```

`stop_recording` measures **that session**: a resumed session that captured no frames is an error
naming the dataset it left unchanged. A resume inherits the schema, so `fps` must match the rate the
dataset was created at (`dataset fps differs: on-disk=30 vs requested=60`). `overwrite` and
`push_to_hub` select a posture, so each must be a boolean rather than read by truthiness, and every
refusal happens before the wipe - a refused call leaves the dataset that was there intact.

The `run_policy` tool drives the whole `start_recording` -> rollout -> `stop_recording` cycle and
takes the same subset as `dataset_cameras=`, plus one `video={"path": ..., "fps": ..., "camera": ...,
"width": ..., "height": ...}` config per episode. `video["path"]` enables it, any other key is
rejected with the accepted set listed, an `_ep{i}` suffix keeps multi-episode clips apart, and
`video_paths` reports the MP4s that landed. Full argument list: [Tool
reference](reference/tools.md#run_policy).

![run_policy tool rollout video (SmolVLA on a simulated SO-101, MuJoCo headless)](assets/run_policy_video_demo.gif)

*A SmolVLA-on-SO-101 rollout recorded through the `run_policy` tool's `video=` config (MuJoCo
headless, `MUJOCO_GL=egl`).*

## Multi-episode recording

A session is one dataset. `run_policy(n_episodes=N)` runs N rollouts back-to-back, flushing an
episode boundary after each and resetting between them:

```python
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=50)
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", n_steps=60, n_episodes=20)
sim.stop_recording()
# -> 20 episodes, each with its own episode_index / length / from_index / to_index
```

`n_steps` (or `duration`) is the per-episode horizon, `reset_between=False` chains episodes from the
previous end state, a `seed` is offset per episode (`seed + i`), and the aggregate carries
`n_episodes_completed`, `episodes_saved`, `total_steps` and a per-episode list.

Driving the loop yourself, the boundary is `save_episode()` - idempotent on an empty buffer - or
`reset()`, which flushes buffered frames as their own episode before re-initializing the world:

```python
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=50)
for _ in range(20):
    sim.reset()
    sim.run_policy(robot_name="so100", instruction="pick up the cube",
                   policy_provider="mock", n_steps=60)
    sim.save_episode()        # flush this rollout as one episode
sim.stop_recording()          # flushes any trailing rollout automatically
```

Without `n_episodes`, a `save_episode()` or a `reset()`, all 20 rollouts append to one buffer that
`stop_recording` flushes as a single `episode_index=0`; `clear_episode_buffer()` discards a partial
rollout instead. LeRobot computes `stats.json` per episode, so per-rollout boundaries keep statistics
correct across the `reset()` teleport. Every backend cuts the boundary through the same rule, with no
exception: Isaac's `reset()` refuses `env_ids` rather than resetting everything, so every reset is a
boundary.

## See also

- [DatasetRecorder API](data/dataset-recorder.md) - the writer, MP4 clips, codecs, publishing.
- [Verify a dataset](data/verifying-datasets.md) - prove the episodes, the pixels, the columns.
- [Read back & replay](data/reading-back.md) - replay, stream, train.
- [Steerable annotation](data/annotation.md) - language conditioning columns.
- [LeRobot dataset docs](https://huggingface.co/docs/lerobot) - upstream spec.
