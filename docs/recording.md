---
description: Pass record= to any run to write a LeRobot v3 dataset, or record from the dashboard's Record tab.
---

# Recording a dataset

Pass `record=` to `run()` to save the run as a LeRobot v3 dataset.

```python
from strands_robots import Robot

arm = Robot("so101")
arm.sim.camera("front", size=(320, 240), attach=arm)
for _ in range(3):
    arm.run(my_policy, seconds=3, record="datasets/so101-wave")
```

`my_policy` stands for your policy: any function from an observation to targets, such as
`lambda obs: {"gripper": 1.0}`. Each run adds one episode to the same directory.
LeRobot's own writer makes the files, so any LeRobot training code can read them.

<img class="sr-media" src="../assets/media/recording.gif" alt="frames read back from a recorded dataset">

| Case | Result |
|---|---|
| The run raises | The episode is dropped. |
| The run is stopped or e-stopped | The frames so far are kept. |
| `record="hf://you/so101-wave"` | Uses a Hub name. Nothing uploads unless you pass `push=True`. |

A dataset directory must stay where it is while a take writes it. So `record=` and the dashboard refuse a dataset
whose path goes through a symlink, or through a directory that another account owns or can write to (unless that
directory has the sticky bit, like `/tmp`). The error names the directory, its owner and its mode. Record into a
directory only you can change (`chmod go-w`). Missing parent directories are created for you. When a take that
created its dataset is dropped, only that directory is deleted. If its path now leads somewhere else, it is kept.

## Frame timing

LeRobot's `timestamp` column is always `frame_index / fps`, and its video frames are found by it, so a slow tick
can't show there. Each frame also stores `time.measured`: when that frame was read on the robot's clock, in seconds
from the episode's first frame. On a real robot a stalled tick shows as a gap longer than `1 / fps` after it, and a
skipped tick as a gap of about `2 / fps`. In a simulation that isn't paced in real time, the clock is sim time, so the
gaps are exactly `1 / fps`. The key starts with neither `observation` nor `action`, so no LeRobot policy takes it as
an input.

<!-- test: skip, reads a dataset recorded above -->
```python
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/so101-wave", root="datasets/so101-wave")
t = np.asarray(ds.hf_dataset["time.measured"], dtype=float).ravel()
gaps = np.diff(t)
late = np.flatnonzero(gaps > 1.5 / ds.fps)  # frames read after a stall (compare within one episode)
print(f"worst gap {gaps.max() * 1e3:.1f} ms; {late.size} late frames")
```

A dataset recorded before frames carried `time.measured` keeps taking episodes, written without it.

## Record from the dashboard

The dashboard's **Record** tab does the same without code. See [Dashboard](dashboard.md) to start one. The tab
writes only inside `~/strands-robots/recordings` (set `STRANDS_ROBOTS_RECORDINGS` to move it). A target is a
name in that directory, a path inside it, or `hf://owner/name`.

1. Tick the robots and cameras to record.
2. Enter the task, a target directory and the fps.
3. Press **rec**, then move the robot any way you like: sliders, a policy, the chat or a script.
4. Press **save episode** to keep the take, **discard** to drop it, or **stop** to finish.

Each robot gets its own dataset at `<target>/<robot id>`. Scripts can call the same routes with the session token
(see [Signing in](dashboard.md#signing-in)). Every `POST` takes a JSON body, `{}` when it has no fields; start with
`{"robots": ["so101"], "target": "so101-wave", "task": "wave", "fps": 30}`:

| Route | Does |
|---|---|
| `GET /api/record/options` | Lists robots, cameras and a default target |
| `POST /api/record/start` | Starts the first episode |
| `POST /api/record/episode/save` | Keeps the episode and starts the next |
| `POST /api/record/episode/discard` | Drops the episode and starts the next |
| `POST /api/record/stop` | Ends the recording |
| `GET /api/record/status` | Shows frames, fps and saved episodes |

Read a dataset back with LeRobot:

<!-- test: skip, reads a dataset recorded from the page -->
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/so101-wave", root="datasets/so101-wave")
print(ds.meta.total_episodes, ds.fps)
```
