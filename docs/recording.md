---
description: DatasetRecorder — LeRobot v3 dataset writer used by both Simulation and HardwareRobot.
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

## Recording paths

| Method | Extra needed | Output |
|--------|-------------|--------|
| `start_recording` / `stop_recording` | `[lerobot]` | LeRobot v3 (parquet + MP4) |
| `start_cameras_recording` / `stop_cameras_recording` | `[sim-mujoco]` alone | Plain MP4, no parquet |

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
    # `joint_names=[...]` instead — the recorder builds the schema for you.
    camera_keys=["default"],
    joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    task="pick up the red cube",
    # root=None → ~/.strands_robots/datasets/
    # vcodec="libsvtav1", streaming_encoding=True, image_writer_threads=4
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
| `push_to_hub(tags=None, private=False)` | Upload to HuggingFace Hub |

## Read back

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="user/my_dataset", root="/tmp/my_dataset")
print(len(ds), ds[0].keys())
```

## See also

- [Training](training/overview.md) — what to do with the data.
- [LeRobot dataset docs](https://huggingface.co/docs/lerobot) — upstream spec.
