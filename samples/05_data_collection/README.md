# 🎓 Sample 05: Data Collection & Recording — Building Datasets

**Level:** 2 (Middle School) | **Time:** 20 minutes | **Hardware:** CPU (real hardware optional)

---

## What You'll Learn

Robots learn by watching — either watching a human demonstrate a task, or watching
a policy attempt it. This sample teaches you how to **record** those demonstrations
into a format that training algorithms understand.

By the end, you'll be able to:

1. Record robot trajectories in simulation
2. Save recordings as LeRobot-compatible datasets
3. Inspect and understand the dataset format
4. Use the `DatasetRecorder` for multi-episode capture
5. Push datasets to HuggingFace Hub

## Prerequisites

- Completed Samples 01–04
- `pip install strands-robots[sim]`
- Optional: `huggingface-cli login` for uploading

---

## 🧠 The Data Pipeline

```
Teleoperation / Policy Rollout
        │
        ▼
  Observations + Actions     ← Every control step (e.g., 30 Hz)
        │
        ▼
   LeRobotDataset            ← Parquet for numbers, MP4 for video
        │
        ▼
   HuggingFace Hub           ← Share with the world
```

Every time a robot moves, it produces two things:
- **Observations**: What the robot sees/feels (joint positions, camera images)
- **Actions**: What the robot does (target joint positions)

Recording captures both at each timestep, creating a dataset that training
algorithms (ACT, Pi0, SmolVLA) can learn from.

## 📁 LeRobot Dataset Format

Datasets are stored in **LeRobot v3 format** — an open standard from HuggingFace:

```
my_dataset/
├── meta/
│   ├── info.json              # Metadata: fps, shapes, robot type
│   └── episodes.jsonl         # Episode boundaries and tasks
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # Joint positions + actions
│       ├── episode_000001.parquet
│       └── ...
└── videos/                    # Camera recordings (if any)
    └── chunk-000/
        ├── observation.images.cam/
        │   ├── episode_000000.mp4
        │   └── ...
```

- **Parquet files** contain numerical data (joint angles, velocities, actions)
- **MP4 files** contain camera frames (for Vision-Language-Action models)
- **info.json** tells the training code what shapes and types to expect

## 💻 Scripts

### 1. `record_episodes.py`
Record 5 episodes in simulation with a mock policy and save as a dataset.
This is the simplest path from "robot moving" to "training data."

### 2. `inspect_dataset.py`
Load a recorded dataset, print shapes, and show trajectory statistics.
Understanding your data is as important as collecting it.

### 3. `record_with_recorder.py`
Use the `DatasetRecorder` class for finer control over multi-episode capture.
This is what `simulation.py` uses internally.

### 4. `push_to_hub.py`
Upload a dataset to HuggingFace Hub (includes dry-run mode).

---

## 🔬 Exercises

1. **Compare policies**: Record 5 episodes with `mock` policy, then 5 with random
   actions (just `np.random.randn(6) * 0.1`). Look at the trajectory JSON — how
   are they different?

2. **Inspect Parquet**: Open a `.parquet` file with `pandas.read_parquet()`.
   What are the column names? What data types?

3. **Plot joint trajectories**: Use matplotlib to plot one joint's position over
   time for 3 different episodes. Do they look similar?

4. **Episode length**: Record episodes of different durations (2s, 5s, 10s).
   How does frame count relate to FPS × duration?

---

## 📚 SDK Surface Covered

| Class / Function | Module | Purpose |
|-----------------|--------|---------|
| `Robot` (sim) | `strands_robots.factory` | Create simulation, run policies |
| `start_recording()` | `strands_robots.simulation` | Begin dataset capture |
| `stop_recording()` | `strands_robots.simulation` | Save episode + finalize |
| `record_video()` | `strands_robots.simulation` | Policy → MP4 in one call |
| `DatasetRecorder` | `strands_robots.dataset_recorder` | Low-level multi-episode recorder |
| `RecordSession` | `strands_robots.record` | Full recording session (real HW) |
| `RecordMode` | `strands_robots.record` | TELEOP / POLICY / IDLE |
| `lerobot_dataset` tool | `strands_robots.tools` | Agent tool for dataset ops |

---

## 🔗 Connections

- **Previous:** [Sample 04 — Gymnasium Training](../04_gymnasium_training/)
- **Next:** [Sample 06 — Ray-Traced Training](../06_ray_traced_training/)
- **Forward:** Sample 07 uses recorded data to fine-tune GR00T N1.6
- **Related:** [Tutorial: Recording Data](../../docs/tutorial/06-recording.md)
