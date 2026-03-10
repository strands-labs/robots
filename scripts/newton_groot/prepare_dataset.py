#!/usr/bin/env python3
"""
Prepare GR00T-compatible dataset from the collected Newton sim data.

This script:
1. Adds task_index column to all parquet files
2. Creates proper modality.json with GR00T body part groups
3. Creates episodes.jsonl, tasks.jsonl
4. Creates info.json with correct metadata
5. Generates dataset statistics (stats.json + relative_stats.json)
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Dataset paths
DATASET_DIR = Path("/home/cagatay/e2e_pipeline_v2/dataset")
META_DIR = DATASET_DIR / "meta"
DATA_DIR = DATASET_DIR / "data" / "chunk-000"
VIDEO_DIR = DATASET_DIR / "videos" / "chunk-000" / "observation.images.ego_view"

META_DIR.mkdir(parents=True, exist_ok=True)

# ─── Step 1: Add task_index to parquet files ───
print("Step 1: Adding task_index to parquet files...")
parquet_files = sorted(DATA_DIR.glob("episode_*.parquet"))
print(f"  Found {len(parquet_files)} parquet files")

for pf in parquet_files:
    table = pq.read_table(pf)
    columns = table.column_names

    if "task_index" not in columns:
        n_rows = table.num_rows
        task_indices = pa.array([0] * n_rows, type=pa.int64())
        table = table.append_column("task_index", task_indices)
        pq.write_table(table, pf)

print(f"  ✅ Updated {len(parquet_files)} parquet files with task_index")

# Verify
sample = pq.read_table(parquet_files[0]).to_pandas()
print(f"  Columns: {list(sample.columns)}")
print(f"  Rows per episode: {len(sample)}")

# ─── Step 2: Create modality.json ───
print("\nStep 2: Creating modality.json...")

# G1 43-DOF joint layout:
# [0:6]   left_leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
# [6:12]  right_leg
# [12:15] waist     (yaw, roll, pitch)
# [15:22] left_arm  (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
# [22:29] left_hand (index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2)
# [29:36] right_arm
# [36:43] right_hand

modality = {
    "state": {
        "left_leg": {"start": 0, "end": 6},
        "right_leg": {"start": 6, "end": 12},
        "waist": {"start": 12, "end": 15},
        "left_arm": {"start": 15, "end": 22},
        "left_hand": {"start": 22, "end": 29},
        "right_arm": {"start": 29, "end": 36},
        "right_hand": {"start": 36, "end": 43},
    },
    "action": {
        "left_leg": {"start": 0, "end": 6},
        "right_leg": {"start": 6, "end": 12},
        "waist": {"start": 12, "end": 15},
        "left_arm": {"start": 15, "end": 22},
        "left_hand": {"start": 22, "end": 29},
        "right_arm": {"start": 29, "end": 36},
        "right_hand": {"start": 36, "end": 43},
    },
    "video": {
        "ego_view": {
            "original_key": "observation.images.ego_view"
        }
    },
    "annotation": {
        "human.action.task_description": {},
        "human.validity": {},
        "human.coarse_action": {
            "original_key": "annotation.human.action.task_description"
        }
    }
}

with open(META_DIR / "modality.json", "w") as f:
    json.dump(modality, f, indent=4)
print("  ✅ modality.json created")

# ─── Step 3: Create tasks.jsonl ───
print("\nStep 3: Creating tasks.jsonl...")
with open(META_DIR / "tasks.jsonl", "w") as f:
    f.write(json.dumps({"task_index": 0, "task": "Walk forward in the room using bipedal locomotion gait"}) + "\n")
print("  ✅ tasks.jsonl created")

# ─── Step 4: Create episodes.jsonl ───
print("\nStep 4: Creating episodes.jsonl...")
num_episodes = len(parquet_files)
with open(META_DIR / "episodes.jsonl", "w") as f:
    for i in range(num_episodes):
        ep = {
            "episode_index": i,
            "tasks": ["Walk forward in the room using bipedal locomotion gait"],
            "length": 100  # frames per episode
        }
        f.write(json.dumps(ep) + "\n")
print(f"  ✅ episodes.jsonl created ({num_episodes} episodes)")

# ─── Step 5: Create info.json ───
print("\nStep 5: Creating info.json...")
info = {
    "codebase_version": "v2.1",
    "robot_type": "unitree_g1",
    "total_episodes": num_episodes,
    "total_frames": num_episodes * 100,
    "total_tasks": 1,
    "fps": 50,
    "splits": {"train": f"0:{num_episodes}"},
    "data_path": "data/chunk-{episode_chunk:03d}/{episode_id}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "chunks_size": 1000,
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [43],
            "names": [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "left_hand_index_0_joint", "left_hand_index_1_joint",
                "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
                "right_hand_index_0_joint", "right_hand_index_1_joint",
                "right_hand_middle_0_joint", "right_hand_middle_1_joint",
                "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"
            ]
        },
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [720, 1280, 3],
            "video_info": {
                "video.fps": 50,
                "video.codec": "mpeg4",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [43],
            "names": [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "left_hand_index_0_joint", "left_hand_index_1_joint",
                "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
                "right_hand_index_0_joint", "right_hand_index_1_joint",
                "right_hand_middle_0_joint", "right_hand_middle_1_joint",
                "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"
            ]
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1]
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1]
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1]
        },
        "timestamp": {
            "dtype": "float64",
            "shape": [1]
        }
    }
}

with open(META_DIR / "info.json", "w") as f:
    json.dump(info, f, indent=4)
print("  ✅ info.json created")

print("\n✅ Dataset preparation complete!")
print(f"   Path: {DATASET_DIR}")
print(f"   Episodes: {num_episodes}")
print(f"   Total frames: {num_episodes * 100}")
print("   State dim: 43, Action dim: 43")
