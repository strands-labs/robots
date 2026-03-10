#!/usr/bin/env python3
"""
Sample 05 — Inspect a Recorded Dataset

Load a recorded dataset (or a pre-built one from HuggingFace),
print its structure, and visualize trajectory statistics.

This teaches you to understand WHAT your data looks like before
feeding it to a training algorithm.

Usage:
    python inspect_dataset.py

    # Or with a specific local trajectory file:
    python inspect_dataset.py trajectory.json

Requirements:
    pip install strands-robots[sim]
    Optional: pip install pandas matplotlib  (for deeper inspection)
"""

import json
import os
import sys
from pathlib import Path


def inspect_trajectory_json(path: str):
    """Inspect a raw trajectory JSON file from simulation recording."""
    print(f"\n📂 Inspecting trajectory: {path}")
    print("-" * 50)

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print("  ⚠️  Empty or invalid trajectory file")
        return

    print(f"  Total frames: {len(data)}")

    # Examine first frame structure
    first = data[0]
    print(f"\n  Frame keys: {sorted(first.keys())}")

    # Time info
    if "time" in first:
        last = data[-1]
        duration = last["time"] - first["time"]
        fps_actual = len(data) / duration if duration > 0 else 0
        print(f"\n  ⏱️  Duration: {duration:.2f}s")
        print(f"  📊 Actual FPS: {fps_actual:.1f}")

    # Joint info
    if "joints" in first:
        joints = first["joints"]
        print(f"\n  🦾 Joints ({len(joints)}):")
        for name, value in joints.items():
            print(f"     {name}: {value:.4f} rad")

        # Compute range of motion across trajectory
        print("\n  📈 Joint ranges across trajectory:")
        for name in joints.keys():
            values = [frame["joints"][name] for frame in data if "joints" in frame]
            min_v = min(values)
            max_v = max(values)
            range_v = max_v - min_v
            print(f"     {name}: [{min_v:.3f}, {max_v:.3f}] (range: {range_v:.3f} rad)")

    # Action info
    if "actions" in first:
        actions = first["actions"]
        if isinstance(actions, dict):
            print(f"\n  🎮 Action dimensions: {len(actions)}")
        elif isinstance(actions, list):
            print(f"\n  🎮 Action dimensions: {len(actions)}")

    # Velocity info
    if "velocities" in first:
        vels = first["velocities"]
        print(f"\n  ⚡ Velocity channels: {len(vels)}")


def inspect_lerobot_dataset_dir(path: str):
    """Inspect a LeRobot dataset directory structure."""
    dataset_dir = Path(path)
    print(f"\n📂 Inspecting LeRobot dataset: {dataset_dir}")
    print("-" * 50)

    # Check meta/info.json
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        print("\n  📋 Dataset Info (meta/info.json):")
        for key in ["codebase_version", "robot_type", "fps", "total_episodes",
                     "total_frames", "total_tasks", "total_videos", "total_chunks"]:
            if key in info:
                print(f"     {key}: {info[key]}")

        if "features" in info:
            print(f"\n  🔧 Features ({len(info['features'])}):")
            for feat_name, feat_info in info["features"].items():
                shape = feat_info.get("shape", "?")
                dtype = feat_info.get("dtype", "?")
                print(f"     {feat_name}: shape={shape}, dtype={dtype}")
    else:
        print("  ⚠️  No meta/info.json found")

    # Check episodes
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        episodes = []
        with open(episodes_path) as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
        print(f"\n  📼 Episodes: {len(episodes)}")
        for ep in episodes[:5]:  # Show first 5
            idx = ep.get("episode_index", "?")
            task = ep.get("task", "?")
            length = ep.get("length", "?")
            print(f"     Episode {idx}: {length} frames — '{task}'")
        if len(episodes) > 5:
            print(f"     ... and {len(episodes) - 5} more")

    # Check parquet files
    data_dir = dataset_dir / "data"
    if data_dir.exists():
        parquet_files = list(data_dir.rglob("*.parquet"))
        print(f"\n  📊 Parquet files: {len(parquet_files)}")

        # Try reading with pandas if available
        if parquet_files:
            try:
                import pandas as pd

                df = pd.read_parquet(parquet_files[0])
                print(f"\n  🔍 First parquet ({parquet_files[0].name}):")
                print(f"     Rows: {len(df)}")
                print(f"     Columns: {list(df.columns)}")
                print("     Dtypes:")
                for col in df.columns:
                    print(f"       {col}: {df[col].dtype}")
                print("\n     First 3 rows:")
                print(df.head(3).to_string(index=False))
            except ImportError:
                print("     (install pandas for deeper inspection: pip install pandas)")

    # Check video files
    videos_dir = dataset_dir / "videos"
    if videos_dir.exists():
        video_files = list(videos_dir.rglob("*.mp4"))
        print(f"\n  🎬 Video files: {len(video_files)}")
        for vf in video_files[:5]:
            size_kb = vf.stat().st_size / 1024
            print(f"     {vf.relative_to(dataset_dir)}: {size_kb:.0f} KB")


def create_sample_trajectory():
    """Create a sample trajectory if none exists (for self-contained demo)."""
    import math

    print("📝 Creating sample trajectory for inspection...")

    fps = 30
    duration = 3.0
    num_frames = int(fps * duration)
    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    trajectory = []
    for i in range(num_frames):
        t = i / fps
        joints = {}
        for j, name in enumerate(joint_names):
            # Sinusoidal motion with different frequencies per joint
            freq = 0.5 + j * 0.2
            amplitude = 0.3 + j * 0.05
            joints[name] = amplitude * math.sin(2 * math.pi * freq * t)

        trajectory.append({
            "time": t,
            "joints": joints,
            "velocities": {
                name: 2 * math.pi * (0.5 + j * 0.2) * (0.3 + j * 0.05)
                * math.cos(2 * math.pi * (0.5 + j * 0.2) * t)
                for j, name in enumerate(joint_names)
            },
            "actions": {name: joints[name] * 0.9 for name in joint_names},
        })

    path = "sample_trajectory.json"
    with open(path, "w") as f:
        json.dump(trajectory, f, indent=2)

    print(f"  ✅ Saved {num_frames} frames to {path}")
    return path


def plot_trajectories(path: str):
    """Plot joint trajectories from a JSON file (requires matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (install matplotlib for plots: pip install matplotlib)")
        return

    with open(path) as f:
        data = json.load(f)

    if not data or "joints" not in data[0]:
        return

    times = [frame["time"] for frame in data]
    joint_names = list(data[0]["joints"].keys())

    fig, axes = plt.subplots(
        len(joint_names), 1, figsize=(10, 2 * len(joint_names)), sharex=True,
    )
    if len(joint_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, joint_names):
        values = [frame["joints"][name] for frame in data]
        ax.plot(times, values, linewidth=1.5)
        ax.set_ylabel(name, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint Trajectories", fontsize=12, fontweight="bold")
    plt.tight_layout()

    plot_path = "joint_trajectories.png"
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"\n  📈 Plot saved: {plot_path}")


def main():
    print("=" * 60)
    print("🎓 Sample 05: Inspecting Robot Datasets")
    print("=" * 60)

    # Determine what to inspect
    if len(sys.argv) > 1:
        target = sys.argv[1]
    elif os.path.exists("trajectory.json"):
        target = "trajectory.json"
    else:
        # Create a sample trajectory for self-contained demo
        target = create_sample_trajectory()

    # Inspect based on file type
    if os.path.isdir(target):
        inspect_lerobot_dataset_dir(target)
    elif target.endswith(".json"):
        inspect_trajectory_json(target)
        plot_trajectories(target)
    else:
        print(f"  ❓ Unknown format: {target}")
        print("     Supported: .json (trajectory), directory (LeRobot dataset)")

    # Try to find and inspect any LeRobot datasets
    home = Path.home()
    lerobot_dir = home / ".cache" / "huggingface" / "lerobot"
    if lerobot_dir.exists():
        datasets = [d for d in lerobot_dir.iterdir() if d.is_dir()]
        if datasets:
            print(f"\n📁 Found {len(datasets)} cached LeRobot datasets:")
            for d in datasets[:5]:
                print(f"   {d.name}")

    # Summary
    print("\n" + "=" * 60)
    print("🔑 Key takeaways:")
    print("   • Trajectories = timestamped joint positions + actions")
    print("   • LeRobot format = Parquet (numbers) + MP4 (video)")
    print("   • meta/info.json describes shapes, fps, robot type")
    print("   • Always inspect data before training!")
    print("=" * 60)


if __name__ == "__main__":
    main()
