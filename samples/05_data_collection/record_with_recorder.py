#!/usr/bin/env python3
"""
Sample 05 — Record with DatasetRecorder (Low-Level API)

The DatasetRecorder class gives you fine-grained control over
multi-episode dataset creation. This is what simulation.py uses
internally when you call start_recording/stop_recording.

This script demonstrates:
1. Creating a DatasetRecorder with custom features
2. Manually adding frames (observation + action) per control step
3. Saving episodes and inspecting the output

Note: This requires lerobot to be installed for full functionality.
Without lerobot, we demonstrate the API pattern with mock data.

Usage:
    python record_with_recorder.py

Requirements:
    pip install strands-robots[sim]
    pip install lerobot  (for actual LeRobotDataset creation)
"""

import json
import math

import numpy as np


def demo_with_simulation():
    """Record using the high-level simulation API (recommended)."""
    print("\n── Method 1: High-Level Simulation Recording ──")
    print("   (Recommended for most users)\n")

    from strands_robots import Robot

    sim = Robot("so100")

    # Add scene objects
    sim(
        action="add_object",
        name="target",
        shape="sphere",
        size=[0.03],
        position=[0.2, 0.1, 0.55],
        color=[0.0, 1.0, 0.0, 1.0],
    )

    # Record 3 episodes using the built-in recording pipeline
    for ep_idx in range(3):
        sim(
            action="start_recording",
            repo_id="local/sample05_highlevel",
            task="reach the green sphere",
            fps=30,
        )

        sim(
            action="run_policy",
            robot_name="so100",
            policy_provider="mock",
            duration=2.0,
        )

        result = sim(action="stop_recording")
        print(f"   Episode {ep_idx + 1}: {result}")

    print("   ✅ High-level recording complete")


def demo_with_dataset_recorder():
    """Record using the low-level DatasetRecorder API.

    This shows what happens under the hood when you call start_recording.
    Use this when you need custom control over the recording process.
    """
    print("\n── Method 2: Low-Level DatasetRecorder API ──")
    print("   (For custom recording pipelines)\n")

    try:
        from strands_robots.dataset_recorder import DatasetRecorder
    except ImportError:
        print("   ⚠️ DatasetRecorder requires lerobot. Showing mock version.")
        demo_mock_recorder()
        return

    # Check if lerobot is actually available
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401

        has_lerobot = True
    except ImportError:
        has_lerobot = False

    if not has_lerobot:
        print("   ⚠️ lerobot not installed. Showing mock version.")
        demo_mock_recorder()
        return

    # Create recorder with explicit features
    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    recorder = DatasetRecorder.create(
        repo_id="local/sample05_manual",
        fps=30,
        robot_type="so100",
        joint_names=joint_names,
        task="reach and grasp object",
        use_videos=False,  # No cameras in this example
    )

    print(f"   Created: {recorder}")

    # Simulate 3 episodes of a control loop
    for ep_idx in range(3):
        num_steps = 60  # 2 seconds at 30 fps

        for step in range(num_steps):
            t = step / 30.0

            # Simulated observation (joint positions)
            observation = {}
            for j, name in enumerate(joint_names):
                freq = 0.5 + j * 0.15
                amplitude = 0.2 + j * 0.03
                observation[name] = amplitude * math.sin(2 * math.pi * freq * t)

            # Simulated action (target joint positions)
            action = {}
            for j, name in enumerate(joint_names):
                freq = 0.5 + j * 0.15
                amplitude = 0.2 + j * 0.03
                # Action is slightly ahead in time (predictive)
                action[name] = amplitude * math.sin(
                    2 * math.pi * freq * (t + 1 / 30)
                )

            # Record the frame
            recorder.add_frame(
                observation=observation,
                action=action,
                task="reach and grasp object",
            )

        # Save the episode
        result = recorder.save_episode()
        print(f"   Episode {ep_idx + 1}: {result}")

    # Finalize
    recorder.finalize()
    print(f"\n   ✅ Dataset saved to: {recorder.root}")
    print(f"   Total frames: {recorder.frame_count}")
    print(f"   Total episodes: {recorder.episode_count}")


def demo_mock_recorder():
    """Demonstrate the recording pattern without lerobot dependency.

    This creates the same data structure manually, showing students
    what the DatasetRecorder does internally.
    """
    print("\n   📝 Mock recording (no lerobot needed):")
    print("      This shows the data flow without creating real datasets.\n")

    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    all_episodes = []

    for ep_idx in range(3):
        episode_data = {
            "observations": [],
            "actions": [],
            "task": "reach and grasp object",
            "fps": 30,
        }

        num_steps = 60  # 2 seconds at 30fps

        for step in range(num_steps):
            t = step / 30.0

            # Observation: current joint positions
            obs = np.array([
                (0.2 + j * 0.03) * math.sin(2 * math.pi * (0.5 + j * 0.15) * t)
                for j in range(len(joint_names))
            ], dtype=np.float32)

            # Action: target joint positions
            act = np.array([
                (0.2 + j * 0.03) * math.sin(
                    2 * math.pi * (0.5 + j * 0.15) * (t + 1 / 30)
                )
                for j in range(len(joint_names))
            ], dtype=np.float32)

            episode_data["observations"].append(obs.tolist())
            episode_data["actions"].append(act.tolist())

        all_episodes.append(episode_data)
        print(f"      Episode {ep_idx + 1}: {num_steps} frames recorded")

    # Save as JSON (what a real recorder would save as parquet)
    output_path = "mock_dataset.json"
    output = {
        "info": {
            "fps": 30,
            "robot_type": "so100",
            "total_episodes": len(all_episodes),
            "total_frames": sum(
                len(ep["observations"]) for ep in all_episodes
            ),
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [len(joint_names)],
                    "names": joint_names,
                },
                "action": {
                    "dtype": "float32",
                    "shape": [len(joint_names)],
                    "names": joint_names,
                },
            },
        },
        "episodes": [
            {
                "episode_index": i,
                "length": len(ep["observations"]),
                "task": ep["task"],
            }
            for i, ep in enumerate(all_episodes)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n      ✅ Mock dataset info saved: {output_path}")
    print(f"      Total: {output['info']['total_episodes']} episodes, "
          f"{output['info']['total_frames']} frames")
    print(f"      State shape: {output['info']['features']['observation.state']['shape']}")
    print(f"      Action shape: {output['info']['features']['action']['shape']}")

    # Show what a real parquet row would look like
    print("\n      📊 What a real dataset row looks like:")
    print("      | observation.state          | action                     | task                    |")
    print(f"      | {all_episodes[0]['observations'][0][:3]}... | {all_episodes[0]['actions'][0][:3]}... | reach and grasp object  |")


def main():
    print("=" * 60)
    print("🎓 Sample 05: DatasetRecorder — Multi-Episode Capture")
    print("=" * 60)

    print("\nTwo ways to record data:\n")
    print("  Method 1: sim.start_recording() / stop_recording()")
    print("     → Best for quick capture in simulation")
    print("  Method 2: DatasetRecorder.create() + add_frame()")
    print("     → Best for custom pipelines and real hardware")

    # Method 1: High-level (uses MuJoCo)
    try:
        demo_with_simulation()
    except Exception as e:
        print(f"   ⚠️ Simulation recording skipped: {e}")
        print("   (This requires mujoco — pip install strands-robots[sim])")

    # Method 2: Low-level (DatasetRecorder)
    demo_with_dataset_recorder()

    # Summary
    print("\n" + "=" * 60)
    print("🔑 DatasetRecorder key methods:")
    print("   .create()        — Initialize with features")
    print("   .add_frame()     — Record one control step")
    print("   .save_episode()  — Finalize episode (parquet + video)")
    print("   .push_to_hub()   — Upload to HuggingFace")
    print("   .finalize()      — Close all writers")
    print("=" * 60)


if __name__ == "__main__":
    main()
