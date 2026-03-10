#!/usr/bin/env python3
"""
Sample 05 — Record Episodes in Simulation

Record 5 episodes of a mock policy controlling an SO-100 arm.
Each episode captures joint positions, velocities, and actions
into a LeRobotDataset (parquet + optional video).

This is the simplest path from "robot moving" to "training data."

Usage:
    python record_episodes.py

What happens:
    1. Creates a MuJoCo simulation with an SO-100 arm and a cube
    2. Starts LeRobotDataset recording
    3. Runs a mock policy for 5 episodes (3 seconds each)
    4. Saves each episode as parquet + optional video
    5. Prints a summary of the recorded dataset

Requirements:
    pip install strands-robots[sim]
"""

import json
import os

# strands_robots uses MuJoCo under the hood — no GPU needed
from strands_robots import Robot


def main():
    print("=" * 60)
    print("🎓 Sample 05: Recording Robot Episodes")
    print("=" * 60)

    # ── Step 1: Create a simulation ──────────────────────────────
    print("\n📦 Creating simulation with SO-100 arm...")
    sim = Robot("so100")

    # The Robot() factory auto-detects simulation mode (no hardware found)
    # and creates a MuJoCo world with the SO-100 arm.

    # Add an object to make the scene interesting
    sim(
        action="add_object",
        name="red_cube",
        shape="box",
        size=[0.04, 0.04, 0.04],
        position=[0.25, 0.0, 0.52],
        color=[1.0, 0.0, 0.0, 1.0],
    )
    print("  ✅ World created with SO-100 + red cube")

    # ── Step 2: Record episodes ──────────────────────────────────
    num_episodes = 5
    episode_duration = 3.0  # seconds each

    print(f"\n🔴 Recording {num_episodes} episodes ({episode_duration}s each)...")

    for ep in range(num_episodes):
        # Start recording to LeRobotDataset format.
        # This creates parquet files for joint data and MP4 for cameras.
        sim(
            action="start_recording",
            repo_id="local/sample05_episodes",
            task="reach toward the red cube",
            fps=30,
        )

        # Run a mock policy — it generates sinusoidal joint trajectories.
        # In a real scenario, this would be a trained VLA (ACT, Pi0, etc.)
        # or teleoperation from a human.
        sim(
            action="run_policy",
            robot_name="so100",
            policy_provider="mock",
            duration=episode_duration,
        )

        # Stop recording — saves the episode (parquet + video encoding)
        result = sim(action="stop_recording")
        print(f"  Episode {ep + 1}/{num_episodes}: {result}")

    # ── Step 3: Also record a video for visual inspection ────────
    print("\n🎬 Recording a demo video...")
    sim(
        action="record_video",
        robot_name="so100",
        policy_provider="mock",
        duration=3.0,
        fps=30,
        output_path="demo_episode.mp4",
    )
    if os.path.exists("demo_episode.mp4"):
        size_kb = os.path.getsize("demo_episode.mp4") / 1024
        print(f"  ✅ Video saved: demo_episode.mp4 ({size_kb:.0f} KB)")

    # ── Step 4: Examine the trajectory data ──────────────────────
    print("\n📊 Examining trajectory data...")

    # The simulation also stores raw trajectory as JSON
    sim(action="start_recording")
    sim(
        action="run_policy",
        robot_name="so100",
        policy_provider="mock",
        duration=2.0,
    )
    result = sim(action="stop_recording", output_path="trajectory.json")

    if os.path.exists("trajectory.json"):
        with open("trajectory.json") as f:
            trajectory = json.load(f)

        if isinstance(trajectory, list) and len(trajectory) > 0:
            print(f"  Trajectory frames: {len(trajectory)}")
            first = trajectory[0]
            print(f"  First frame keys: {list(first.keys())}")
            if "time" in first:
                last = trajectory[-1]
                print(f"  Time range: {first['time']:.3f}s → {last['time']:.3f}s")
            if "joints" in first:
                print(f"  Joints recorded: {len(first['joints'])}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Recording complete!")
    print(f"   Episodes recorded: {num_episodes}")
    print(f"   Duration per episode: {episode_duration}s")
    print("   FPS: 30")
    print(f"   Expected frames/episode: ~{int(episode_duration * 30)}")
    print()
    print("📁 Output files:")
    print("   trajectory.json    — Raw joint trajectory (JSON)")
    if os.path.exists("demo_episode.mp4"):
        print("   demo_episode.mp4   — Visual demo video")
    print()
    print("🔑 Key takeaway: Recording = run a policy + capture obs/action each step")
    print("=" * 60)


if __name__ == "__main__":
    main()
