#!/usr/bin/env python3
"""
Sample 05 — Push Dataset to HuggingFace Hub

Upload a locally recorded dataset to HuggingFace Hub so others
can download and train from your demonstrations.

Includes a dry-run mode that shows what WOULD be uploaded without
actually pushing (for students without HuggingFace accounts).

Usage:
    # Dry run (no account needed):
    python push_to_hub.py

    # Real push (requires: huggingface-cli login):
    python push_to_hub.py --push

Requirements:
    pip install strands-robots[sim]
    For actual push: pip install huggingface_hub && huggingface-cli login
"""

import argparse


def check_hf_auth() -> bool:
    """Check if user is authenticated with HuggingFace."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.whoami()
        print(f"  ✅ Authenticated as: {info['name']}")
        return True
    except Exception:
        return False


def dry_run_push():
    """Show what a push would look like without actually pushing."""
    print("\n── Dry Run Mode ──")
    print("   (Shows what would be uploaded without actually pushing)")
    print()

    # Create a small sample dataset structure
    sample_info = {
        "codebase_version": "v3.0",
        "robot_type": "so100",
        "fps": 30,
        "total_episodes": 5,
        "total_frames": 450,
        "total_tasks": 1,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ],
            },
        },
    }

    print("  📋 Dataset that would be uploaded:")
    print(f"     Robot type: {sample_info['robot_type']}")
    print(f"     FPS: {sample_info['fps']}")
    print(f"     Episodes: {sample_info['total_episodes']}")
    print(f"     Frames: {sample_info['total_frames']}")
    print(f"     Features: {list(sample_info['features'].keys())}")
    print()
    print("  📁 Files that would be uploaded:")
    print("     meta/info.json            (dataset metadata)")
    print("     meta/episodes.jsonl        (episode boundaries)")
    print("     data/chunk-000/")
    print("       episode_000000.parquet   (~15 KB)")
    print("       episode_000001.parquet   (~15 KB)")
    print("       episode_000002.parquet   (~15 KB)")
    print("       episode_000003.parquet   (~15 KB)")
    print("       episode_000004.parquet   (~15 KB)")
    print()
    print("  🌐 Would be available at:")
    print("     https://huggingface.co/datasets/<your-username>/sample05_reach_cube")
    print()
    print("  📥 Others could download with:")
    print('     from lerobot.datasets import LeRobotDataset')
    print('     ds = LeRobotDataset("<your-username>/sample05_reach_cube")')
    print(f'     print(len(ds))  # → {sample_info["total_frames"]}')
    print()
    print("  💡 To actually push, run:")
    print("     huggingface-cli login")
    print("     python push_to_hub.py --push")


def real_push():
    """Actually record and push a dataset to HuggingFace Hub."""
    print("\n── Real Push Mode ──\n")

    if not check_hf_auth():
        print("  ❌ Not authenticated with HuggingFace.")
        print("     Run: huggingface-cli login")
        print("     Then try again with: python push_to_hub.py --push")
        return

    # Get username
    from huggingface_hub import HfApi

    api = HfApi()
    username = api.whoami()["name"]

    repo_id = f"{username}/sample05_reach_cube"
    print(f"  📦 Will push to: {repo_id}")

    # Record a small dataset
    from strands_robots import Robot

    sim = Robot("so100")
    sim(
        action="add_object",
        name="cube",
        shape="box",
        size=[0.04, 0.04, 0.04],
        position=[0.25, 0.0, 0.52],
        color=[1.0, 0.0, 0.0, 1.0],
    )

    num_episodes = 3
    for ep in range(num_episodes):
        sim(
            action="start_recording",
            repo_id=repo_id,
            task="reach toward the red cube",
            fps=30,
            push_to_hub=ep == num_episodes - 1,  # Push on last episode
        )

        sim(
            action="run_policy",
            robot_name="so100",
            policy_provider="mock",
            duration=3.0,
        )

        result = sim(action="stop_recording")
        print(f"  Episode {ep + 1}/{num_episodes}: {result}")

    print(f"\n  ✅ Dataset pushed to: https://huggingface.co/datasets/{repo_id}")
    print("  📥 Download with:")
    print('     from lerobot.datasets import LeRobotDataset')
    print(f'     ds = LeRobotDataset("{repo_id}")')


def main():
    print("=" * 60)
    print("🎓 Sample 05: Push Dataset to HuggingFace Hub")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Push robot dataset to HuggingFace")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Actually push (requires huggingface-cli login)",
    )
    args = parser.parse_args()

    if args.push:
        real_push()
    else:
        dry_run_push()

    # Summary
    print("\n" + "=" * 60)
    print("🔑 HuggingFace Hub integration:")
    print("   • LeRobot datasets are HuggingFace-native")
    print("   • Anyone can download and train from shared data")
    print("   • Pre-trained models reference their training datasets")
    print("   • Community datasets: https://huggingface.co/lerobot")
    print("=" * 60)


if __name__ == "__main__":
    main()
