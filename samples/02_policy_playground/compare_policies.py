#!/usr/bin/env python3
"""
Sample 02: Compare Policies Exercise
K12 Level 1 (Elementary)

Compare how the same policy behaves on different robots!

This exercise demonstrates:
1. Running the same policy on multiple robots
2. Recording videos for comparison
3. Understanding action dimensions (number of joints)
4. Observing how robot morphology affects behavior
"""

from strands_robots import Robot


def test_robot_with_mock_policy(robot_name: str):
    """
    Test a robot with the mock policy and report results.

    Args:
        robot_name: Name of the robot to test (e.g., "so100", "panda")
    """
    print(f"\n{'=' * 60}")
    print(f"🤖 Testing: {robot_name}")
    print(f"{'=' * 60}\n")

    # Create robot simulation
    print(f"📦 Creating {robot_name} simulation...")
    sim = Robot(robot_name)
    print("✅ Robot created!\n")

    # Run mock policy
    print("🎪 Running mock policy (2 seconds)...")
    sim.run_policy(
        robot_name=robot_name,
        policy_provider="mock",
        duration=2.0,
    )
    print("✅ Policy completed!\n")

    # Record video
    video_path = f"mock_{robot_name}.mp4"
    print(f"🎥 Recording video: {video_path}")
    sim.record_video(
        robot_name=robot_name,
        policy_provider="mock",
        duration=2.0,
        fps=30,
        output_path=video_path,
    )
    print(f"✅ Video saved: {video_path}\n")

    # Get robot info
    print("📊 Robot Information:")
    print(f"   • Name: {robot_name}")
    print(f"   • Video: {video_path}")

    # Try to get action dimensions from the robot
    # (This shows how many joints the robot has)
    try:
        # Get observation space to infer action dimensions
        # Different robots have different numbers of joints
        from strands_robots.envs import make_env

        env = make_env(robot_name)
        action_space = env.action_space
        action_dim = action_space.shape[0] if hasattr(action_space, "shape") else "?"

        print(f"   • Action Dimensions: {action_dim} joints")
        print(f"   • Action Space: {action_space}")

    except Exception as e:
        print(f"   • Action Dimensions: Could not determine ({e})")

    print()


def main():
    """Compare mock policy on multiple robots."""
    print("=" * 60)
    print("🔬 Policy Comparison Lab")
    print("=" * 60)
    print()
    print("This exercise compares how the SAME policy (mock)")
    print("behaves on DIFFERENT robots.")
    print()
    print("We'll test 4 robots:")
    print("  1. so100      - Stanford Humanoid (~20 joints)")
    print("  2. unitree_g1 - Unitree G1 Humanoid (~30 joints)")
    print("  3. panda      - Franka Panda Arm (7 joints)")
    print("  4. aloha      - ALOHA Bimanual (14 joints)")
    print()
    print("Each will run the mock policy and record a video.")
    print()

    # List of robots to test
    robots_to_test = [
        "so100",  # Stanford Humanoid
        "unitree_g1",  # Unitree G1 Humanoid
        "panda",  # Franka Panda Arm
        "aloha",  # ALOHA Bimanual System
    ]

    # Store results
    results = []

    # Test each robot
    for robot_name in robots_to_test:
        try:
            test_robot_with_mock_policy(robot_name)
            results.append((robot_name, "✅ Success"))
        except Exception as e:
            print(f"❌ Error testing {robot_name}: {e}")
            results.append((robot_name, f"❌ Failed: {e}"))
            print()

    # Summary
    print("=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print()

    for robot_name, status in results:
        print(f"  {robot_name:15s} → {status}")

    print()
    print("=" * 60)
    print("🤔 Think About It!")
    print("=" * 60)
    print()
    print("Questions to explore:")
    print()
    print("1. Do all robots move the same way with the mock policy?")
    print("   Why or why not?")
    print()
    print("2. Which robot has the most joints? The fewest?")
    print("   How does this affect the motion?")
    print()
    print("3. What would happen with the 'stopped' policy?")
    print("   Try changing 'mock' to 'stopped' and find out!")
    print()
    print("4. Can you predict what the 'random' policy would do?")
    print()
    print("💡 Tip: Watch the videos side-by-side to compare!")
    print()
    print("🚀 Challenge: Modify this script to test the 'stopped'")
    print("   or 'random' policy instead of 'mock'!")
    print()


if __name__ == "__main__":
    main()
