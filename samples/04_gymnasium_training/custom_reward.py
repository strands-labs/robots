#!/usr/bin/env python3
"""
Sample 04: Gymnasium Training - Custom Reward Functions
=======================================================

This script demonstrates how to create and use custom reward functions
to guide robot learning behavior.

Examples:
1. Reach target position reward
2. Smooth motion reward (penalize high velocities)
3. Stay upright reward
4. Combined multi-objective reward

Level: K12 Level 2 (Middle School)
Time: 20 minutes
"""

import numpy as np

try:
    from strands_robots.envs import StrandsSimEnv
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    print("⚠️  strands-robots not installed. Running in demo mode.")


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


# ============================================================================
# Custom Reward Functions
# ============================================================================

def reach_target_reward(obs: np.ndarray, action: np.ndarray, info: dict) -> float:
    """
    Reward the robot for reaching a target position.

    This is a common RL task: teach the robot to navigate to a specific
    location in 3D space.

    Args:
        obs: Observation vector (robot state)
        action: Action taken
        info: Additional information from environment

    Returns:
        float: Reward value (higher = better)
    """
    # Define target position in 3D space (x, y, z)
    target_pos = np.array([1.0, 0.5, 0.3])

    # Extract current position from observation
    # (Assuming first 3 elements are x, y, z coordinates)
    current_pos = obs[:3]

    # Calculate Euclidean distance to target
    distance = np.linalg.norm(target_pos - current_pos)

    # Reward is negative distance (closer = higher reward)
    reward = -distance

    # Bonus for reaching target (within 10cm)
    if distance < 0.1:
        reward += 10.0
        print(f"🎯 Target reached! Distance: {distance:.4f}m")

    return reward


def smooth_motion_reward(obs: np.ndarray, action: np.ndarray, info: dict) -> float:
    """
    Encourage smooth, controlled motion by penalizing high velocities.

    This teaches the robot to move gracefully rather than jerkily.

    Args:
        obs: Observation vector (robot state)
        action: Action taken
        info: Additional information from environment

    Returns:
        float: Reward value (higher = better)
    """
    # Extract joint velocities from observation
    # (Assuming obs[6:12] are joint velocities for 6-DOF robot)
    if len(obs) >= 12:
        velocities = obs[6:12]
    else:
        velocities = obs[len(obs)//2:]  # Fallback: second half of obs

    # Penalize high velocities (L1 norm)
    velocity_penalty = -0.01 * np.sum(np.abs(velocities))

    # Penalize large actions (encourage efficiency)
    action_penalty = -0.001 * np.sum(np.square(action))

    # Small positive reward for existing
    base_reward = 0.1

    return base_reward + velocity_penalty + action_penalty


def stay_upright_reward(obs: np.ndarray, action: np.ndarray, info: dict) -> float:
    """
    Reward the robot for maintaining an upright position.

    This is useful for balancing tasks (e.g., humanoid robots, inverted pendulum).

    Args:
        obs: Observation vector (robot state)
        action: Action taken
        info: Additional information from environment

    Returns:
        float: Reward value (higher = better)
    """
    # Extract z-coordinate (height) from observation
    z_pos = obs[2] if len(obs) > 2 else 0.0

    # Define minimum acceptable height
    MIN_HEIGHT = 0.3

    # Large penalty if fallen over
    if z_pos < MIN_HEIGHT:
        return -10.0

    # Positive reward for staying upright
    upright_reward = 1.0

    # Bonus for being at optimal height
    optimal_height = 0.5
    height_bonus = -abs(z_pos - optimal_height)

    return upright_reward + height_bonus


def combined_reward(obs: np.ndarray, action: np.ndarray, info: dict) -> float:
    """
    Combine multiple reward objectives with weights.

    This demonstrates multi-objective reward engineering, a common
    technique in real-world RL applications.

    Args:
        obs: Observation vector (robot state)
        action: Action taken
        info: Additional information from environment

    Returns:
        float: Weighted sum of multiple reward components
    """
    # Weight each objective
    w_target = 1.0      # Reach target is most important
    w_smooth = 0.3      # Smooth motion is moderately important
    w_upright = 0.5     # Staying upright is important

    # Calculate individual rewards
    r_target = reach_target_reward(obs, action, info)
    r_smooth = smooth_motion_reward(obs, action, info)
    r_upright = stay_upright_reward(obs, action, info)

    # Weighted sum
    total_reward = (w_target * r_target +
                    w_smooth * r_smooth +
                    w_upright * r_upright)

    return total_reward


# ============================================================================
# Demo Functions
# ============================================================================

def demo_custom_reward_functions() -> None:
    """
    Demonstrate custom reward functions with example observations.
    """
    print_header("Custom Reward Functions Demo")

    # Create synthetic observations for demonstration
    print("📊 Testing reward functions with synthetic data:\n")

    # Example 1: Robot close to target
    obs_near_target = np.array([0.95, 0.48, 0.32, 0.0, 0.0, 0.0,  # pos
                                0.1, 0.1, 0.05, 0.0, 0.0, 0.0])   # vel
    action = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
    info = {}

    print("Test 1: Robot near target position")
    print(f"  Position: {obs_near_target[:3]}")
    print("  Target:   [1.0, 0.5, 0.3]")
    r1 = reach_target_reward(obs_near_target, action, info)
    print(f"  Reward:   {r1:.4f}\n")

    # Example 2: Robot far from target
    obs_far_target = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print("Test 2: Robot far from target")
    print(f"  Position: {obs_far_target[:3]}")
    print("  Target:   [1.0, 0.5, 0.3]")
    r2 = reach_target_reward(obs_far_target, action, info)
    print(f"  Reward:   {r2:.4f}\n")

    # Example 3: High velocity motion
    obs_fast = np.array([0.5, 0.5, 0.4, 0.0, 0.0, 0.0,
                         2.5, 3.0, 1.5, 2.0, 1.8, 2.2])  # High velocities!

    print("Test 3: High velocity motion")
    print(f"  Velocities: {obs_fast[6:]}")
    r3 = smooth_motion_reward(obs_fast, action, info)
    print(f"  Reward:     {r3:.4f}\n")

    # Example 4: Low velocity motion
    obs_slow = np.array([0.5, 0.5, 0.4, 0.0, 0.0, 0.0,
                         0.1, 0.1, 0.05, 0.08, 0.06, 0.09])  # Low velocities

    print("Test 4: Smooth, slow motion")
    print(f"  Velocities: {obs_slow[6:]}")
    r4 = smooth_motion_reward(obs_slow, action, info)
    print(f"  Reward:     {r4:.4f}\n")

    # Example 5: Fallen over
    obs_fallen = np.array([0.5, 0.5, 0.1, 0.0, 0.0, 0.0,  # z = 0.1 (too low!)
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print("Test 5: Robot fallen over")
    print(f"  Height (z): {obs_fallen[2]:.2f}")
    r5 = stay_upright_reward(obs_fallen, action, info)
    print(f"  Reward:     {r5:.4f}\n")

    # Example 6: Upright and stable
    obs_upright = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print("Test 6: Robot upright and stable")
    print(f"  Height (z): {obs_upright[2]:.2f}")
    r6 = stay_upright_reward(obs_upright, action, info)
    print(f"  Reward:     {r6:.4f}\n")

    # Example 7: Combined reward
    print("Test 7: Combined multi-objective reward")
    print(f"  Position:   {obs_near_target[:3]}")
    print(f"  Velocities: {obs_near_target[6:]}")
    print(f"  Height:     {obs_near_target[2]:.2f}")
    r7 = combined_reward(obs_near_target, action, info)
    print(f"  Reward:     {r7:.4f}\n")


def demo_custom_reward_with_env() -> None:
    """
    Demonstrate using a custom reward function with StrandsSimEnv.
    """
    print_header("Using Custom Rewards in Training")

    if not STRANDS_AVAILABLE:
        print("Demo mode: Showing how to use custom rewards...\n")
        print("Code example:")
        print("  def my_reward(obs, action, info):")
        print("      # Custom logic here")
        print("      return reward_value")
        print()
        print("  env = StrandsSimEnv(")
        print("      data_config='so100',")
        print("      reward_fn=my_reward  # Hook in your function!")
        print("  )")
        print()
        print("  obs, info = env.reset()")
        print("  action = env.action_space.sample()")
        print("  obs, reward, terminated, truncated, info = env.step(action)")
        print("  # 'reward' now comes from my_reward() function!")
        print("\n✅ Demo complete! Install strands-robots to run with real env.")
        return

    try:
        print("🤖 Creating environment with custom reward function...")

        # Create environment with custom reward
        env = StrandsSimEnv(
            data_config="so100",
            reward_fn=reach_target_reward  # Use our custom reward!
        )

        print("✅ Environment created with reach_target_reward function\n")

        # Run a few steps to show custom rewards in action
        print("🚀 Running 20 steps with custom reward...\n")

        obs, info = env.reset()
        total_reward = 0.0

        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            # Print every 5 steps
            if step % 5 == 0:
                print(f"  Step {step:2d}: Pos={obs[:3].round(3)} | Reward={reward:7.4f}")

            if terminated or truncated:
                print(f"  Episode ended at step {step}")
                obs, info = env.reset()

        print(f"\n📊 Total Reward (20 steps): {total_reward:.4f}")
        print("   (Compare this to default reward to see the difference!)")

        print("\n💡 Key Insight:")
        print("   By changing the reward function, you change what the robot learns.")
        print("   This is called 'reward engineering' and is crucial for RL success!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """Main entry point."""
    # Part 1: Demo reward functions with synthetic data
    demo_custom_reward_functions()

    # Part 2: Demo with real environment (if available)
    demo_custom_reward_with_env()

    # Summary
    print_header("Summary & Next Steps")
    print("🎉 Custom Reward Functions Complete!")
    print("\n📚 What You Learned:")
    print("  1. How to write custom reward functions")
    print("  2. Common reward patterns (reach target, stay upright, smooth motion)")
    print("  3. Multi-objective rewards with weighted sums")
    print("  4. How to hook custom rewards into StrandsSimEnv")

    print("\n🔬 Exercise Ideas:")
    print("  1. Create a reward that keeps the robot in a specific region")
    print("  2. Penalize energy consumption (sum of squared actions)")
    print("  3. Combine 3+ objectives with different weights")
    print("  4. Design a reward for a specific task (pick & place, navigation)")

    print("\n💡 Pro Tips:")
    print("  • Start simple, add complexity gradually")
    print("  • Normalize rewards to similar scales (e.g., -1 to +1)")
    print("  • Use dense rewards (every step) not sparse (only at goal)")
    print("  • Log reward components separately for debugging")

    print("\n📖 Further Reading:")
    print("  • Reward Shaping: https://arxiv.org/abs/1908.08542")
    print("  • Multi-Objective RL: https://arxiv.org/abs/1310.1196")
    print("  • Reward Engineering Best Practices: OpenAI Spinning Up")


if __name__ == "__main__":
    main()
