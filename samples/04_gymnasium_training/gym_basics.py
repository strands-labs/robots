#!/usr/bin/env python3
"""
Sample 04: Gymnasium Training - Random Agent Baseline
=====================================================

This script demonstrates the basic Gymnasium API with strands-robots:
1. Create a StrandsSimEnv (wraps Robot() as a Gym environment)
2. Inspect observation and action spaces
3. Run a random agent for 100 steps
4. Collect and visualize rewards

Level: K12 Level 2 (Middle School)
Time: 15 minutes
"""

import numpy as np

try:
    from strands_robots.envs import StrandsSimEnv
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    print("⚠️  strands-robots not installed. Running in demo mode.")
    print("   Install with: pip install strands-robots")


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def demo_gym_basics() -> None:
    """
    Demonstrate Gymnasium API basics with a random agent.

    This function:
    1. Creates a StrandsSimEnv
    2. Prints observation/action spaces
    3. Runs a random agent for 100 steps
    4. Collects rewards and prints statistics
    """
    print_header("Gymnasium Training Basics - Random Agent")

    if not STRANDS_AVAILABLE:
        print("Demo mode: Simulating Gymnasium environment...")
        print("\nWhat you would see with strands-robots installed:")
        print("  env = StrandsSimEnv(data_config='so100')")
        print("  obs, info = env.reset()")
        print("  action = env.action_space.sample()")
        print("  obs, reward, terminated, truncated, info = env.step(action)")
        print("\nObservation Space: Box(low=-inf, high=inf, shape=(18,), dtype=float32)")
        print("Action Space: Box(low=-1.0, high=1.0, shape=(6,), dtype=float32)")
        print("\nRunning 100 random steps (simulated)...")

        # Simulate random rewards
        rewards = []
        for step in range(100):
            reward = np.random.uniform(-1.0, 0.5)
            rewards.append(reward)
            if step < 10 or step % 20 == 0:
                print(f"  Step {step:3d}: Reward = {reward:7.4f}")

        print("\n📊 Statistics (100 steps):")
        print(f"  Average Reward: {np.mean(rewards):7.4f}")
        print(f"  Min Reward:     {np.min(rewards):7.4f}")
        print(f"  Max Reward:     {np.max(rewards):7.4f}")
        print(f"  Std Deviation:  {np.std(rewards):7.4f}")

        print("\n✅ Demo complete! Install strands-robots to run real training.")
        return

    # Real implementation with strands-robots
    try:
        # Step 1: Create the Gymnasium environment
        print("🤖 Creating StrandsSimEnv with SO-100 configuration...")
        env = StrandsSimEnv(data_config="so100")
        print("✅ Environment created successfully!")

        # Step 2: Inspect observation and action spaces
        print_header("Environment Spaces")
        print("📥 Observation Space:")
        print(f"   {env.observation_space}")
        print(f"   Shape: {env.observation_space.shape}")
        print(f"   Type: {env.observation_space.dtype}")

        print("\n📤 Action Space:")
        print(f"   {env.action_space}")
        print(f"   Shape: {env.action_space.shape}")
        print(f"   Type: {env.action_space.dtype}")
        print(f"   Range: [{env.action_space.low[0]:.2f}, {env.action_space.high[0]:.2f}]")

        # Step 3: Run a random agent
        print_header("Running Random Agent (100 steps)")

        obs, info = env.reset()
        print(f"🔄 Environment reset. Initial observation shape: {obs.shape}")
        print(f"   Initial observation sample: {obs[:5]}...")  # First 5 values

        rewards = []
        observations = []
        episode_count = 0

        for step in range(100):
            # Sample a random action
            action = env.action_space.sample()

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Store data
            rewards.append(reward)
            observations.append(obs)

            # Print progress
            if step < 10 or step % 20 == 0:
                print(f"  Step {step:3d}: Reward = {reward:7.4f} | Obs[0:3] = {obs[:3]}")

            # Reset if episode ended
            if terminated or truncated:
                episode_count += 1
                print(f"  ⚠️  Episode ended at step {step} ({'terminated' if terminated else 'truncated'})")
                obs, info = env.reset()

        # Step 4: Print statistics
        print_header("Statistics Summary")
        print("📊 Performance over 100 steps:")
        print(f"  Total Episodes:  {episode_count + 1}")
        print(f"  Average Reward:  {np.mean(rewards):7.4f}")
        print(f"  Min Reward:      {np.min(rewards):7.4f}")
        print(f"  Max Reward:      {np.max(rewards):7.4f}")
        print(f"  Std Deviation:   {np.std(rewards):7.4f}")
        print(f"  Cumulative:      {np.sum(rewards):7.4f}")

        # Observation statistics
        obs_array = np.array(observations)
        print("\n📏 Observation Statistics:")
        print(f"  Mean:  {np.mean(obs_array, axis=0)[:5]}...")
        print(f"  Std:   {np.std(obs_array, axis=0)[:5]}...")

        # Reward curve visualization (simple text-based)
        print_header("Reward Curve (Last 20 Steps)")
        last_20 = rewards[-20:]
        max_reward = max(last_20)
        min_reward = min(last_20)
        range_reward = max_reward - min_reward if max_reward != min_reward else 1.0

        for i, r in enumerate(last_20):
            normalized = int(((r - min_reward) / range_reward) * 40)
            bar = "█" * normalized
            print(f"  {80+i:3d}: {bar} {r:6.3f}")

        # Reset environment one more time to demonstrate
        print_header("Resetting Environment")
        obs, info = env.reset()
        print("✅ Environment reset successfully!")
        print(f"   New observation shape: {obs.shape}")
        print(f"   Info keys: {list(info.keys())}")

        print("\n" + "="*60)
        print("  🎉 Gymnasium Basics Demo Complete!")
        print("="*60)
        print("\n💡 Key Takeaways:")
        print("  1. Gymnasium provides a standard API: reset(), step(action)")
        print("  2. Observation space defines what the robot 'sees'")
        print("  3. Action space defines what the robot can control")
        print("  4. Random actions = baseline performance (usually poor!)")
        print("  5. RL training will learn to maximize rewards")

        print("\n📚 Next Steps:")
        print("  → Run train_ppo.py to train a real RL agent")
        print("  → Try custom_reward.py to define your own reward function")
        print("  → Experiment with different robot configurations")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("   This might be due to missing dependencies or configuration issues.")
        print("   Please ensure MuJoCo and strands-robots are properly installed.")
        import traceback
        traceback.print_exc()


def main() -> None:
    """Main entry point."""
    demo_gym_basics()


if __name__ == "__main__":
    main()
