#!/usr/bin/env python3
"""
Sample 04: Gymnasium Training - PPO Training
============================================

This script demonstrates how to train a Proximal Policy Optimization (PPO)
agent using the RLTrainer from strands_robots.training.rl_trainer.

Steps:
1. Create RLTrainer with SO-100 configuration
2. Train for 1000 steps (small demo)
3. Evaluate the trained policy
4. Save and load checkpoints

Level: K12 Level 2 (Middle School)
Time: 20 minutes
"""

import os
import sys

try:
    from strands_robots.training.rl_trainer import RLTrainer
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False
    print("⚠️  strands-robots training module not available.")


try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def check_dependencies() -> bool:
    """
    Check if all required dependencies are available.

    Returns:
        bool: True if all dependencies are available, False otherwise.
    """
    print_header("Dependency Check")

    # Check strands-robots
    if not TRAINER_AVAILABLE:
        print("❌ strands-robots training module not found")
        print("   Install with: pip install strands-robots")
        return False
    else:
        print("✅ strands-robots training module found")

    # Check GPU (optional but recommended)
    if GPU_AVAILABLE:
        print("✅ GPU available - training will be faster!")
        try:
            import torch
            print(f"   Device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    else:
        print("⚠️  No GPU detected - training will be slower")
        print("   Consider using Google Colab for free GPU access")

    return True


def train_ppo_agent() -> None:
    """
    Train a PPO agent on the SO-100 robot configuration.

    This function demonstrates:
    1. Creating an RLTrainer instance
    2. Training for a small number of steps
    3. Evaluating the trained policy
    4. Saving and loading checkpoints
    """
    print_header("PPO Training Demo")

    if not TRAINER_AVAILABLE:
        print("Demo mode: Showing what PPO training would look like...\n")
        print("Code structure:")
        print("  trainer = RLTrainer(")
        print("      env_name='StrandsSimEnv',")
        print("      data_config='so100',")
        print("      algorithm='PPO',")
        print("      total_timesteps=1000")
        print("  )")
        print("  trainer.train()")
        print("  trainer.save('ppo_so100_checkpoint')")
        print("  trainer.evaluate(num_episodes=5)")
        print("\nExpected output:")
        print("  Timestep 100/1000 | Avg Reward: -5.23")
        print("  Timestep 200/1000 | Avg Reward: -4.15")
        print("  Timestep 300/1000 | Avg Reward: -3.67")
        print("  ...")
        print("  Training complete!")
        print("  Evaluation (5 episodes): -2.34 ± 0.56")
        print("\n✅ Demo complete! Install dependencies to run real training.")
        return

    try:
        # Configuration
        TOTAL_TIMESTEPS = 1000  # Small for demo (use 100k+ for real training)
        CHECKPOINT_DIR = "ppo_so100_checkpoint"
        NUM_EVAL_EPISODES = 5

        print("🤖 Configuration:")
        print("   Robot: SO-100")
        print("   Algorithm: PPO")
        print(f"   Total Timesteps: {TOTAL_TIMESTEPS}")
        print(f"   Checkpoint Directory: {CHECKPOINT_DIR}")
        print(f"   GPU: {'Yes' if GPU_AVAILABLE else 'No'}")

        # Step 1: Create the trainer
        print_header("Creating RLTrainer")
        print("Initializing trainer...")

        trainer = RLTrainer(
            env_name="StrandsSimEnv",
            data_config="so100",
            algorithm="PPO",
            total_timesteps=TOTAL_TIMESTEPS,
            # PPO hyperparameters (reasonable defaults for beginners)
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )

        print("✅ Trainer created successfully!")

        # Step 2: Train the agent
        print_header("Training PPO Agent")
        print(f"🚀 Starting training for {TOTAL_TIMESTEPS} timesteps...")
        print("   (This may take a few minutes)\n")

        trainer.train()

        print("\n✅ Training complete!")

        # Step 3: Evaluate the trained policy
        print_header("Evaluating Trained Policy")
        print(f"Running {NUM_EVAL_EPISODES} evaluation episodes...\n")

        mean_reward, std_reward = trainer.evaluate(num_episodes=NUM_EVAL_EPISODES)

        print("\n📊 Evaluation Results:")
        print(f"   Mean Reward: {mean_reward:.2f}")
        print(f"   Std Reward:  {std_reward:.2f}")
        print(f"   Episodes:    {NUM_EVAL_EPISODES}")

        # Step 4: Save the checkpoint
        print_header("Saving Checkpoint")
        print(f"💾 Saving model to {CHECKPOINT_DIR}...")

        trainer.save(CHECKPOINT_DIR)

        print("✅ Checkpoint saved successfully!")
        print(f"   Files: {os.listdir(CHECKPOINT_DIR)}")

        # Step 5: Demonstrate loading
        print_header("Loading Checkpoint (Demo)")
        print(f"📂 Loading model from {CHECKPOINT_DIR}...")

        # Create a new trainer and load the checkpoint
        new_trainer = RLTrainer(
            env_name="StrandsSimEnv",
            data_config="so100",
            algorithm="PPO"
        )
        new_trainer.load(CHECKPOINT_DIR)

        print("✅ Checkpoint loaded successfully!")
        print("   You can now continue training or deploy this policy.")

        # Step 6: Summary
        print_header("Training Summary")
        print("🎉 PPO Training Complete!")
        print("\n📈 What happened:")
        print("  1. Created a PPO trainer for SO-100 robot")
        print("  2. Trained for 1000 timesteps (very short demo)")
        print("  3. Evaluated the policy on 5 episodes")
        print("  4. Saved checkpoint for future use")
        print("  5. Demonstrated checkpoint loading")

        print("\n💡 Key Concepts:")
        print("  • PPO: A stable, sample-efficient RL algorithm")
        print("  • Timesteps: Number of environment interactions")
        print("  • Checkpoint: Saved model weights for later use")
        print("  • Evaluation: Test the policy without training")

        print("\n🔬 Experiment Ideas:")
        print("  1. Increase timesteps to 10,000+ for better learning")
        print("  2. Adjust learning_rate (try 1e-3, 3e-4, 1e-4)")
        print("  3. Change n_steps (buffer size before update)")
        print("  4. Compare with other algorithms (SAC, TD3, A2C)")

        print("\n📚 Next Steps:")
        print("  → Run custom_reward.py to define custom rewards")
        print("  → Increase training time for better performance")
        print("  → Deploy the trained policy on a real robot!")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\nPossible causes:")
        print("  • Missing dependencies (MuJoCo, Gymnasium, Stable-Baselines3)")
        print("  • GPU memory issues (reduce batch_size or n_steps)")
        print("  • Invalid configuration")

        import traceback
        traceback.print_exc()

        print("\n🛟 Troubleshooting:")
        print("  1. Check that all dependencies are installed")
        print("  2. Try reducing TOTAL_TIMESTEPS to 500")
        print("  3. Ensure MuJoCo license is valid (free since v2.3.0)")
        print("  4. Run gym_basics.py first to verify environment works")


def main() -> None:
    """Main entry point."""
    # Check dependencies first
    if not check_dependencies():
        print("\n⚠️  Cannot proceed without required dependencies.")
        print("   Install with: pip install strands-robots[training]")
        sys.exit(1)

    # Run training
    train_ppo_agent()


if __name__ == "__main__":
    main()
