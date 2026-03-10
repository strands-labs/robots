#!/usr/bin/env python3
"""
Sample 06 — Isaac Lab RL Training Pipeline.

Full RL training using Isaac Lab's infrastructure with strands-robots' unified
trainer API. Supports 4 RL frameworks:

1. RSL-RL: Best for locomotion (quadrupeds, humanoids)
2. Stable-Baselines3: General purpose (PPO, SAC, TD3)
3. SKRL: Modular RL library
4. RL-Games: GPU-accelerated (fastest for large-scale)

The IsaacLabTrainer wraps these frameworks so you can switch between them
with a single config change while keeping the same training API.

Level: 3 (Advanced)
Hardware: NVIDIA GPU with 24GB+ VRAM
Runners: isaac-sim.yml (EC2 L40S 48GB) | thor.yml (AGX Thor)

Usage:
    # Train CartPole with RSL-RL (quick verification):
    python samples/06_raytraced_training/train_isaac_lab.py --task cartpole --framework rsl_rl

    # Train with SB3:
    python samples/06_raytraced_training/train_isaac_lab.py --task cartpole --framework sb3

    # Full locomotion training (4096 envs, 2000 iterations):
    python samples/06_raytraced_training/train_isaac_lab.py --task anymal_c_flat --envs 4096 --iterations 2000
"""

from __future__ import annotations

import argparse
import time


def train_rsl_rl(
    task: str = "cartpole",
    num_envs: int = 4096,
    max_iterations: int = 1000,
    device: str = "cuda:0",
    output_dir: str = "outputs/isaac_lab",
) -> dict:
    """Train with RSL-RL framework via strands-robots' IsaacLabTrainer.

    RSL-RL is ETH Zurich's RL library, optimized for locomotion tasks.
    It's the default framework for Isaac Lab's locomotion examples.

    Args:
        task: Isaac Lab task name (cartpole, anymal_c_flat, etc.).
        num_envs: Number of parallel GPU environments.
        max_iterations: Maximum training iterations.
        device: CUDA device.
        output_dir: Directory for checkpoints and logs.

    Returns:
        Training results including final metrics.
    """
    from strands_robots.isaac import IsaacLabTrainer, IsaacLabTrainerConfig

    print(f"\n🏋️ Isaac Lab Training: {task} × {num_envs} envs (RSL-RL)")
    print("=" * 60)

    config = IsaacLabTrainerConfig(
        task=task,
        rl_framework="rsl_rl",
        algorithm="PPO",
        num_envs=num_envs,
        device=device,
        max_iterations=max_iterations,
        output_dir=output_dir,
        seed=42,
        headless=True,
    )

    print(f"  Task:          {config.task}")
    print(f"  Framework:     {config.rl_framework}")
    print(f"  Algorithm:     {config.algorithm}")
    print(f"  Environments:  {config.num_envs:,}")
    print(f"  Iterations:    {config.max_iterations:,}")
    print(f"  Device:        {config.device}")
    print(f"  Output:        {config.output_dir}")
    print()

    trainer = IsaacLabTrainer(config)

    t0 = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - t0

    print(f"\n{'─' * 60}")
    print(f"  Training time:     {elapsed:.1f}s")
    print(f"  Result:            {result.get('status', 'unknown')}")
    if "final_reward" in result:
        print(f"  Final reward:      {result['final_reward']:.2f}")
    if "checkpoint_path" in result:
        print(f"  Checkpoint:        {result['checkpoint_path']}")
    if "env_steps_per_second" in result:
        print(f"  Env-steps/s:       {result['env_steps_per_second']:,.0f}")
    print(f"{'─' * 60}")

    return result


def train_sb3(
    task: str = "cartpole",
    num_envs: int = 64,
    total_timesteps: int = 1_000_000,
    device: str = "cuda:0",
    output_dir: str = "outputs/isaac_lab_sb3",
) -> dict:
    """Train with Stable-Baselines3 via strands-robots' IsaacLabTrainer.

    SB3 provides well-tested implementations of PPO, SAC, TD3, etc.
    Good for experimentation and smaller-scale training.

    Args:
        task: Isaac Lab task name.
        num_envs: Number of parallel environments (SB3 typically uses fewer).
        total_timesteps: Total environment steps for training.
        device: CUDA device.
        output_dir: Output directory.

    Returns:
        Training results.
    """
    from strands_robots.isaac import IsaacLabTrainer, IsaacLabTrainerConfig

    print(f"\n🏋️ Isaac Lab Training: {task} × {num_envs} envs (SB3)")
    print("=" * 60)

    config = IsaacLabTrainerConfig(
        task=task,
        rl_framework="sb3",
        algorithm="PPO",
        num_envs=num_envs,
        device=device,
        total_timesteps=total_timesteps,
        output_dir=output_dir,
        seed=42,
        headless=True,
    )

    print(f"  Task:          {config.task}")
    print(f"  Framework:     SB3 ({config.algorithm})")
    print(f"  Environments:  {config.num_envs:,}")
    print(f"  Timesteps:     {config.total_timesteps:,}")

    trainer = IsaacLabTrainer(config)

    t0 = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - t0

    print(f"\n  ✅ Training complete in {elapsed:.1f}s")
    return result


def train_newton_gym(
    robot_name: str = "so100",
    task: str = "reach the target position",
    num_envs: int = 4096,
    total_steps: int = 100_000,
    solver: str = "featherstone",
    device: str = "cuda:0",
) -> dict:
    """Train with Newton's Gymnasium wrapper (NewtonGymEnv + SB3).

    This is the Newton-native training path: NewtonGymEnv provides a standard
    gym.VectorEnv interface that works with any RL library (SB3, CleanRL, etc.).

    Args:
        robot_name: Robot to train (so100, go2, etc.).
        task: Natural language task description.
        num_envs: Number of parallel GPU environments.
        total_steps: Total training steps.
        solver: Newton solver backend.
        device: CUDA device.

    Returns:
        Training results.
    """
    from strands_robots.newton import NewtonConfig
    from strands_robots.newton.newton_gym_env import NewtonGymEnv

    print(f"\n🏋️ Newton Gym Training: {robot_name} × {num_envs} envs ({solver})")
    print("=" * 60)

    config = NewtonConfig(num_envs=num_envs, solver=solver, device=device)
    env = NewtonGymEnv(
        robot_name=robot_name,
        task=task,
        config=config,
    )

    print(f"  Robot:         {robot_name}")
    print(f"  Task:          {task}")
    print(f"  Environments:  {num_envs:,}")
    print(f"  Solver:        {solver}")
    print(f"  Obs space:     {env.observation_space}")
    print(f"  Act space:     {env.action_space}")

    # Training loop (simplified — real training uses SB3 or CleanRL)
    obs, info = env.reset()
    total_reward = 0
    t0 = time.perf_counter()

    for step in range(total_steps):
        # Random actions for demonstration — replace with RL policy
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)

        if hasattr(reward, "sum"):
            total_reward += float(reward.sum())
        else:
            total_reward += float(reward)

        # Reset terminated environments
        if hasattr(terminated, "any") and terminated.any():
            obs, info = env.reset()
        elif isinstance(terminated, bool) and terminated:
            obs, info = env.reset()

        if step % 10000 == 0 and step > 0:
            elapsed = time.perf_counter() - t0
            throughput = int(step * num_envs / elapsed)
            print(f"  Step {step:>7,}: reward={total_reward / (step + 1):.3f}, throughput={throughput:,} steps/s")

    elapsed = time.perf_counter() - t0
    throughput = int(total_steps * num_envs / elapsed)

    env.close()

    print("\n  ✅ Training complete")
    print(f"     Total reward: {total_reward:,.1f}")
    print(f"     Throughput:   {throughput:,} env-steps/s")
    print(f"     Elapsed:      {elapsed:.1f}s")

    return {
        "robot": robot_name,
        "solver": solver,
        "total_reward": total_reward,
        "throughput": throughput,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sample 06: Isaac Lab RL Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="cartpole", help="Isaac Lab task (default: cartpole)")
    parser.add_argument("--framework", default="rsl_rl", choices=["rsl_rl", "sb3", "newton_gym"], help="RL framework")
    parser.add_argument("--envs", type=int, default=4096, help="Parallel environments (default: 4096)")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations (default: 1000)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--output", default="outputs/isaac_lab", help="Output directory")

    args = parser.parse_args()

    if args.framework == "rsl_rl":
        train_rsl_rl(
            task=args.task,
            num_envs=args.envs,
            max_iterations=args.iterations,
            device=args.device,
            output_dir=args.output,
        )
    elif args.framework == "sb3":
        train_sb3(
            task=args.task,
            num_envs=min(args.envs, 64),  # SB3 typically uses fewer envs
            total_timesteps=args.iterations * args.envs,
            device=args.device,
            output_dir=args.output,
        )
    elif args.framework == "newton_gym":
        train_newton_gym(
            robot_name="so100",
            num_envs=args.envs,
            total_steps=args.iterations * 100,
            device=args.device,
        )


if __name__ == "__main__":
    main()
