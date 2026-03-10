#!/usr/bin/env python3
"""Sample 04-C: Evaluate a trained model.

Loads a saved PPO / SAC model, runs evaluation episodes,
and prints per-episode and aggregate metrics.

Usage:
    python evaluate.py --model-path ./ppo_so100/final_model
    python evaluate.py --model-path ./ppo_so100/final_model --episodes 20

Requirements:
    pip install strands-robots[sim] gymnasium stable-baselines3
"""

from __future__ import annotations

import argparse

import numpy as np

from strands_robots import StrandsSimEnv


def load_model(path: str, algorithm: str = "ppo"):
    """Load a stable-baselines3 model from *path*."""
    from stable_baselines3 import PPO, SAC

    cls = PPO if algorithm == "ppo" else SAC
    return cls.load(path)


def evaluate(
    model,
    env: StrandsSimEnv,
    num_episodes: int = 10,
) -> dict:
    """Run *num_episodes* and collect statistics."""
    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        lengths.append(steps)
        successes.append(info.get("is_success", False))

        print(
            f"  Episode {ep + 1:3d} | "
            f"reward {ep_reward:+8.2f} | "
            f"steps {steps:4d} | "
            f"success {info.get('is_success', False)}"
        )

    return {
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)) * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-path", required=True, help="Path to .zip model")
    parser.add_argument("--robot", default="so100")
    parser.add_argument("--task", default="reach a target position")
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print(f"📊 Evaluating {args.algorithm.upper()} — {args.model_path}")
    print("=" * 60)

    env = StrandsSimEnv(
        robot_name=args.robot,
        task=args.task,
        render_mode=None,
        max_episode_steps=500,
    )

    model = load_model(args.model_path, args.algorithm)
    stats = evaluate(model, env, num_episodes=args.episodes)

    env.close()

    print("\n" + "=" * 60)
    print("📈 Aggregate Results")
    print(f"   Mean reward   : {stats['mean_reward']:+.2f} ± {stats['std_reward']:.2f}")
    print(f"   Mean length   : {stats['mean_length']:.0f} steps")
    print(f"   Success rate  : {stats['success_rate']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
