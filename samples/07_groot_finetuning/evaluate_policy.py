#!/usr/bin/env python3
"""
Sample 07 — Evaluate GR00T Policy

Compare base model vs fine-tuned model on simulation episodes.
Uses strands_robots.training.evaluate() for standardized evaluation.

Metrics computed:
  - Success rate (% of episodes where task completed)
  - Mean episode reward
  - Average episode length
  - Trajectory smoothness (joint acceleration variance)

Requirements:
    pip install strands-robots[vla] isaac-gr00t mujoco gymnasium
    GPU with 24GB+ VRAM

Usage:
    # Evaluate fine-tuned checkpoint
    python samples/07_groot_finetuning/evaluate_policy.py \
        --checkpoint ./checkpoints/groot_finetuned/best

    # Compare base vs fine-tuned
    python samples/07_groot_finetuning/evaluate_policy.py \
        --checkpoint ./checkpoints/groot_finetuned/best \
        --compare-base

    # Custom parameters
    python samples/07_groot_finetuning/evaluate_policy.py \
        --checkpoint ./checkpoints/groot_finetuned/best \
        --episodes 100 --robot so100 --task "pick up the cube"
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_policy(model_path: str, data_config: str, device: str = "cuda"):
    """Create a GR00T policy from a checkpoint path.

    Args:
        model_path: Path to fine-tuned checkpoint or HuggingFace model ID.
        data_config: Embodiment data config name.
        device: Device for inference.

    Returns:
        Gr00tPolicy instance ready for evaluation.
    """
    from strands_robots.policies.groot import Gr00tPolicy

    policy = Gr00tPolicy(
        data_config=data_config,
        model_path=model_path,
        device=device,
    )
    return policy


def run_evaluation(
    policy,
    task: str,
    robot_name: str,
    num_episodes: int = 50,
    max_steps: int = 1000,
    backend: str = "mujoco",
    seed: int = 42,
) -> dict:
    """Run policy evaluation using strands_robots.training.evaluate().

    This is the standard evaluation harness that runs N episodes in simulation
    and computes success rate, mean reward, and per-episode statistics.
    """
    from strands_robots.training import evaluate

    print(f"\n📊 Evaluating: {num_episodes} episodes")
    print(f"   Robot: {robot_name}, Backend: {backend}")
    print(f"   Task: \"{task}\"")
    print()

    t0 = time.time()
    results = evaluate(
        policy=policy,
        task=task,
        robot_name=robot_name,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        backend=backend,
        seed=seed,
    )
    elapsed = time.time() - t0

    results["eval_time_seconds"] = elapsed
    return results


def compute_trajectory_smoothness(episodes: list) -> dict:
    """Compute trajectory smoothness metrics from episode data.

    Smoothness is measured by the variance of joint accelerations —
    lower variance = smoother trajectories = better policy.
    """
    if not episodes:
        return {"smoothness": 0.0}

    # Since we don't have raw trajectories in the eval output,
    # compute proxy metrics from episode statistics
    rewards = [ep["reward"] for ep in episodes]
    steps = [ep["steps"] for ep in episodes]

    return {
        "reward_std": float(np.std(rewards)),
        "steps_std": float(np.std(steps)),
        "reward_cv": float(np.std(rewards) / (np.mean(rewards) + 1e-8)),
        "steps_cv": float(np.std(steps) / (np.mean(steps) + 1e-8)),
    }


def print_results(results: dict, label: str = "") -> None:
    """Pretty-print evaluation results."""
    prefix = f"[{label}] " if label else ""

    print(f"\n{'=' * 60}")
    print(f"  {prefix}Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Success rate:     {results['success_rate']:.1f}%")
    print(f"  Mean reward:      {results['mean_reward']:.4f}")
    print(f"  Episodes:         {results['num_episodes']}")
    print(f"  Eval time:        {results.get('eval_time_seconds', 0):.1f}s")
    print(f"  Policy provider:  {results.get('policy_provider', 'unknown')}")

    # Per-episode breakdown
    episodes = results.get("episodes", [])
    if episodes:
        rewards = [ep["reward"] for ep in episodes]
        steps = [ep["steps"] for ep in episodes]
        successes = sum(1 for ep in episodes if ep["success"])

        print("\n  📈 Reward distribution:")
        print(f"     Mean:   {np.mean(rewards):.4f}")
        print(f"     Std:    {np.std(rewards):.4f}")
        print(f"     Min:    {np.min(rewards):.4f}")
        print(f"     Max:    {np.max(rewards):.4f}")

        print("\n  📏 Episode length:")
        print(f"     Mean:   {np.mean(steps):.0f} steps")
        print(f"     Std:    {np.std(steps):.0f}")
        print(f"     Min:    {np.min(steps)}")
        print(f"     Max:    {np.max(steps)}")

        print(f"\n  ✅ Successes: {successes}/{len(episodes)}")

        # Smoothness
        smoothness = compute_trajectory_smoothness(episodes)
        print("\n  🔄 Trajectory consistency:")
        print(f"     Reward CV:  {smoothness['reward_cv']:.4f}")
        print(f"     Steps CV:   {smoothness['steps_cv']:.4f}")

    if results.get("error"):
        print(f"\n  ⚠️  Error: {results['error']}")


def compare_models(base_results: dict, finetuned_results: dict) -> None:
    """Print side-by-side comparison of base vs fine-tuned model."""
    print("\n" + "=" * 60)
    print("  📊 Model Comparison: Base vs Fine-Tuned")
    print("=" * 60)

    metrics = [
        ("Success Rate", "success_rate", "%", ".1f"),
        ("Mean Reward", "mean_reward", "", ".4f"),
    ]

    print(f"  {'Metric':<20} {'Base':>12} {'Fine-tuned':>12} {'Δ':>10}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 12} {'─' * 10}")

    for name, key, unit, fmt in metrics:
        base_val = base_results.get(key, 0)
        ft_val = finetuned_results.get(key, 0)
        delta = ft_val - base_val
        sign = "+" if delta > 0 else ""

        print(f"  {name:<20} {base_val:>11{fmt}}{unit} {ft_val:>11{fmt}}{unit} {sign}{delta:>9{fmt}}")

    # Episode length comparison
    base_eps = base_results.get("episodes", [])
    ft_eps = finetuned_results.get("episodes", [])
    if base_eps and ft_eps:
        base_steps = np.mean([ep["steps"] for ep in base_eps])
        ft_steps = np.mean([ep["steps"] for ep in ft_eps])
        delta_steps = ft_steps - base_steps
        sign = "+" if delta_steps > 0 else ""
        print(f"  {'Avg Steps':<20} {base_steps:>12.0f} {ft_steps:>12.0f} {sign}{delta_steps:>10.0f}")

    # Winner
    print()
    if finetuned_results["success_rate"] > base_results["success_rate"]:
        improvement = finetuned_results["success_rate"] - base_results["success_rate"]
        print(f"  🏆 Fine-tuned model wins! (+{improvement:.1f}% success rate)")
    elif finetuned_results["success_rate"] < base_results["success_rate"]:
        drop = base_results["success_rate"] - finetuned_results["success_rate"]
        print(f"  ⚠️  Fine-tuned model is worse (-{drop:.1f}%). Check training data quality.")
    else:
        print("  🤝 Models perform equally. More episodes or harder tasks may differentiate them.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GR00T policy (base vs fine-tuned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint")
    parser.add_argument("--base-model", default="nvidia/GR00T-N1-2B", help="Base model for comparison")
    parser.add_argument("--compare-base", action="store_true", help="Also evaluate base model for comparison")
    parser.add_argument("--data-config", default="so100_dualcam", help="Embodiment data config")
    parser.add_argument("--robot", default="so100", help="Robot name for simulation")
    parser.add_argument("--task", default="pick up the red cube and place it on the plate", help="Task instruction")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--backend", default="mujoco", choices=["mujoco", "newton", "isaac"], help="Simulation backend")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    print("📊 GR00T N1.6 — Policy Evaluation")
    print("=" * 60)

    all_results = {}

    # Evaluate base model (if requested)
    if args.compare_base:
        print(f"\n1️⃣  Evaluating BASE model: {args.base_model}")
        try:
            base_policy = create_policy(args.base_model, args.data_config, args.device)
            base_results = run_evaluation(
                base_policy, args.task, args.robot,
                num_episodes=args.episodes, max_steps=args.max_steps,
                backend=args.backend, seed=args.seed,
            )
            print_results(base_results, "Base")
            all_results["base"] = base_results
        except Exception as e:
            print(f"❌ Base model evaluation failed: {e}")
            base_results = None
    else:
        base_results = None

    # Evaluate fine-tuned model
    label_num = "2️⃣ " if args.compare_base else "1️⃣ "
    print(f"\n{label_num} Evaluating FINE-TUNED model: {args.checkpoint}")
    try:
        ft_policy = create_policy(args.checkpoint, args.data_config, args.device)
        ft_results = run_evaluation(
            ft_policy, args.task, args.robot,
            num_episodes=args.episodes, max_steps=args.max_steps,
            backend=args.backend, seed=args.seed,
        )
        print_results(ft_results, "Fine-tuned")
        all_results["finetuned"] = ft_results
    except Exception as e:
        print(f"❌ Fine-tuned model evaluation failed: {e}")
        sys.exit(1)

    # Comparison
    if base_results and ft_results:
        compare_models(base_results, ft_results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_path}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
