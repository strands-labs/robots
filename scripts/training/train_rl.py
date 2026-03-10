#!/usr/bin/env python3
"""
RL Training Script (PPO/SAC) for Newton/MuJoCo

Trains PPO or SAC policies using stable-baselines3.

Usage:
    python scripts/training/train_rl.py --algo ppo --robot unitree_g1 --task "walk forward" --timesteps 10000000
    python scripts/training/train_rl.py --algo sac --robot so100 --task "pick up the red cube" --timesteps 2000000
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="RL Training (PPO/SAC)")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--robot", default="so100")
    parser.add_argument("--task", default="pick up the red cube")
    parser.add_argument("--backend", default="mujoco", choices=["mujoco", "newton"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-after", action="store_true")
    args = parser.parse_args()

    output = args.output or f"./{args.algo}_{args.robot}_{args.backend}"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info(f"⚡ RL Training: {args.algo.upper()} on {args.robot} ({args.backend})")
    logger.info(f"   Task: {args.task}, Timesteps: {args.timesteps:,}")

    from strands_robots.rl_trainer import create_rl_trainer
    trainer = create_rl_trainer(
        algorithm=args.algo,
        env_config={
            "robot_name": args.robot,
            "task": args.task,
            "backend": args.backend,
            "num_envs": args.num_envs,
        },
        total_timesteps=args.timesteps,
        output_dir=output,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    result = trainer.train()
    logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")

    if args.eval_after and result.get("status") == "success":
        logger.info("📊 Evaluating trained policy...")
        eval_result = trainer.evaluate(num_episodes=20)
        logger.info(f"   Success: {eval_result.get('success_rate', 0):.1f}%, Reward: {eval_result.get('mean_reward', 0):.2f}")

if __name__ == "__main__":
    main()
