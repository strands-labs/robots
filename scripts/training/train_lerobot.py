#!/usr/bin/env python3
"""
LeRobot ACT/Pi0 Training Script

Trains LeRobot policies (ACT, Pi0, SmolVLA, Diffusion) on recorded data.

Usage:
    python scripts/training/train_lerobot.py --policy act --dataset lerobot/so100_wipe
    python scripts/training/train_lerobot.py --policy pi0 --dataset cagataycali/groot-train-data --steps 50000
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="LeRobot Policy Training")
    parser.add_argument("--policy", default="act", choices=["act", "pi0", "smolvla", "diffusion"],
                       help="Policy type")
    parser.add_argument("--dataset", required=True, help="HF dataset repo ID")
    parser.add_argument("--output", default=None, help="Output directory (default: ./{policy}_trained)")
    parser.add_argument("--steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eval-after", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()

    output = args.output or f"./{args.policy}_trained"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info(f"🧠 LeRobot Training: {args.policy} → {output}")
    logger.info(f"   Dataset: {args.dataset}, Steps: {args.steps}, Batch: {args.batch_size}")

    from strands_robots.training import create_trainer
    trainer = create_trainer(
        "lerobot",
        policy_type=args.policy,
        dataset_repo_id=args.dataset,
        output_dir=output,
        max_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    logger.info(f"✅ Training complete in {elapsed:.1f}s: {json.dumps(result, indent=2, default=str)}")

    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "training_meta.json"), "w") as f:
        json.dump({"args": vars(args), "result": result, "elapsed": elapsed}, f, indent=2, default=str)

    if args.eval_after:
        logger.info(f"📊 Evaluating {args.policy}...")
        from strands_robots.policies import create_policy
        from strands_robots.training import evaluate
        policy = create_policy("lerobot_local", pretrained_name_or_path=os.path.join(output, "best"))
        eval_result = evaluate(policy=policy, task="pick up the red cube", robot_name="so100",
                               num_episodes=args.eval_episodes)
        logger.info(f"   Success: {eval_result.get('success_rate', 0):.1f}%")

if __name__ == "__main__":
    main()
