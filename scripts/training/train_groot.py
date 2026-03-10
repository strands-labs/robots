#!/usr/bin/env python3
"""
GR00T N1.6 Fine-Tuning Script

Fine-tunes GR00T N1.6 on recorded demonstration data.
Supports SO-100 dual-cam and other data_configs.

Usage:
    python scripts/training/train_groot.py --dataset cagataycali/groot-train-data --steps 10000
    python scripts/training/train_groot.py --dataset ./local_data --output ./groot_finetuned
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
    parser = argparse.ArgumentParser(description="GR00T N1.6 Fine-Tuning")
    parser.add_argument("--model", default="nvidia/GR00T-N1-2B", help="Base model path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF repo ID")
    parser.add_argument("--embodiment", default="so100", help="Embodiment tag")
    parser.add_argument("--data-config", default="so100_dualcam", help="Data config preset")
    parser.add_argument("--output", default="./groot_finetuned", help="Output directory")
    parser.add_argument("--steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--eval-after", action="store_true", help="Run evaluation after training")
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info(f"🤖 GR00T Fine-Tuning: {args.model} → {args.output}")
    logger.info(f"   Dataset: {args.dataset}, Embodiment: {args.embodiment}")
    logger.info(f"   Steps: {args.steps}, Batch: {args.batch_size}, LR: {args.lr}")

    from strands_robots.training import create_trainer
    trainer = create_trainer(
        "groot",
        base_model_path=args.model,
        dataset_path=args.dataset,
        embodiment_tag=args.embodiment,
        data_config=args.data_config,
        output_dir=args.output,
        max_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    logger.info(f"✅ Training complete in {elapsed:.1f}s")
    logger.info(f"   Result: {json.dumps(result, indent=2, default=str)}")

    # Save training metadata
    meta = {"args": vars(args), "result": result, "elapsed": elapsed}
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if args.eval_after:
        logger.info(f"📊 Running evaluation ({args.eval_episodes} episodes)...")
        from strands_robots.policies import create_policy
        from strands_robots.training import evaluate
        policy = create_policy("groot", model_path=os.path.join(args.output, "best"),
                               data_config=args.data_config)
        for task in ["pick up the red cube", "stack the blue block on green"]:
            eval_result = evaluate(policy=policy, task=task, robot_name="so100",
                                   num_episodes=args.eval_episodes)
            logger.info(f"   {task}: {eval_result.get('success_rate', 0):.1f}%")

if __name__ == "__main__":
    main()
