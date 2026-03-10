#!/usr/bin/env python3
"""
Cosmos Predict 2.5 Post-Training Script

Post-trains Cosmos Predict for robot policy generation.

Usage:
    python scripts/training/train_cosmos.py --dataset cagataycali/cosmos-isaac-data --steps 5000
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Cosmos Predict 2.5 Post-Training")
    parser.add_argument("--model", default="nvidia/Cosmos-Predict2.5-2B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", default="policy", choices=["policy", "action_conditioned", "video"])
    parser.add_argument("--output", default="./cosmos_posttrained")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info(f"🌌 Cosmos Post-Training: {args.model} mode={args.mode}")

    from strands_robots.training import create_trainer
    trainer = create_trainer(
        "cosmos_predict",
        base_model_path=args.model,
        dataset_path=args.dataset,
        mode=args.mode,
        output_dir=args.output,
        max_steps=args.steps,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
    )

    result = trainer.train()
    logger.info(f"✅ {json.dumps(result, indent=2, default=str)}")

if __name__ == "__main__":
    main()
