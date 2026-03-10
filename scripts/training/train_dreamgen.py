#!/usr/bin/env python3
"""
DreamGen IDM Training + VLA Fine-Tuning Script

Runs the DreamGen pipeline: IDM training → VLA fine-tuning.

Usage:
    python scripts/training/train_dreamgen.py --mode idm --dataset /data/demos --steps 20000
    python scripts/training/train_dreamgen.py --mode vla --dataset /data/neural_trajectories
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="DreamGen Training")
    parser.add_argument("--mode", default="idm", choices=["idm", "vla"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--data-config", default="so100")
    parser.add_argument("--model", default="nvidia/GR00T-N1-2B")
    parser.add_argument("--embodiment", default="so100")
    args = parser.parse_args()

    output = args.output or f"./dreamgen_{args.mode}_trained"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info(f"🎥 DreamGen {args.mode.upper()} Training → {output}")

    from strands_robots.training import create_trainer

    if args.mode == "idm":
        trainer = create_trainer("dreamgen_idm",
            dataset_path=args.dataset, data_config=args.data_config,
            output_dir=output, max_steps=args.steps)
    else:
        trainer = create_trainer("dreamgen_vla",
            base_model_path=args.model, dataset_path=args.dataset,
            embodiment_tag=args.embodiment, output_dir=output, max_steps=args.steps)

    result = trainer.train()
    logger.info(f"✅ {json.dumps(result, indent=2, default=str)}")

if __name__ == "__main__":
    main()
