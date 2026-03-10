#!/usr/bin/env python3
"""
Cosmos Transfer 2.5 Post-Training Script

Post-trains Cosmos Transfer 2.5 for sim-to-real visual augmentation
with ControlNet conditioning (depth, edge, segmentation).

Usage:
    python scripts/training/train_cosmos_transfer.py \
        --dataset /data/sim_real_pairs \
        --control-type depth \
        --mode sim2real \
        --steps 5000

    # Control-only fine-tuning (lower VRAM, faster)
    python scripts/training/train_cosmos_transfer.py \
        --dataset /data/sim_real_pairs \
        --mode control_finetuning \
        --control-type edge \
        --steps 2000

    # Multi-GPU on Thor (132GB)
    python scripts/training/train_cosmos_transfer.py \
        --dataset /data/sim_real_pairs \
        --num-gpus 2 \
        --steps 10000
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Cosmos Transfer 2.5 Post-Training")
    parser.add_argument("--model", default="nvidia/Cosmos-Transfer2-7B",
                        help="Base model path or HuggingFace ID")
    parser.add_argument("--dataset", required=True,
                        help="Path to sim/real paired training data")
    parser.add_argument("--control-type", default="depth",
                        choices=["depth", "edge", "seg", "vis"],
                        help="Control signal type for ControlNet conditioning")
    parser.add_argument("--mode", default="sim2real",
                        choices=["sim2real", "domain_adaptation", "control_finetuning"],
                        help="Training mode")
    parser.add_argument("--output", default="./cosmos_transfer_posttrained",
                        help="Output directory for checkpoints")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Maximum training steps")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--control-weight", type=float, default=1.0,
                        help="Control signal weight")
    parser.add_argument("--guidance", type=float, default=3.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--resolution", default="720",
                        choices=["480", "720", "1080"],
                        help="Output video resolution")
    parser.add_argument("--robot-variant", default=None,
                        help="Robot-specific multiview variant (e.g., multiview-gr1-depth)")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="Freeze backbone weights (default: True)")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                        help="Unfreeze backbone (requires ~80GB+ VRAM)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info(f"🔄 Cosmos Transfer Post-Training: {args.model} mode={args.mode} control={args.control_type}")

    from strands_robots.training import TrainConfig, create_trainer

    config = TrainConfig(
        max_steps=args.steps,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        output_dir=args.output,
    )

    trainer = create_trainer(
        "cosmos_transfer",
        base_model_path=args.model,
        dataset_path=args.dataset,
        control_type=args.control_type,
        mode=args.mode,
        config=config,
        control_weight=args.control_weight,
        guidance=args.guidance,
        output_resolution=args.resolution,
        robot_variant=args.robot_variant,
        freeze_backbone=args.freeze_backbone,
    )

    result = trainer.train()
    logger.info(f"✅ {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
