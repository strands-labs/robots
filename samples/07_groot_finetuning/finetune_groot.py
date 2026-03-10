#!/usr/bin/env python3
"""
Sample 07 — GR00T N1.6 Fine-Tuning Pipeline

Fine-tune GR00T N1.6 on custom robot data using the strands-robots training API.
Supports both full fine-tuning and component-selective training.

The pipeline:
  1. Load base model from HuggingFace
  2. Load dataset (LeRobot v3 format from Sample 05)
  3. Configure which components to train
  4. Run training with progress logging
  5. Save checkpoint

Requirements:
    pip install strands-robots[vla] isaac-gr00t
    GPU with 24GB+ VRAM (46GB+ for full fine-tune)

Usage:
    # Default: fine-tune projector + diffusion head on SO-100 data
    python samples/07_groot_finetuning/finetune_groot.py \
        --dataset ./datasets/my_pick_and_place

    # Full fine-tune (including LLM — needs more VRAM)
    python samples/07_groot_finetuning/finetune_groot.py \
        --dataset ./datasets/my_pick_and_place \
        --tune-llm --tune-visual

    # Use a YAML config file
    python samples/07_groot_finetuning/finetune_groot.py \
        --config samples/07_groot_finetuning/configs/finetune_so100.yaml
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        print("❌ PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"📄 Loaded config from {config_path}")
    return config


def create_trainer_from_args(args) -> "Gr00tTrainer":  # noqa: F821
    """Create a GR00T trainer from command-line arguments.

    Uses strands_robots.training.create_trainer() which returns a Gr00tTrainer
    configured with the Isaac-GR00T fine-tuning pipeline.
    """
    from strands_robots.training import TrainConfig, create_trainer

    # Build TrainConfig from args
    train_config = TrainConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_gpus=args.num_gpus,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        use_wandb=args.wandb,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
    )

    # Create the GR00T trainer
    trainer = create_trainer(
        "groot",
        base_model_path=args.model,
        dataset_path=args.dataset,
        embodiment_tag=args.embodiment_tag,
        data_config=args.data_config,
        tune_llm=args.tune_llm,
        tune_visual=args.tune_visual,
        tune_projector=args.tune_projector,
        tune_diffusion_model=args.tune_diffusion,
        config=train_config,
    )

    return trainer


def create_trainer_from_yaml(config: dict) -> "Gr00tTrainer":  # noqa: F821
    """Create a GR00T trainer from a YAML config dictionary."""
    from strands_robots.training import TrainConfig, create_trainer

    # Extract training config
    training = config.get("training", {})
    train_config = TrainConfig(
        dataset_path=config.get("dataset_path", ""),
        output_dir=config.get("output_dir", "./checkpoints/groot_finetuned"),
        max_steps=training.get("max_steps", 10000),
        batch_size=training.get("batch_size", 16),
        learning_rate=training.get("learning_rate", 1e-4),
        weight_decay=training.get("weight_decay", 1e-5),
        warmup_ratio=training.get("warmup_ratio", 0.05),
        num_gpus=training.get("num_gpus", 1),
        save_steps=training.get("save_steps", 1000),
        save_total_limit=training.get("save_total_limit", 5),
        use_wandb=training.get("use_wandb", False),
        dataloader_num_workers=training.get("num_workers", 2),
        seed=training.get("seed", 42),
    )

    # Extract component tuning flags
    components = config.get("components", {})

    trainer = create_trainer(
        "groot",
        base_model_path=config.get("base_model_path", "nvidia/GR00T-N1-2B"),
        dataset_path=config.get("dataset_path", ""),
        embodiment_tag=config.get("embodiment_tag", "new_embodiment"),
        data_config=config.get("data_config"),
        tune_llm=components.get("tune_llm", False),
        tune_visual=components.get("tune_visual", False),
        tune_projector=components.get("tune_projector", True),
        tune_diffusion_model=components.get("tune_diffusion_model", True),
        config=train_config,
    )

    return trainer


def print_training_plan(trainer, args_or_config) -> None:
    """Display the training plan before starting."""
    print("\n📋 Training Plan:")
    print("=" * 60)
    print(f"  Base model:     {trainer.base_model_path}")
    print(f"  Dataset:        {trainer.dataset_path}")
    print(f"  Embodiment:     {trainer.embodiment_tag}")
    if trainer.data_config:
        print(f"  Data config:    {trainer.data_config}")
    print(f"  Output:         {trainer.config.output_dir}")
    print()
    print("  🔧 Components to train:")
    print(f"     LLM (Qwen3):          {'✅ YES' if trainer.tune_llm else '❌ Frozen'}")
    print(f"     Vision (Eagle):        {'✅ YES' if trainer.tune_visual else '❌ Frozen'}")
    print(f"     Projector:             {'✅ YES' if trainer.tune_projector else '❌ Frozen'}")
    print(f"     Diffusion head:        {'✅ YES' if trainer.tune_diffusion_model else '❌ Frozen'}")
    print()
    print("  📊 Hyperparameters:")
    print(f"     Max steps:      {trainer.config.max_steps}")
    print(f"     Batch size:     {trainer.config.batch_size}")
    print(f"     Learning rate:  {trainer.config.learning_rate}")
    print(f"     Weight decay:   {trainer.config.weight_decay}")
    print(f"     Warmup ratio:   {trainer.config.warmup_ratio}")
    print(f"     GPUs:           {trainer.config.num_gpus}")
    print(f"     Save every:     {trainer.config.save_steps} steps")
    print(f"     Seed:           {trainer.config.seed}")

    # Estimate VRAM usage
    if trainer.tune_llm and trainer.tune_visual:
        vram_est = "~60-80GB (full fine-tune)"
    elif trainer.tune_llm:
        vram_est = "~40-50GB (LLM + projector + diffusion)"
    else:
        vram_est = "~20-30GB (projector + diffusion only)"
    print(f"\n  💾 Estimated VRAM: {vram_est}")


def run_training(trainer) -> dict:
    """Execute the training pipeline and return results."""
    print("\n🚀 Starting GR00T N1.6 fine-tuning...")
    print("   (This calls Isaac-GR00T's launch_finetune internally)")
    print()

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    # Display results
    status = result.get("status", "unknown")
    if status == "completed":
        print(f"\n✅ Training completed in {elapsed / 60:.1f} minutes")
        print(f"   Output directory: {result.get('output_dir')}")
    else:
        print(f"\n❌ Training failed (returncode={result.get('returncode')})")
        print(f"   Elapsed: {elapsed / 60:.1f} minutes")

    # Save training metadata
    result["elapsed_seconds"] = elapsed
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    output_dir = Path(result.get("output_dir", "./checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"   Metadata saved to: {meta_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GR00T N1.6 on custom robot data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Component Training Modes:
  Default (fastest):     --tune-projector --tune-diffusion
  + Language model:      --tune-llm --tune-projector --tune-diffusion
  Full fine-tune:        --tune-llm --tune-visual --tune-projector --tune-diffusion

Typical Training Times (L40S, 10K steps):
  Projector + Diffusion:  ~15 minutes
  + LLM:                  ~45 minutes
  Full:                   ~90 minutes
""",
    )

    # Model configuration
    parser.add_argument("--model", default="nvidia/GR00T-N1-2B", help="Base model path or HF ID")
    parser.add_argument("--dataset", default="", help="Path to LeRobot v3 dataset")
    parser.add_argument("--data-config", default="so100_dualcam", help="Embodiment data config")
    parser.add_argument("--embodiment-tag", default="new_embodiment", help="Embodiment tag")
    parser.add_argument("--output-dir", default="./checkpoints/groot_finetuned", help="Output directory")

    # Component selection
    parser.add_argument("--tune-llm", action="store_true", help="Train the Qwen3 LLM backbone")
    parser.add_argument("--tune-visual", action="store_true", help="Train the Eagle vision encoder")
    parser.add_argument("--tune-projector", action="store_true", default=True, help="Train the projector (default: yes)")
    parser.add_argument("--no-tune-projector", dest="tune_projector", action="store_false")
    parser.add_argument("--tune-diffusion", action="store_true", default=True, help="Train diffusion head (default: yes)")
    parser.add_argument("--no-tune-diffusion", dest="tune_diffusion", action="store_false")

    # Training hyperparameters
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides --max-steps)")
    parser.add_argument("--batch-size", type=int, default=16, help="Global batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--save-total-limit", type=int, default=5, help="Max checkpoints to keep")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")

    # Config file (overrides CLI args)
    parser.add_argument("--config", default=None, help="Path to YAML config file")

    # Dry run
    parser.add_argument("--dry-run", action="store_true", help="Show plan without training")

    args = parser.parse_args()

    print("🧠 GR00T N1.6 — Fine-Tuning Pipeline")
    print("=" * 60)

    # Create trainer
    if args.config:
        config = load_yaml_config(args.config)
        trainer = create_trainer_from_yaml(config)
    else:
        if not args.dataset:
            print("❌ --dataset is required (path to LeRobot v3 dataset)")
            print("   Collect data first: python samples/05_data_collection/record_episodes.py")
            sys.exit(1)
        trainer = create_trainer_from_args(args)

    # Show training plan
    print_training_plan(trainer, args)

    if args.dry_run:
        print("\n🏁 Dry run — no training executed.")
        print("   Remove --dry-run to start training.")
        return

    # Confirm
    print("\n" + "─" * 60)
    response = input("Start training? [Y/n] ").strip().lower()
    if response and response != "y":
        print("Cancelled.")
        return

    # Train
    result = run_training(trainer)

    # Next steps
    if result.get("status") == "completed":
        print("\n🎯 Next steps:")
        print("   1. Evaluate: python samples/07_groot_finetuning/evaluate_policy.py \\")
        print(f"        --checkpoint {result.get('output_dir')}")
        print("   2. Deploy:   python samples/08_sim_to_real/deploy_to_hardware.py \\")
        print(f"        --model {result.get('output_dir')}")


if __name__ == "__main__":
    main()
