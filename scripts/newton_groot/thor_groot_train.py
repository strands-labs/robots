#!/usr/bin/env python3
"""
🤖 Thor GR00T Training: Fine-tune GR00T N1.6-3B on COSMOS-transferred Dataset
==============================================================================
Issue #204 — Designed for NVIDIA AGX Thor (132GB unified GPU)

Fine-tunes nvidia/GR00T-N1.6-3B on the assembled dataset with COSMOS-transferred
photorealistic videos and original joint state/action data.

Usage:
    python3 scripts/newton_groot/thor_groot_train.py [--max-steps 5000] [--resume]

Prerequisites:
    - Assembled dataset at /home/cagatay/e2e_pipeline/final_dataset/
    - GR00T N1.6 model access (HuggingFace token)
    - ~32-64GB GPU memory for training
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STATUS_FILE = Path("/home/cagatay/e2e_pipeline/status.json")

# Configuration
DATASET_DIR = Path(os.environ.get("THOR_FINAL_DATASET", "/home/cagatay/e2e_pipeline/final_dataset"))
OUTPUT_DIR = Path(os.environ.get("THOR_GROOT_OUTPUT", "/home/cagatay/e2e_pipeline/groot_finetuned"))
BASE_MODEL = "nvidia/GR00T-N1.6-3B"
EMBODIMENT_TAG = "unitree_g1"
BATCH_SIZE = 32
MAX_STEPS = 5000
LEARNING_RATE = 1e-5
SAVE_STEPS = 500
EVAL_STEPS = 250
NUM_DOFS = 43


def update_status(step_name, status, extra=None):
    """Update pipeline status file."""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE) as f:
                data = json.load(f)
        else:
            data = {"pipeline": "issue_204_e2e", "steps": {}}

        step_data = {"status": status, "updated_at": datetime.now(tz=timezone.utc).isoformat()}
        if extra:
            step_data.update(extra)
        data["steps"][step_name] = step_data
        data["current_step"] = step_name

        with open(STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def check_groot_installed():
    """Check if GR00T N1.6 training module is available."""
    try:
        import importlib
        importlib.import_module("gr00t")
        return True
    except ImportError:
        return False


def install_groot():
    """Attempt to install Isaac-GR00T."""
    print("  Attempting to install Isaac-GR00T...")

    # Method 1: pip install
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "isaac-gr00t"],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    # Method 2: Clone from GitHub
    groot_dir = Path("/home/cagatay/Isaac-GR00T")
    if not groot_dir.exists():
        try:
            subprocess.run([
                "git", "clone", "https://github.com/NVIDIA/Isaac-GR00T",
                str(groot_dir)
            ], timeout=120, check=True)
        except Exception:
            pass

    if groot_dir.exists():
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(groot_dir)],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    return False


def create_data_config(dataset_path, output_path):
    """Create GR00T-compatible data configuration."""
    config = {
        "embodiment_tag": EMBODIMENT_TAG,
        "state_dim": NUM_DOFS,
        "action_dim": NUM_DOFS,
        "cameras": ["ego_view"],
        "image_size": [224, 224],  # GR00T resizes internally
        "dataset_path": str(dataset_path),
        "task_description": "Navigate around a room using bipedal locomotion",
        "robot_type": "unitree_g1",
        "action_space": "joint_position",
    }

    config_path = output_path / "data_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def assemble_dataset():
    """Assemble final dataset from sim data + COSMOS transferred videos."""
    print("\n  Assembling final dataset...")

    sim_dataset = Path("/home/cagatay/e2e_pipeline/gr00t_dataset")
    cosmos_output = Path("/home/cagatay/e2e_pipeline/cosmos_transferred")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Copy parquet data
    src_data = sim_dataset / "data"
    dst_data = DATASET_DIR / "data"
    if src_data.exists() and not dst_data.exists():
        shutil.copytree(src_data, dst_data, dirs_exist_ok=True)
        parquet_count = len(list(dst_data.rglob("*.parquet")))
        print(f"    Copied {parquet_count} parquet files")

    # Copy videos (prefer COSMOS-transferred, fallback to sim)
    dst_videos = DATASET_DIR / "videos"
    if not dst_videos.exists() or not any(dst_videos.rglob("*.mp4")):
        if cosmos_output.exists() and any(cosmos_output.rglob("*.mp4")):
            shutil.copytree(cosmos_output, dst_videos, dirs_exist_ok=True)
            video_count = len(list(dst_videos.rglob("*.mp4")))
            print(f"    Copied {video_count} COSMOS-transferred videos")
        else:
            src_videos = sim_dataset / "videos"
            if src_videos.exists():
                shutil.copytree(src_videos, dst_videos, dirs_exist_ok=True)
                video_count = len(list(dst_videos.rglob("*.mp4")))
                print(f"    Copied {video_count} original sim videos")

    # Copy metadata
    src_meta = sim_dataset / "meta"
    dst_meta = DATASET_DIR / "meta"
    if src_meta.exists():
        shutil.copytree(src_meta, dst_meta, dirs_exist_ok=True)

        # Update info.json
        info_path = dst_meta / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            info["cosmos_transferred"] = cosmos_output.exists() and any(cosmos_output.rglob("*.mp4"))
            info["pipeline"] = "Newton GPU Sim → COSMOS Transfer2.5 → GR00T N1.6"
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)

    parquet_count = len(list(DATASET_DIR.rglob("*.parquet")))
    video_count = len(list(DATASET_DIR.rglob("*.mp4")))
    print(f"  ✅ Final dataset: {parquet_count} parquet, {video_count} videos")
    return parquet_count > 0


def train_groot(args):
    """Run GR00T N1.6 fine-tuning."""
    update_status("step5_groot_train", "running", {
        "base_model": BASE_MODEL,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create data config
    config_path = create_data_config(DATASET_DIR, OUTPUT_DIR)
    print(f"  Data config: {config_path}")

    # Method 1: GR00T training pipeline
    cmd = [
        sys.executable, "-m", "gr00t.experiment.launch_finetune",
        "--base_model_path", BASE_MODEL,
        "--dataset_path", str(DATASET_DIR),
        "--embodiment_tag", EMBODIMENT_TAG,
        "--output_dir", str(OUTPUT_DIR),
        "--max_steps", str(args.max_steps),
        "--global_batch_size", str(args.batch_size),
        "--learning_rate", str(args.lr),
        "--save_steps", str(args.save_steps),
        "--num_gpus", "1",
    ]

    if args.resume and (OUTPUT_DIR / "checkpoint-latest").exists():
        cmd.extend(["--resume_from_checkpoint", str(OUTPUT_DIR / "checkpoint-latest")])

    print("\n  Training command:")
    print(f"    {' '.join(cmd)}")
    print(f"\n  Starting GR00T fine-tuning ({args.max_steps} steps)...")

    try:
        result = subprocess.run(cmd, timeout=14400)  # 4 hour timeout

        if result.returncode == 0:
            print("  ✅ GR00T fine-tuning completed!")
            update_status("step5_groot_train", "completed", {
                "duration_s": 0,
                "checkpoint_dir": str(OUTPUT_DIR),
            })
            return True
        else:
            print(f"  ⚠️  GR00T training exited with code {result.returncode}")
            update_status("step5_groot_train", "failed", {"exit_code": result.returncode})
            return False

    except subprocess.TimeoutExpired:
        print("  ⚠️  GR00T training timed out (>4 hours)")
        update_status("step5_groot_train", "timeout")
        return False
    except FileNotFoundError:
        print("  ❌ GR00T training module not found")
        print("  Creating training placeholder...")

        # Create placeholder summary
        summary = {
            "status": "gr00t_module_not_available",
            "base_model": BASE_MODEL,
            "dataset_path": str(DATASET_DIR),
            "embodiment_tag": EMBODIMENT_TAG,
            "planned_config": {
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "save_steps": args.save_steps,
            },
            "note": "GR00T training module not installed. Install Isaac-GR00T to enable training.",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(OUTPUT_DIR / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        update_status("step5_groot_train", "skipped", {"reason": "gr00t_not_installed"})
        return False


def main():
    parser = argparse.ArgumentParser(description="Thor GR00T Training")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--skip-assembly", action="store_true", help="Skip dataset assembly")
    args = parser.parse_args()

    print("=" * 70)
    print("  🤖 Thor GR00T Training — N1.6-3B Fine-Tuning")
    print(f"  Base Model: {BASE_MODEL}")
    print(f"  Dataset:    {DATASET_DIR}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  Steps: {args.max_steps} | Batch: {args.batch_size} | LR: {args.lr}")
    print("=" * 70)

    # Step 4: Assemble dataset (unless skipped)
    if not args.skip_assembly:
        update_status("step4_assemble_dataset", "running")
        if assemble_dataset():
            update_status("step4_assemble_dataset", "completed")
        else:
            print("  ❌ No data available for training!")
            update_status("step4_assemble_dataset", "failed", {"reason": "no_data"})
            return 1

    # Check GR00T installation
    has_groot = check_groot_installed()
    if not has_groot:
        has_groot = install_groot()

    if not has_groot:
        print("\n  ⚠️  GR00T N1.6 not available — generating training config only")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        create_data_config(DATASET_DIR, OUTPUT_DIR)

        summary = {
            "status": "ready_for_training",
            "base_model": BASE_MODEL,
            "dataset_path": str(DATASET_DIR),
            "parquet_files": len(list(DATASET_DIR.rglob("*.parquet"))),
            "video_files": len(list(DATASET_DIR.rglob("*.mp4"))),
            "training_command": (
                f"python3 -m gr00t.experiment.launch_finetune "
                f"--base_model_path {BASE_MODEL} "
                f"--dataset_path {DATASET_DIR} "
                f"--embodiment_tag {EMBODIMENT_TAG} "
                f"--output_dir {OUTPUT_DIR} "
                f"--max_steps {args.max_steps} "
                f"--global_batch_size {args.batch_size} "
                f"--learning_rate {args.lr}"
            ),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(OUTPUT_DIR / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  📄 Training config saved to {OUTPUT_DIR}/training_summary.json")
        print("  To train, install Isaac-GR00T and run the command above.")

        update_status("step5_groot_train", "pending_installation", {
            "note": "GR00T not installed, config saved"
        })
        return 0

    # Step 5: Train GR00T
    return 0 if train_groot(args) else 1


if __name__ == "__main__":
    sys.exit(main())
