#!/usr/bin/env python3
"""
GR00T N1.6-3B Fine-Tuning on Unitree G1 Locomotion Dataset
===========================================================
Thor E2E Pipeline - Step 5

Device: NVIDIA AGX Thor (sm_110, 122GB unified GPU, CUDA 13.0)

Usage:
    python3 /home/cagatay/e2e_pipeline_v2/train_groot.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ─── Paths ───
DATASET_PATH = "/home/cagatay/e2e_pipeline_v2/dataset"
OUTPUT_DIR = "/home/cagatay/e2e_pipeline_v2/groot_finetuned"
STATUS_FILE = "/home/cagatay/e2e_pipeline_v2/status.json"
MODALITY_CONFIG = "/home/cagatay/e2e_pipeline_v2/g1_locomotion_config.py"
BASE_MODEL = "nvidia/GR00T-N1.6-3B"

# ─── Training config ───
MAX_STEPS = 5000
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3
NUM_WORKERS = 2
SHARD_SIZE = 256

def update_status(step_name, status, **kwargs):
    """Update status.json checkpoint"""
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
    except Exception:
        data = {"pipeline": "issue_204_e2e_v2", "steps": {}}

    data["steps"][step_name] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    data["current_step"] = step_name

    with open(STATUS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    start_time = time.time()

    print("=" * 70)
    print("🤖 GR00T N1.6-3B Fine-Tuning on Unitree G1 Locomotion")
    print("=" * 70)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Config:  max_steps={MAX_STEPS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print()

    # ─── Register custom modality config ───
    sys.path.insert(0, str(Path(MODALITY_CONFIG).parent))

    update_status("step5_groot_train", "loading_model")

    # ─── Import GR00T training modules ───
    print("Loading GR00T training modules...")
    from gr00t.configs.base_config import get_default_config
    from gr00t.experiment.experiment import run

    # ─── Build config ───
    print("Building training config...")
    config = get_default_config().load_dict({
        "data": {
            "download_cache": False,
            "video_backend": "ffmpeg",  # torchcodec not available on Thor aarch64
            "datasets": [
                {
                    "dataset_paths": [DATASET_PATH],
                    "mix_ratio": 1.0,
                    "embodiment_tag": "unitree_g1",
                }
            ],
        }
    })
    config.load_config_path = None

    # Explicitly set video backend to ffmpeg
    config.data.video_backend = "ffmpeg"

    # Model config
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.state_dropout_prob = 0.0
    config.model.random_rotation_angle = None
    config.model.color_jitter_params = None
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # Training config
    config.training.start_from_checkpoint = BASE_MODEL
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = BATCH_SIZE
    config.training.dataloader_num_workers = NUM_WORKERS
    config.training.learning_rate = LEARNING_RATE
    config.training.gradient_accumulation_steps = 1
    config.training.output_dir = OUTPUT_DIR
    config.training.save_steps = SAVE_STEPS
    config.training.save_total_limit = SAVE_TOTAL_LIMIT
    config.training.num_gpus = 1
    config.training.use_wandb = False
    config.training.max_steps = MAX_STEPS
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.wandb_project = "finetune-gr00t-n1d6"

    # Data config
    config.data.shard_size = SHARD_SIZE
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = int(1e5)

    print("\n📋 Training Configuration:")
    print(f"  Base model: {BASE_MODEL}")
    print("  Embodiment: unitree_g1")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Save every: {SAVE_STEPS} steps")
    print(f"  Shard size: {SHARD_SIZE}")
    print("  Video backend: ffmpeg")
    print("  Tune projector: True, Tune diffusion: True")
    print("  Tune LLM: False, Tune visual: False")
    print()

    update_status("step5_groot_train", "training_started",
                  max_steps=MAX_STEPS, batch_size=BATCH_SIZE)

    # ─── Launch training ───
    print("🚀 Starting GR00T fine-tuning...")
    print("=" * 70)

    try:
        run(config)

        elapsed = time.time() - start_time
        update_status("step5_groot_train", "completed",
                      duration_s=round(elapsed, 1),
                      output_dir=OUTPUT_DIR)

        print("=" * 70)
        print(f"✅ Training complete! Duration: {elapsed/60:.1f} minutes")
        print(f"   Checkpoint: {OUTPUT_DIR}")

    except Exception as e:
        elapsed = time.time() - start_time
        update_status("step5_groot_train", "error",
                      error=str(e)[:500],
                      duration_s=round(elapsed, 1))
        print(f"\n❌ Training error after {elapsed/60:.1f} minutes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
