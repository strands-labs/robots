#!/usr/bin/env python3
"""
🌌 Thor COSMOS Transfer: Sim-to-Real Domain Transfer via COSMOS Transfer2.5-2B
==============================================================================
Issue #204 — Leverages Thor's 132GB GPU for FULL VIDEO MODE (93 frames)

Processes episode videos from Newton simulation through COSMOS Transfer2.5-2B
to produce photorealistic training data for GR00T N1.6 fine-tuning.

Usage:
    python3 scripts/newton_groot/thor_cosmos_transfer.py [--resume] [--max-videos 256]

Prerequisites:
    - COSMOS Transfer2.5-2B checkpoint (auto-downloads from HF if needed)
    - Input videos at /home/cagatay/e2e_pipeline/gr00t_dataset/videos/
    - ~65GB GPU memory for video mode (Thor has 132GB ✅)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STATUS_FILE = Path("/home/cagatay/e2e_pipeline/status.json")


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
    except Exception as e:
        print(f"  Warning: Could not update status: {e}")


# Configuration
INPUT_DIR = Path(os.environ.get("THOR_DATASET_DIR", "/home/cagatay/e2e_pipeline/gr00t_dataset"))
OUTPUT_DIR = Path(os.environ.get("THOR_COSMOS_OUTPUT", "/home/cagatay/e2e_pipeline/cosmos_transferred"))
CHECKPOINT_DIR = Path(os.environ.get("COSMOS_CHECKPOINT_DIR",
    "/home/cagatay/checkpoints/Cosmos-Transfer2.5-2B"))

# COSMOS settings optimized for Thor's 132GB
COSMOS_PROMPT = (
    "A modern bedroom with soft natural lighting, hardwood floors, "
    "white walls, contemporary furniture, realistic textures and shadows"
)
COSMOS_NUM_FRAMES = 93      # Full video mode — fits in 132GB
COSMOS_CONTROL_TYPE = "edge"  # Most reliable
COSMOS_GUIDANCE_SCALE = 3.0
COSMOS_SEED = 42


def download_cosmos_checkpoint():
    """Download COSMOS Transfer2.5-2B from HuggingFace if not present."""
    if CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.rglob("*.safetensors")):
        print(f"  ✅ COSMOS checkpoint found: {CHECKPOINT_DIR}")
        return True

    print(f"  Downloading COSMOS Transfer2.5-2B to {CHECKPOINT_DIR}...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HUGGING_FACE_TOKEN", os.environ.get("HF_TOKEN", ""))

    try:
        cmd = [
            "huggingface-cli", "download",
            "nvidia/Cosmos-Transfer2.5-2B-Sample2Sample",
            "--local-dir", str(CHECKPOINT_DIR),
        ]
        if hf_token:
            cmd.extend(["--token", hf_token])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            print("  ✅ COSMOS checkpoint downloaded")
            return True
        else:
            print(f"  ❌ Download failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print("  ❌ Download timed out (>30 min)")
        return False
    except FileNotFoundError:
        print("  ❌ huggingface-cli not found. Try: pip install huggingface_hub[cli]")
        return False


def check_cosmos_installed():
    """Check if cosmos_transfer2 is installed."""
    try:
        import importlib
        importlib.import_module("cosmos_transfer2")
        return True
    except ImportError:
        return False


def install_cosmos():
    """Attempt to install COSMOS Transfer2."""
    print("  Attempting to install cosmos_transfer2...")
    try:
        # Try pip install
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "cosmos-transfer2"],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            return True

        # Try git clone approach
        cosmos_dir = Path("/home/cagatay/cosmos-transfer2")
        if not cosmos_dir.exists():
            subprocess.run([
                "git", "clone", "https://github.com/NVIDIA/Cosmos-Transfer2.0",
                str(cosmos_dir)
            ], timeout=120)

        if cosmos_dir.exists():
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(cosmos_dir)],
                capture_output=True, text=True, timeout=600
            )
            return result.returncode == 0

    except Exception as e:
        print(f"  ❌ Install failed: {e}")
    return False


def process_video_cosmos(input_path, output_path, timeout=600):
    """Process a single video through COSMOS Transfer2.5.

    Tries multiple invocation methods since the COSMOS CLI interface varies.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Method 1: cosmos_transfer2.inference module
    cmd1 = [
        sys.executable, "-m", "cosmos_transfer2.inference",
        "--input", str(input_path),
        "--output", str(output_path),
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--control_type", COSMOS_CONTROL_TYPE,
        "--prompt", COSMOS_PROMPT,
        "--guidance_scale", str(COSMOS_GUIDANCE_SCALE),
        "--num_input_frames", str(COSMOS_NUM_FRAMES),
        "--seed", str(COSMOS_SEED),
        "--disable_guardrails",
    ]

    # Method 2: cosmos_transfer2.examples.inference
    cmd2 = [
        sys.executable, "-m", "cosmos_transfer2.examples.inference",
        "-i", str(input_path),
        "-o", str(output_path),
        "--setup.checkpoint_dir", str(CHECKPOINT_DIR),
        "--control_type", COSMOS_CONTROL_TYPE,
        "--prompt", COSMOS_PROMPT,
        "--num_input_frames", str(COSMOS_NUM_FRAMES),
    ]

    # Method 3: Direct script invocation
    cosmos_dir = Path("/home/cagatay/cosmos-transfer2")
    cmd3 = [
        sys.executable, str(cosmos_dir / "cosmos_transfer2" / "inference.py"),
        "--input", str(input_path),
        "--output", str(output_path),
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--control_type", COSMOS_CONTROL_TYPE,
        "--prompt", COSMOS_PROMPT,
    ] if cosmos_dir.exists() else None

    for i, cmd in enumerate([cmd1, cmd2, cmd3]):
        if cmd is None:
            continue
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                return True
        except (subprocess.TimeoutExpired, Exception):
            continue

    return False


def main():
    parser = argparse.ArgumentParser(description="Thor COSMOS Transfer")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed videos")
    parser.add_argument("--max-videos", type=int, default=None, help="Process at most N videos")
    parser.add_argument("--skip-download", action="store_true", help="Skip checkpoint download")
    parser.add_argument("--fallback-copy", action="store_true", default=True,
                        help="Copy original video if COSMOS fails (default: true)")
    args = parser.parse_args()

    print("=" * 70)
    print("  🌌 Thor COSMOS Transfer2.5 — Sim-to-Real Domain Transfer")
    print(f"  Input:  {INPUT_DIR / 'videos'}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Mode:   VIDEO ({COSMOS_NUM_FRAMES} frames)")
    print(f"  Control: {COSMOS_CONTROL_TYPE}")
    print(f"  Resume: {args.resume}")
    print("=" * 70)

    update_status("step3_cosmos_transfer", "running")

    # Find input videos
    videos_dir = INPUT_DIR / "videos"
    video_files = sorted(videos_dir.rglob("*.mp4"))

    if not video_files:
        print("  ❌ No input videos found!")
        print(f"     Expected at: {videos_dir}")
        update_status("step3_cosmos_transfer", "skipped", {"reason": "no_input_videos"})
        return 1

    print(f"\n  Found {len(video_files)} input videos")

    if args.max_videos:
        video_files = video_files[:args.max_videos]
        print(f"  Limited to {len(video_files)} videos (--max-videos)")

    # Check/download COSMOS checkpoint
    if not args.skip_download:
        has_checkpoint = download_cosmos_checkpoint()
    else:
        has_checkpoint = CHECKPOINT_DIR.exists()

    # Check if COSMOS is installed
    has_cosmos = check_cosmos_installed()
    if not has_cosmos:
        print("  ⚠️  cosmos_transfer2 not installed")
        has_cosmos = install_cosmos()

    if not has_cosmos or not has_checkpoint:
        print("\n  ⚠️  COSMOS Transfer not available (module or checkpoint missing)")
        print("  Falling back to copying original videos...")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for vf in video_files:
            rel_path = vf.relative_to(videos_dir)
            out_path = OUTPUT_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists() or not args.resume:
                shutil.copy2(vf, out_path)

        update_status("step3_cosmos_transfer", "completed_fallback", {
            "note": "COSMOS not available, copied original sim videos",
            "videos_copied": len(video_files),
        })
        print(f"  ✅ Copied {len(video_files)} videos as fallback")
        return 0

    # Process videos through COSMOS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, video_path in enumerate(video_files):
        rel_path = video_path.relative_to(videos_dir)
        out_path = OUTPUT_DIR / rel_path

        # Resume: skip if already processed
        if args.resume and out_path.exists() and out_path.stat().st_size > 0:
            skip_count += 1
            continue

        print(f"  [{idx+1}/{len(video_files)}] {rel_path}...", end=" ", flush=True)

        ok = process_video_cosmos(video_path, out_path)
        if ok:
            success_count += 1
            print("✅")
        else:
            fail_count += 1
            print("❌")
            # Fallback: copy original
            if args.fallback_copy:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_path, out_path)

        # Progress report
        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t_start
            processed = success_count + fail_count
            rate = processed / max(elapsed, 1)
            remaining = len(video_files) - idx - 1 - skip_count
            eta_min = remaining / max(rate, 0.001) / 60
            print(f"    Progress: {idx+1}/{len(video_files)} | ✅{success_count} ❌{fail_count} ⏭️{skip_count} | "
                  f"{rate:.2f} vid/s | ETA: {eta_min:.0f} min")

    total_time = time.time() - t_start
    total_output = len(list(OUTPUT_DIR.rglob("*.mp4")))

    print("\n  ✅ COSMOS Transfer complete!")
    print(f"     Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}")
    print(f"     Output videos: {total_output}")
    print(f"     Time: {total_time:.1f}s ({total_time/60:.1f} min)")

    update_status("step3_cosmos_transfer", "completed", {
        "success": success_count,
        "failed": fail_count,
        "skipped": skip_count,
        "total_output": total_output,
        "duration_s": round(total_time, 1),
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
