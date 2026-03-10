#!/usr/bin/env python3
"""
Validate strands-kitchen-stereo4d dataset completeness and structural integrity.

This script ensures all 6 modalities (stereo_left, stereo_right, depth, disparity,
point_clouds, audio) are populated with correctly-shaped data before HuggingFace upload.

Usage:
    python scripts/validate_dataset.py [dataset_path]

    # Default path: datasets/strands-kitchen-stereo4d
    python scripts/validate_dataset.py

    # Custom path:
    python scripts/validate_dataset.py /path/to/my/dataset
"""

import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_SUBDIRS = ["stereo_left", "stereo_right", "depth", "disparity", "point_clouds", "audio"]


def validate(dataset_path: str) -> bool:
    """Validate dataset structure, completeness, and data integrity.

    Returns True if valid, raises AssertionError with details on failure.
    """
    root = Path(dataset_path)
    errors = []
    warnings = []

    # ─── Top-level files ───
    if not (root / "metadata.json").exists():
        errors.append("Missing metadata.json at dataset root")
    if not (root / "README.md").exists():
        errors.append("Missing README.md at dataset root")

    if errors:
        for e in errors:
            print(f"❌ {e}")
        return False

    meta = json.loads((root / "metadata.json").read_text())
    episodes = sorted(root.glob("episode_*"))

    if len(episodes) == 0:
        print("❌ No episode directories found")
        return False

    if len(episodes) != meta.get("num_episodes", -1):
        warnings.append(
            f"Episode count mismatch: found {len(episodes)} dirs, "
            f"metadata says {meta.get('num_episodes')}"
        )

    print(f"📂 Dataset: {root}")
    print(f"📊 Metadata: {meta.get('num_episodes')} episodes, "
          f"{meta.get('total_frames')} frames, robot={meta.get('robot')}")
    print(f"📁 Found {len(episodes)} episode directories")
    print()

    total_frames = 0
    all_valid = True

    for ep_dir in episodes:
        ep_name = ep_dir.name
        ep_errors = []

        # Check all subdirs exist
        for subdir in EXPECTED_SUBDIRS:
            d = ep_dir / subdir
            if not d.exists():
                ep_errors.append(f"Missing subdirectory: {subdir}/")
                continue

            files = sorted(d.iterdir())

            if subdir == "audio":
                if len(files) == 0:
                    ep_errors.append("Empty audio/ directory")
                else:
                    # Validate audio files
                    for af in files:
                        if af.suffix == ".npy":
                            audio = np.load(af)
                            if audio.dtype != np.float32:
                                ep_errors.append(f"Audio {af.name}: expected float32, got {audio.dtype}")
                        elif af.suffix == ".wav":
                            pass  # WAV is fine
                        else:
                            warnings.append(f"{ep_name}/audio/{af.name}: unexpected format")

            elif subdir == "point_clouds":
                if len(files) == 0:
                    ep_errors.append("⚠️  EMPTY point_clouds/ — this is the critical gap")
                else:
                    # Validate shape and dtype of first point cloud
                    pc = np.load(files[0])
                    meta.get("stereo_config", {}).get("image_size", [640, 480])[1]
                    meta.get("stereo_config", {}).get("image_size", [640, 480])[0]
                    if len(pc.shape) != 3 or pc.shape[2] != 3:
                        ep_errors.append(
                            f"Point cloud shape: {pc.shape}, expected (H, W, 3)"
                        )
                    if pc.dtype != np.float32:
                        ep_errors.append(f"Point cloud dtype: {pc.dtype}, expected float32")

            elif subdir == "depth":
                if len(files) == 0:
                    ep_errors.append("Empty depth/ directory")
                else:
                    depth = np.load(files[0])
                    if len(depth.shape) != 2:
                        ep_errors.append(f"Depth shape: {depth.shape}, expected (H, W)")
                    if depth.dtype != np.float32:
                        ep_errors.append(f"Depth dtype: {depth.dtype}, expected float32")
                    if depth.min() < 0:
                        ep_errors.append(f"Negative depth values: min={depth.min():.4f}")
                    if depth.max() > 100.0:
                        warnings.append(f"{ep_name}/depth: max depth {depth.max():.1f}m seems large")

            elif subdir == "disparity":
                if len(files) == 0:
                    ep_errors.append("Empty disparity/ directory")

            elif subdir in ("stereo_left", "stereo_right"):
                if len(files) == 0:
                    ep_errors.append(f"Empty {subdir}/ directory")

        # Cross-check frame counts across modalities
        frame_subdirs = ["stereo_left", "stereo_right", "depth", "disparity", "point_clouds"]
        frame_counts = {}
        for subdir in frame_subdirs:
            d = ep_dir / subdir
            if d.exists():
                frame_counts[subdir] = len(list(d.iterdir()))

        unique_counts = set(frame_counts.values())
        if len(unique_counts) > 1 and 0 not in unique_counts:
            ep_errors.append(
                f"Frame count mismatch across modalities: {frame_counts}"
            )

        n_frames = frame_counts.get("depth", 0)
        total_frames += n_frames

        # Report
        if ep_errors:
            all_valid = False
            print(f"  ❌ {ep_name}: {n_frames} frames")
            for err in ep_errors:
                print(f"     └─ {err}")
        else:
            print(f"  ✅ {ep_name}: {n_frames} frames, all 6 modalities present")

    # Summary
    print()
    if meta.get("total_frames") and total_frames != meta["total_frames"]:
        warnings.append(
            f"Total frame count: found {total_frames}, metadata says {meta['total_frames']}"
        )

    if warnings:
        print("⚠️  Warnings:")
        for w in warnings:
            print(f"   └─ {w}")
        print()

    if all_valid:
        print(f"✅ VALID: {len(episodes)} episodes, {total_frames} frames, all 6 modalities present")
        return True
    else:
        print("❌ INVALID: Dataset has structural issues (see above)")
        return False


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "datasets/strands-kitchen-stereo4d"
    valid = validate(path)
    sys.exit(0 if valid else 1)
