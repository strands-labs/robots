#!/usr/bin/env python3
"""
HuggingFace Multi-Embodiment Pick-and-Place Data Collection

Searches HuggingFace Hub for pick-and-place manipulation datasets,
maps them to GR00T N1.6 embodiment tags, and optionally downloads
+ converts to LeRobot v3 format for GR00T fine-tuning.

Pipeline:
    1. Search HF Hub for pick/place/manipulation datasets
    2. Map discovered datasets → GR00T embodiment tags
    3. Download + convert to unified LeRobot v3 format
    4. Generate data manifests for GR00T fine-tuning

Usage:
    # Search only (no download) — safe for CPU runners
    python scripts/collect_hf_pick_place.py --search-only

    # Search + generate manifest
    python scripts/collect_hf_pick_place.py --manifest-only --output ./data_manifests

    # Full download + conversion (Thor GPU runner recommended)
    python scripts/collect_hf_pick_place.py --download --output ./hf_pick_place_data

    # Download specific embodiment only
    python scripts/collect_hf_pick_place.py --download --embodiment so100 --output ./so100_data

    # Dry run with verbose logging
    python scripts/collect_hf_pick_place.py --search-only --verbose

Refs: Issue #125, #65
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  GR00T Embodiment Tag Mapping
# ═══════════════════════════════════════════════════════════════════════

# Maps GR00T N1.6 embodiment tags → known HuggingFace dataset patterns
# Each entry: (embodiment_tag, data_config, [hf_dataset_patterns])
EMBODIMENT_DATASET_MAP: Dict[str, Dict[str, Any]] = {
    "new_embodiment": {
        "description": "SO-100 / SO-101 low-cost arms (LeRobot ecosystem)",
        "data_configs": ["so100", "so100_dualcam", "so100_4cam"],
        "hf_patterns": [
            # Official LeRobot SO-100 datasets
            "lerobot/so100_pick",
            "lerobot/so100_wipe",
            "lerobot/so100_push",
            "lerobot/so100_stack",
            "lerobot/so100_fold",
            # Community SO-100/101 datasets
            "cadene/so100_pick_and_place",
            "hf://datasets/lerobot*so100*",
            "hf://datasets/lerobot*so101*",
            # Koch v1.1 (same form factor as SO-100)
            "lerobot/koch_pick",
            "lerobot/koch_pick_place",
        ],
        "search_queries": [
            "so100 pick place",
            "so101 manipulation",
            "koch robot pick",
            "lerobot pick place",
        ],
    },
    "gr1": {
        "description": "Fourier GR-1 humanoid",
        "data_configs": ["fourier_gr1_arms_only", "fourier_gr1_arms_waist", "fourier_gr1_full_upper_body"],
        "hf_patterns": [
            "nvidia/PhysicalAI-GR1-Dataset",
            "nvidia/GR00T-GR1-*",
            "hf://datasets/*fourier*gr1*",
            "hf://datasets/*GR-1*",
        ],
        "search_queries": [
            "fourier gr1 manipulation",
            "gr1 humanoid pick",
            "groot gr1",
        ],
    },
    "unitree_g1": {
        "description": "Unitree G1 humanoid",
        "data_configs": ["unitree_g1", "unitree_g1_full_body", "unitree_g1_locomanip"],
        "hf_patterns": [
            "nvidia/PhysicalAI-UnitreeG1-Dataset",
            "hf://datasets/*unitree*g1*",
            "hf://datasets/*unitree_g1*",
        ],
        "search_queries": [
            "unitree g1 manipulation",
            "unitree g1 pick place",
            "g1 humanoid grasping",
        ],
    },
    "robocasa_panda_omron": {
        "description": "Franka Panda + RoboCasa + Omron mobile manipulation",
        "data_configs": ["bimanual_panda_gripper", "single_panda_gripper"],
        "hf_patterns": [
            "nvidia/PhysicalAI-RoboCasa-Dataset",
            "nvidia/GR00T-RoboCasa-*",
            "hf://datasets/*robocasa*",
            "hf://datasets/*panda*pick*",
        ],
        "search_queries": [
            "robocasa manipulation",
            "franka panda pick place",
            "panda gripper grasping",
        ],
    },
    "oxe_droid": {
        "description": "Open X-Embodiment DROID dataset",
        "data_configs": ["oxe_droid"],
        "hf_patterns": [
            "droid_rss/droid_100",
            "hf://datasets/*droid*",
            "hf://datasets/*oxe*droid*",
        ],
        "search_queries": [
            "droid robot manipulation",
            "oxe droid pick",
        ],
    },
    "oxe_google": {
        "description": "Open X-Embodiment Google RT datasets",
        "data_configs": ["oxe_google"],
        "hf_patterns": [
            "hf://datasets/*rt1*",
            "hf://datasets/*rt2*",
            "hf://datasets/*bridge*",
            "hf://datasets/*google*robot*",
        ],
        "search_queries": [
            "google rt1 pick place",
            "bridge v2 manipulation",
            "open x-embodiment pick",
        ],
    },
    "oxe_widowx": {
        "description": "Open X-Embodiment WidowX datasets",
        "data_configs": ["oxe_widowx"],
        "hf_patterns": [
            "hf://datasets/*widowx*",
            "hf://datasets/*bridge*widowx*",
            "hf://datasets/lerobot*widowx*",
        ],
        "search_queries": [
            "widowx pick place",
            "widowx robot manipulation",
            "bridge widowx grasping",
        ],
    },
    "libero_panda": {
        "description": "LIBERO simulation benchmark (Panda arm)",
        "data_configs": ["libero_panda"],
        "hf_patterns": [
            "nvidia/PhysicalAI-LIBERO-Dataset",
            "hf://datasets/*libero*",
            "hf://datasets/lerobot*libero*",
        ],
        "search_queries": [
            "libero simulation pick",
            "libero panda manipulation",
        ],
    },
    "behavior_r1_pro": {
        "description": "Galaxea R1 Pro / BEHAVIOR bimanual manipulation",
        "data_configs": ["galaxea_r1_pro"],
        "hf_patterns": [
            "nvidia/PhysicalAI-BEHAVIOR-Dataset",
            "hf://datasets/*behavior*r1*",
            "hf://datasets/*galaxea*",
        ],
        "search_queries": [
            "behavior r1 manipulation",
            "galaxea r1 pro pick",
        ],
    },
}


@dataclass
class DatasetInfo:
    """Metadata for a discovered HuggingFace dataset."""
    repo_id: str
    embodiment_tag: str
    data_config: str
    description: str = ""
    num_episodes: int = 0
    num_steps: int = 0
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    download_url: str = ""
    is_lerobot_v2: bool = False
    confidence: float = 0.0  # How confident the mapping is (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "embodiment_tag": self.embodiment_tag,
            "data_config": self.data_config,
            "description": self.description,
            "num_episodes": self.num_episodes,
            "num_steps": self.num_steps,
            "size_bytes": self.size_bytes,
            "tags": self.tags,
            "task_types": self.task_types,
            "is_lerobot_v2": self.is_lerobot_v2,
            "confidence": self.confidence,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Search Engine
# ═══════════════════════════════════════════════════════════════════════

def search_hf_datasets(
    query: str,
    limit: int = 50,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search HuggingFace Hub for datasets matching query.

    Args:
        query: Search query string.
        limit: Maximum results to return.
        tags: Optional tag filters (e.g., ["task_categories:robotics"]).

    Returns:
        List of dataset metadata dicts.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        results = []

        datasets = api.list_datasets(
            search=query,
            limit=limit,
            sort="downloads",
        )

        for ds in datasets:
            info = {
                "repo_id": ds.id,
                "description": getattr(ds, "description", "") or "",
                "tags": list(getattr(ds, "tags", None) or []),
                "downloads": getattr(ds, "downloads", 0) or 0,
                "likes": getattr(ds, "likes", 0) or 0,
                "created_at": str(getattr(ds, "created_at", "")) if getattr(ds, "created_at", None) else "",
                "last_modified": str(getattr(ds, "last_modified", "")) if getattr(ds, "last_modified", None) else "",
            }
            results.append(info)

        return results

    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return []
    except Exception as e:
        logger.warning(f"HF search failed for '{query}': {e}")
        return []


def check_dataset_exists(repo_id: str) -> Optional[Dict[str, Any]]:
    """Check if a specific HF dataset exists and get its metadata.

    Handles auth errors gracefully — returns None for gated/private repos
    when HF_TOKEN is not set rather than crashing.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            info = api.dataset_info(repo_id)
            return {
                "repo_id": info.id,
                "description": getattr(info, "description", "") or "",
                "tags": list(getattr(info, "tags", None) or []),
                "downloads": getattr(info, "downloads", 0) or 0,
                "likes": getattr(info, "likes", 0) or 0,
                "size": getattr(info, "dataset_size", None),
                "exists": True,
            }
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "unauthorized" in err_str:
                logger.debug(f"Auth required for {repo_id} (set HF_TOKEN)")
            elif "404" in err_str or "not found" in err_str:
                logger.debug(f"Dataset not found: {repo_id}")
            else:
                logger.debug(f"Cannot check {repo_id}: {e}")
            return None

    except ImportError:
        logger.error("huggingface_hub not installed")
        return None


def classify_dataset_task(info: Dict[str, Any]) -> List[str]:
    """Classify a dataset's task types based on metadata."""
    tasks = []
    text = f"{info.get('repo_id', '')} {info.get('description', '')}".lower()
    tags = [t.lower() for t in info.get("tags", [])]

    # Pick and place detection
    pick_keywords = ["pick", "grasp", "grip", "grab", "lifting", "pick_and_place",
                     "pick-and-place", "pickandplace", "pick place"]
    if any(kw in text for kw in pick_keywords):
        tasks.append("pick_and_place")

    # Other manipulation tasks (also useful for training)
    if any(kw in text for kw in ["push", "pushing"]):
        tasks.append("pushing")
    if any(kw in text for kw in ["stack", "stacking"]):
        tasks.append("stacking")
    if any(kw in text for kw in ["wipe", "wiping", "clean"]):
        tasks.append("wiping")
    if any(kw in text for kw in ["fold", "folding"]):
        tasks.append("folding")
    if any(kw in text for kw in ["pour", "pouring"]):
        tasks.append("pouring")
    if any(kw in text for kw in ["open", "close", "door", "drawer"]):
        tasks.append("articulated")
    if any(kw in text for kw in ["manipulat", "robot", "arm", "gripper"]):
        tasks.append("manipulation")

    # LeRobot format detection
    if "lerobot" in text or "lerobot" in " ".join(tags):
        tasks.append("lerobot_format")

    return tasks if tasks else ["unknown"]


def map_dataset_to_embodiment(
    info: Dict[str, Any],
) -> Optional[Tuple[str, str, float]]:
    """Map a dataset to a GR00T embodiment tag based on metadata.

    Returns:
        (embodiment_tag, data_config, confidence) or None.
    """
    repo_id = info.get("repo_id", "").lower()
    description = info.get("description", "").lower()
    tags = [t.lower() for t in info.get("tags", [])]
    text = f"{repo_id} {description} {' '.join(tags)}"

    # Direct repo_id pattern matching (highest confidence)
    direct_matches = [
        # SO-100/101
        (["so100", "so101", "so-100", "so-101", "koch"], "new_embodiment", "so100", 0.95),
        # Fourier GR-1
        (["gr-1", "gr1", "fourier"], "gr1", "fourier_gr1_arms_only", 0.9),
        # Unitree G1
        (["unitree_g1", "unitree-g1", "g1_humanoid"], "unitree_g1", "unitree_g1", 0.9),
        # DROID
        (["droid"], "oxe_droid", "oxe_droid", 0.85),
        # Google RT / Bridge
        (["rt1", "rt2", "bridge_v2", "bridge-v2"], "oxe_google", "oxe_google", 0.85),
        # WidowX
        (["widowx", "widow_x"], "oxe_widowx", "oxe_widowx", 0.85),
        # LIBERO
        (["libero"], "libero_panda", "libero_panda", 0.9),
        # RoboCasa
        (["robocasa"], "robocasa_panda_omron", "single_panda_gripper", 0.85),
        # Panda / Franka
        (["panda", "franka"], "robocasa_panda_omron", "bimanual_panda_gripper", 0.7),
        # BEHAVIOR / Galaxea R1
        (["behavior", "galaxea", "r1_pro"], "behavior_r1_pro", "galaxea_r1_pro", 0.85),
    ]

    for keywords, emb_tag, data_cfg, conf in direct_matches:
        if any(kw in text for kw in keywords):
            return (emb_tag, data_cfg, conf)

    # Open X-Embodiment generic
    if any(kw in text for kw in ["open_x", "oxe", "x-embodiment", "open x-embodiment"]):
        return ("oxe_google", "oxe_google", 0.6)

    # Generic LeRobot dataset → default to new_embodiment
    if "lerobot" in text:
        return ("new_embodiment", "so100", 0.5)

    return None


# ═══════════════════════════════════════════════════════════════════════
#  Discovery Pipeline
# ═══════════════════════════════════════════════════════════════════════

def discover_datasets(
    embodiments: Optional[List[str]] = None,
    max_per_query: int = 30,
    verbose: bool = False,
) -> Dict[str, List[DatasetInfo]]:
    """Discover HuggingFace datasets for all (or selected) embodiment tags.

    Args:
        embodiments: List of embodiment tags to search for, or None for all.
        max_per_query: Max results per HF search query.
        verbose: Enable verbose logging.

    Returns:
        Dict mapping embodiment_tag → list of DatasetInfo.
    """
    results: Dict[str, List[DatasetInfo]] = {}
    seen_repos = set()

    target_embodiments = embodiments or list(EMBODIMENT_DATASET_MAP.keys())

    for emb_tag in target_embodiments:
        if emb_tag not in EMBODIMENT_DATASET_MAP:
            logger.warning(f"Unknown embodiment tag: {emb_tag}")
            continue

        emb_info = EMBODIMENT_DATASET_MAP[emb_tag]
        results[emb_tag] = []

        logger.info(f"\n🔍 Searching for {emb_tag}: {emb_info['description']}")

        # 1. Check known dataset patterns directly
        for pattern in emb_info["hf_patterns"]:
            if pattern.startswith("hf://datasets/"):
                continue  # Skip wildcard patterns for direct checks
            ds_info = check_dataset_exists(pattern)
            if ds_info and ds_info.get("exists") and pattern not in seen_repos:
                seen_repos.add(pattern)
                tasks = classify_dataset_task(ds_info)
                dataset = DatasetInfo(
                    repo_id=pattern,
                    embodiment_tag=emb_tag,
                    data_config=emb_info["data_configs"][0],
                    description=ds_info.get("description", ""),
                    tags=ds_info.get("tags", []),
                    task_types=tasks,
                    is_lerobot_v2="lerobot" in pattern.lower(),
                    confidence=0.95,
                )
                results[emb_tag].append(dataset)
                if verbose:
                    logger.info(f"  ✅ Found known: {pattern} (tasks: {tasks})")

        # 2. Search HF Hub with embodiment-specific queries
        for query in emb_info["search_queries"]:
            search_results = search_hf_datasets(query, limit=max_per_query)
            for sr in search_results:
                repo_id = sr["repo_id"]
                if repo_id in seen_repos:
                    continue
                seen_repos.add(repo_id)

                # Classify task and check relevance
                tasks = classify_dataset_task(sr)
                is_manipulation = any(t in tasks for t in [
                    "pick_and_place", "pushing", "stacking", "manipulation",
                    "wiping", "folding", "pouring", "articulated",
                ])

                if not is_manipulation:
                    if verbose:
                        logger.debug(f"  ⏭️ Skipped (not manipulation): {repo_id}")
                    continue

                # Map to embodiment
                mapping = map_dataset_to_embodiment(sr)
                if mapping:
                    mapped_emb, mapped_cfg, confidence = mapping
                    # If it mapped to this embodiment or we found it via this search
                    target_emb = mapped_emb if mapped_emb in target_embodiments else emb_tag
                    dataset = DatasetInfo(
                        repo_id=repo_id,
                        embodiment_tag=target_emb,
                        data_config=mapped_cfg,
                        description=sr.get("description", ""),
                        tags=sr.get("tags", []),
                        task_types=tasks,
                        is_lerobot_v2="lerobot" in repo_id.lower() or "lerobot_format" in tasks,
                        confidence=confidence,
                    )
                    results.setdefault(target_emb, []).append(dataset)
                    if verbose:
                        logger.info(f"  📦 Mapped: {repo_id} → {target_emb}/{mapped_cfg} (conf={confidence:.2f})")
                else:
                    # Assign to the embodiment we were searching for
                    dataset = DatasetInfo(
                        repo_id=repo_id,
                        embodiment_tag=emb_tag,
                        data_config=emb_info["data_configs"][0],
                        description=sr.get("description", ""),
                        tags=sr.get("tags", []),
                        task_types=tasks,
                        is_lerobot_v2="lerobot" in repo_id.lower() or "lerobot_format" in tasks,
                        confidence=0.3,  # Low confidence — needs manual review
                    )
                    results[emb_tag].append(dataset)
                    if verbose:
                        logger.info(f"  ❓ Unconfirmed: {repo_id} → {emb_tag} (conf=0.3, needs review)")

    # 3. Run broad pick-and-place searches
    logger.info("\n🔍 Running broad pick-and-place searches...")
    broad_queries = [
        "robot pick and place dataset",
        "robotic manipulation dataset",
        "robot grasping dataset",
        "lerobot pick place",
        "robot arm demonstration dataset",
    ]

    for query in broad_queries:
        search_results = search_hf_datasets(query, limit=max_per_query)
        for sr in search_results:
            repo_id = sr["repo_id"]
            if repo_id in seen_repos:
                continue
            seen_repos.add(repo_id)

            tasks = classify_dataset_task(sr)
            is_manipulation = any(t in tasks for t in [
                "pick_and_place", "pushing", "stacking", "manipulation",
            ])
            if not is_manipulation:
                continue

            mapping = map_dataset_to_embodiment(sr)
            if mapping:
                mapped_emb, mapped_cfg, confidence = mapping
                if mapped_emb in target_embodiments:
                    dataset = DatasetInfo(
                        repo_id=repo_id,
                        embodiment_tag=mapped_emb,
                        data_config=mapped_cfg,
                        description=sr.get("description", ""),
                        tags=sr.get("tags", []),
                        task_types=tasks,
                        is_lerobot_v2="lerobot" in repo_id.lower() or "lerobot_format" in tasks,
                        confidence=confidence,
                    )
                    results.setdefault(mapped_emb, []).append(dataset)
                    if verbose:
                        logger.info(f"  📦 Broad match: {repo_id} → {mapped_emb} (conf={confidence:.2f})")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Download & Conversion
# ═══════════════════════════════════════════════════════════════════════

def download_dataset(
    repo_id: str,
    output_dir: str,
    max_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Download a HuggingFace dataset.

    Args:
        repo_id: HuggingFace dataset repo ID.
        output_dir: Local directory to save data.
        max_episodes: Maximum episodes to download (None = all).

    Returns:
        Dict with download stats.
    """
    try:
        from huggingface_hub import snapshot_download

        local_path = os.path.join(output_dir, repo_id.replace("/", "_"))
        os.makedirs(local_path, exist_ok=True)

        logger.info(f"⬇️ Downloading {repo_id} → {local_path}")
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_path,
        )

        # Count episodes (heuristic: count parquet files or episode dirs)
        episode_count = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith((".parquet", ".arrow", ".json")):
                    episode_count += 1

        return {
            "repo_id": repo_id,
            "local_path": str(path),
            "episode_count_estimate": episode_count,
            "status": "downloaded",
        }

    except Exception as e:
        logger.error(f"Download failed for {repo_id}: {e}")
        return {"repo_id": repo_id, "status": "failed", "error": str(e)}


def convert_to_lerobot_v3(
    dataset_path: str,
    embodiment_tag: str,
    data_config: str,
    output_path: str,
) -> Dict[str, Any]:
    """Convert a downloaded dataset to LeRobot v3 format.

    This is the format expected by GR00T N1.6 fine-tuning.
    The conversion maps dataset-specific keys to GR00T data_config keys.

    Args:
        dataset_path: Path to downloaded dataset.
        embodiment_tag: GR00T embodiment tag.
        data_config: GR00T data config name.
        output_path: Path for converted output.

    Returns:
        Dict with conversion stats.
    """
    try:
        # Import data config to get expected keys
        from strands_robots.policies.groot.data_config import load_data_config

        config = load_data_config(data_config)
        expected_video_keys = config.video_keys
        expected_state_keys = config.state_keys
        expected_action_keys = config.action_keys

        logger.info(f"🔄 Converting {dataset_path} → LeRobot v3")
        logger.info(f"   Embodiment: {embodiment_tag}, Config: {data_config}")
        logger.info(f"   Expected video keys: {expected_video_keys}")
        logger.info(f"   Expected state keys: {expected_state_keys}")
        logger.info(f"   Expected action keys: {expected_action_keys}")

        # TODO: Implement actual conversion logic per dataset format
        # This requires per-dataset adapters since HF datasets have varying schemas.
        # The conversion needs to:
        # 1. Load the source dataset (parquet/arrow/videos)
        # 2. Map source observation keys → GR00T video keys
        # 3. Map source state keys → GR00T state keys
        # 4. Map source action keys → GR00T action keys
        # 5. Add language annotation if missing
        # 6. Write in LeRobot v3 format (parquet + video files)

        os.makedirs(output_path, exist_ok=True)

        # Write conversion manifest
        manifest = {
            "embodiment_tag": embodiment_tag,
            "data_config": data_config,
            "source_path": dataset_path,
            "output_path": output_path,
            "expected_keys": {
                "video": expected_video_keys,
                "state": expected_state_keys,
                "action": expected_action_keys,
            },
            "status": "manifest_generated",
            "note": "Full conversion requires running on GPU runner with datasets library",
        }

        manifest_path = os.path.join(output_path, "conversion_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return {"status": "failed", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  Manifest Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_manifest(
    discovered: Dict[str, List[DatasetInfo]],
    output_dir: str,
    min_confidence: float = 0.3,
) -> Dict[str, Any]:
    """Generate a data collection manifest for all discovered datasets.

    Creates a structured JSON manifest that can be used by Thor to
    execute the actual downloads and conversions.

    Args:
        discovered: Output from discover_datasets().
        output_dir: Directory to write manifests.
        min_confidence: Minimum confidence threshold to include.

    Returns:
        Dict with manifest statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_datasets": 0,
        "total_embodiments": 0,
        "embodiments": {},
        "download_commands": [],
    }

    for emb_tag, datasets in discovered.items():
        filtered = [d for d in datasets if d.confidence >= min_confidence]
        if not filtered:
            continue

        # Deduplicate by repo_id
        seen = set()
        unique = []
        for d in filtered:
            if d.repo_id not in seen:
                seen.add(d.repo_id)
                unique.append(d)

        manifest["embodiments"][emb_tag] = {
            "description": EMBODIMENT_DATASET_MAP.get(emb_tag, {}).get("description", ""),
            "data_configs": EMBODIMENT_DATASET_MAP.get(emb_tag, {}).get("data_configs", []),
            "datasets": [d.to_dict() for d in unique],
            "count": len(unique),
            "high_confidence": len([d for d in unique if d.confidence >= 0.7]),
        }

        manifest["total_datasets"] += len(unique)
        manifest["total_embodiments"] += 1

        # Generate download command for Thor
        for d in unique:
            if d.confidence >= 0.5:  # Only include reasonably confident mappings
                manifest["download_commands"].append({
                    "repo_id": d.repo_id,
                    "embodiment_tag": emb_tag,
                    "data_config": d.data_config,
                    "confidence": d.confidence,
                    "command": f"python scripts/collect_hf_pick_place.py --download --repo {d.repo_id} --embodiment {emb_tag} --data-config {d.data_config}",
                })

    # Write full manifest
    manifest_path = os.path.join(output_dir, "hf_pick_place_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write per-embodiment manifests
    for emb_tag, emb_data in manifest["embodiments"].items():
        emb_path = os.path.join(output_dir, f"manifest_{emb_tag}.json")
        with open(emb_path, "w") as f:
            json.dump(emb_data, f, indent=2)

    # Write Thor dispatch script
    thor_script = generate_thor_dispatch_script(manifest)
    thor_path = os.path.join(output_dir, "thor_dispatch.sh")
    with open(thor_path, "w") as f:
        f.write(thor_script)
    os.chmod(thor_path, 0o755)

    logger.info(f"\n📋 Manifest generated: {manifest_path}")
    logger.info(f"   Embodiments: {manifest['total_embodiments']}")
    logger.info(f"   Total datasets: {manifest['total_datasets']}")
    logger.info(f"   Download commands: {len(manifest['download_commands'])}")

    return manifest


def generate_thor_dispatch_script(manifest: Dict[str, Any]) -> str:
    """Generate a shell script for Thor to execute the data collection."""
    total_ds = manifest.get('total_datasets', 0)
    total_emb = manifest.get('total_embodiments', 0)
    gen_at = manifest.get('generated_at', 'unknown')

    lines = [
        "#!/bin/bash",
        "# Auto-generated Thor dispatch script for HF data collection",
        f"# Generated: {gen_at}",
        f"# Total datasets: {total_ds}",
        "",
        "set -euo pipefail",
        "",
        "OUTPUT_DIR=${{1:-./hf_pick_place_data}}",
        "mkdir -p $OUTPUT_DIR",
        "",
        "echo 'HuggingFace Multi-Embodiment Data Collection'",
        f"echo '   {total_ds} datasets across {total_emb} embodiments'",
        "",
        "# Ensure HuggingFace token is available",
        'if [ -z "${{HF_TOKEN:-}}" ]; then',
        "    echo 'HF_TOKEN not set. Some datasets may require authentication.'",
        "    echo '   Run: huggingface-cli login'",
        "fi",
        "",
    ]

    for cmd in manifest.get("download_commands", []):
        repo_id = cmd['repo_id']
        emb_tag = cmd['embodiment_tag']
        conf = cmd['confidence']
        dl_cmd = cmd['command']
        lines.append(f"# {emb_tag} - {repo_id} (confidence={conf:.2f})")
        lines.append(f"echo 'Downloading {repo_id}...'")
        lines.append(f"{dl_cmd} --output $OUTPUT_DIR || echo 'Failed: {repo_id}'")
        lines.append("")

    lines.extend([
        "echo 'Data collection complete!'",
        'echo "   Output: $OUTPUT_DIR"',
        "ls -la $OUTPUT_DIR/",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Summary Report
# ═══════════════════════════════════════════════════════════════════════

def print_summary(discovered: Dict[str, List[DatasetInfo]]):
    """Print a human-readable summary of discovered datasets."""
    print("\n" + "=" * 78)
    print("🧠 HuggingFace Multi-Embodiment Pick-and-Place Dataset Discovery")
    print("=" * 78)

    total = 0
    high_conf = 0

    for emb_tag in sorted(discovered.keys()):
        datasets = discovered[emb_tag]
        if not datasets:
            continue

        # Deduplicate
        seen = set()
        unique = [d for d in datasets if d.repo_id not in seen and not seen.add(d.repo_id)]

        emb_desc = EMBODIMENT_DATASET_MAP.get(emb_tag, {}).get("description", emb_tag)
        hc = len([d for d in unique if d.confidence >= 0.7])

        print(f"\n🤖 {emb_tag} — {emb_desc}")
        print(f"   Datasets found: {len(unique)} ({hc} high-confidence)")
        print(f"   Data configs: {EMBODIMENT_DATASET_MAP.get(emb_tag, {}).get('data_configs', [])}")

        for d in sorted(unique, key=lambda x: -x.confidence)[:8]:
            conf_icon = "🟢" if d.confidence >= 0.7 else "🟡" if d.confidence >= 0.5 else "🔴"
            tasks_str = ", ".join(d.task_types[:3])
            lerobot_badge = " [LeRobot]" if d.is_lerobot_v2 else ""
            print(f"   {conf_icon} {d.repo_id}{lerobot_badge} — {tasks_str} (conf={d.confidence:.2f})")

        if len(unique) > 8:
            print(f"   ... and {len(unique) - 8} more")

        total += len(unique)
        high_conf += hc

    print(f"\n{'=' * 78}")
    print(f"📊 Summary: {total} datasets across {len([k for k, v in discovered.items() if v])} embodiments")
    print(f"   High-confidence (≥0.7): {high_conf}")
    print(f"   Medium-confidence (0.5-0.7): {total - high_conf}")
    print(f"{'=' * 78}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Multi-Embodiment Data Collection for GR00T N1.6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search and display summary (CPU safe)
  python scripts/collect_hf_pick_place.py --search-only

  # Generate manifest for Thor dispatch
  python scripts/collect_hf_pick_place.py --manifest-only --output ./manifests

  # Download specific dataset
  python scripts/collect_hf_pick_place.py --download --repo lerobot/so100_pick --output ./data

  # Full pipeline (Thor recommended)
  python scripts/collect_hf_pick_place.py --download --output ./hf_pick_place_data
        """,
    )
    parser.add_argument("--search-only", action="store_true",
                        help="Only search and display results (no download)")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Generate manifest without downloading")
    parser.add_argument("--download", action="store_true",
                        help="Download discovered datasets")
    parser.add_argument("--repo", type=str, default=None,
                        help="Download a specific repo ID")
    parser.add_argument("--embodiment", type=str, default=None,
                        help="Filter to specific embodiment tag")
    parser.add_argument("--data-config", type=str, default=None,
                        help="Override data config for download")
    parser.add_argument("--output", type=str, default="./hf_pick_place_data",
                        help="Output directory (default: ./hf_pick_place_data)")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum confidence threshold (default: 0.3)")
    parser.add_argument("--max-per-query", type=int, default=30,
                        help="Max results per HF search query (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Single repo download mode
    if args.repo:
        emb = args.embodiment or "new_embodiment"
        cfg = args.data_config or "so100"
        result = download_dataset(args.repo, args.output)
        if result.get("status") == "downloaded":
            convert_to_lerobot_v3(result["local_path"], emb, cfg,
                                  os.path.join(args.output, "converted", emb))
        print(json.dumps(result, indent=2))
        return

    # Discovery
    embodiments = [args.embodiment] if args.embodiment else None
    discovered = discover_datasets(
        embodiments=embodiments,
        max_per_query=args.max_per_query,
        verbose=args.verbose,
    )

    # Summary
    print_summary(discovered)

    # Manifest
    if args.manifest_only or args.download:
        generate_manifest(
            discovered, args.output,
            min_confidence=args.min_confidence,
        )

    # Download
    if args.download:
        logger.info("\n⬇️ Starting downloads...")
        for emb_tag, datasets in discovered.items():
            for d in datasets:
                if d.confidence >= args.min_confidence:
                    result = download_dataset(d.repo_id, args.output)
                    if result.get("status") == "downloaded":
                        convert_to_lerobot_v3(
                            result["local_path"], emb_tag, d.data_config,
                            os.path.join(args.output, "converted", emb_tag),
                        )

    if args.search_only:
        logger.info("\n💡 To generate manifests: add --manifest-only")
        logger.info("💡 To download: add --download")
        logger.info("💡 For Thor dispatch: python scripts/collect_hf_pick_place.py --manifest-only --output ./manifests")


if __name__ == "__main__":
    main()
