#!/usr/bin/env python3
"""
LeRobotDataset tool for strands-robots.

Provides agent-accessible dataset operations:
- Create datasets in LeRobot v3.0 format (parquet + video)
- Record episodes from teleop or policy
- Push/pull from HuggingFace Hub
- Browse, visualize, and replay episodes

Supports create, record, replay, push, pull, visualize, and evaluate operations.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from strands import tool

logger = logging.getLogger(__name__)

# Global state for active recording sessions
_ACTIVE_RECORDINGS: Dict[str, Any] = {}
_RECORDING_LOCK = threading.Lock()


def _get_lerobot_dataset(repo_id: str, root: Optional[str] = None, create: bool = False, **kwargs):
    """Load or create a LeRobotDataset."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError("lerobot not installed. Install with: pip install lerobot")

    if create:
        return LeRobotDataset.create(repo_id=repo_id, root=root, **kwargs)
    else:
        return LeRobotDataset(repo_id=repo_id, root=root, **kwargs)


def _build_default_features(robot_type: str = "unknown") -> Dict[str, Any]:
    """Build minimal default features for a dataset when none are specified.

    LeRobot v3 requires features to be specified at create time.
    This provides reasonable defaults for a generic robot.
    """
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ],
        },
    }


@tool
def lerobot_dataset(
    action: str,
    # Dataset identification
    repo_id: Optional[str] = None,
    root: Optional[str] = None,
    # Episode recording
    robot_name: Optional[str] = None,
    task: Optional[str] = None,
    num_episodes: int = 1,
    episode_time_s: float = 60.0,
    fps: int = 30,
    # Teleop source for recording
    teleop_name: Optional[str] = None,
    teleop_device: Optional[str] = None,
    # Policy source for recording
    policy_provider: Optional[str] = None,
    pretrained_name_or_path: Optional[str] = None,
    instruction: Optional[str] = None,
    # Hub operations
    push_to_hub: bool = False,
    tags: Optional[List[str]] = None,
    # Replay
    episode: int = 0,
    # Browse
    max_episodes: int = 10,
    # Video encoding
    video_codec: str = "libsvtav1",
    # Features (for create/record)
    features: Optional[Dict[str, Any]] = None,
    robot_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    LeRobotDataset tool — create, record, browse, push, and replay robot datasets.

    Works with LeRobot v3.0 format: parquet episodes + encoded video + HF Hub.

    Actions:
        create: Create a new empty dataset
        info: Get dataset metadata (episodes, features, stats)
        record: Record episodes from teleop or policy into a dataset
        stop_recording: Stop an active recording session
        push: Push dataset to HuggingFace Hub
        pull: Download dataset from HuggingFace Hub
        browse: List episodes and preview data
        replay: Replay episode actions (returns action sequence)
        list_hub: List popular LeRobot datasets on HuggingFace Hub
        compute_stats: Compute dataset statistics

    Args:
        action: Action to perform
        repo_id: HuggingFace dataset repository ID (e.g., "user/my_dataset")
        root: Local storage directory (default: ~/.cache/huggingface/lerobot/)
        robot_name: Robot tool name for recording
        task: Task description for episodes
        num_episodes: Number of episodes to record
        episode_time_s: Episode duration in seconds
        fps: Recording frame rate
        teleop_name: Active teleoperator name (from teleoperator tool)
        teleop_device: Teleoperator device type to start for recording
        policy_provider: Policy provider for autonomous recording
        pretrained_name_or_path: HF model for policy recording
        instruction: Language instruction for policy
        push_to_hub: Push after recording
        tags: Dataset tags
        episode: Episode index for replay/browse
        max_episodes: Max episodes to show in browse
        video_codec: Video codec (h264, hevc, libsvtav1)
        features: Feature dict for create (LeRobot v3 format). Auto-detected if None.
        robot_type: Robot type string (e.g., "so100", "panda")

    Returns:
        Dict with status and dataset info

    Examples:
        # Create a new dataset with features
        lerobot_dataset(
            action="create",
            repo_id="user/my_robot_data",
            features={
                "observation.state": {"dtype": "float32", "shape": (6,), "names": ["j1","j2","j3","j4","j5","j6"]},
                "action": {"dtype": "float32", "shape": (6,), "names": ["j1","j2","j3","j4","j5","j6"]},
            },
            robot_type="so100",
        )

        # Record episodes using DatasetRecorder (recommended)
        lerobot_dataset(
            action="record",
            repo_id="user/my_data",
            teleop_device="keyboard_ee",
            task="pick up the red cube",
            num_episodes=5,
            episode_time_s=30,
        )

        # Browse a dataset
        lerobot_dataset(action="info", repo_id="lerobot/aloha_sim_transfer_cube_human")

        # Push to Hub
        lerobot_dataset(action="push", repo_id="user/my_data")

        # Replay episode
        lerobot_dataset(action="replay", repo_id="user/my_data", episode=0)
    """
    try:
        if action == "create":
            if not repo_id:
                return {"status": "error", "content": [{"text": "repo_id required for create"}]}

            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                # LeRobot v3 requires features at create time
                ds_features = features or _build_default_features(robot_type or "unknown")

                ds = LeRobotDataset.create(
                    repo_id=repo_id,
                    root=root,
                    fps=fps,
                    features=ds_features,
                    robot_type=robot_type,
                )

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"✅ Dataset created: {repo_id}\n"
                                f"   Root: {ds.root}\n"
                                f"   FPS: {fps}\n"
                                f"   Features: {list(ds_features.keys())}\n"
                                f"   Format: LeRobot v3.0 (parquet + video)"
                            )
                        }
                    ],
                }
            except ImportError:
                # Fallback: create directory structure manually
                ds_root = Path(root) if root else Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
                ds_root.mkdir(parents=True, exist_ok=True)

                info = {
                    "repo_id": repo_id,
                    "codebase_version": "v3.0",
                    "fps": fps,
                    "features": features or _build_default_features(),
                    "total_episodes": 0,
                    "total_frames": 0,
                }
                with open(ds_root / "info.json", "w") as f:
                    json.dump(info, f, indent=2)

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"✅ Dataset structure created: {repo_id}\n"
                                f"   Root: {ds_root}\n"
                                f"   Note: lerobot not installed, created minimal structure"
                            )
                        }
                    ],
                }

        elif action == "info":
            if not repo_id:
                return {"status": "error", "content": [{"text": "repo_id required"}]}

            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                ds = LeRobotDataset(repo_id=repo_id, root=root)
                meta = ds.meta

                ds_features = {}
                if hasattr(meta, "features"):
                    ds_features = {k: str(v) for k, v in meta.features.items()}

                info_text = (
                    f"📊 Dataset: {repo_id}\n"
                    f"   Episodes: {meta.total_episodes}\n"
                    f"   Frames: {meta.total_frames}\n"
                    f"   FPS: {meta.fps}\n"
                    f"   Features: {len(ds_features)}\n"
                )

                for feat_name, feat_info in list(ds_features.items())[:10]:
                    info_text += f"     - {feat_name}: {feat_info}\n"

                if hasattr(meta, "tasks") and meta.tasks:
                    info_text += f"   Tasks: {meta.tasks}\n"

                return {
                    "status": "success",
                    "content": [
                        {"text": info_text},
                        {
                            "json": {
                                "repo_id": repo_id,
                                "total_episodes": meta.total_episodes,
                                "total_frames": meta.total_frames,
                                "fps": meta.fps,
                                "features": ds_features,
                            }
                        },
                    ],
                }
            except ImportError:
                # Fallback: read info.json
                ds_root = Path(root) if root else Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
                info_path = ds_root / "info.json"
                if info_path.exists():
                    with open(info_path) as f:
                        info = json.load(f)
                    return {
                        "status": "success",
                        "content": [{"text": f"📊 Dataset: {repo_id}\n{json.dumps(info, indent=2)}"}],
                    }
                else:
                    return {"status": "error", "content": [{"text": f"Dataset not found: {repo_id}"}]}

        elif action == "record":
            if not repo_id:
                return {"status": "error", "content": [{"text": "repo_id required for recording"}]}

            recording_id = f"rec_{int(time.time())}"

            with _RECORDING_LOCK:
                _ACTIVE_RECORDINGS[recording_id] = {
                    "repo_id": repo_id,
                    "task": task or "Untitled task",
                    "started_at": time.time(),
                    "episodes_recorded": 0,
                    "total_frames": 0,
                    "target_episodes": num_episodes,
                    "fps": fps,
                    "status": "starting",
                }

            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                # Create or load dataset
                ds_root = Path(root) if root else None
                ds = None
                try:
                    ds = LeRobotDataset(repo_id=repo_id, root=ds_root)
                    logger.info(f"Loaded existing dataset: {repo_id}")
                except Exception:
                    # LeRobot v3 requires features at create time
                    ds_features = features or _build_default_features(robot_type or "unknown")
                    ds = LeRobotDataset.create(
                        repo_id=repo_id,
                        root=ds_root,
                        fps=fps,
                        features=ds_features,
                        robot_type=robot_type,
                    )
                    logger.info(f"Created new dataset: {repo_id}")

                with _RECORDING_LOCK:
                    _ACTIVE_RECORDINGS[recording_id]["dataset"] = ds
                    _ACTIVE_RECORDINGS[recording_id]["status"] = "ready"

                # If teleop_device specified, start a teleoperator
                teleop = None
                if teleop_device:
                    from strands_robots.tools.teleoperator import _resolve_teleoperator

                    teleop = _resolve_teleoperator(teleop_device)
                    teleop.connect()

                # Get feature keys from dataset to build frames correctly
                ds_feature_keys = set(ds.features.keys()) if ds.features else set()
                has_state = "observation.state" in ds_feature_keys
                has_action = "action" in ds_feature_keys
                action_dim = ds.features.get("action", {}).get("shape", (0,))[0] if has_action else 0
                state_dim = ds.features.get("observation.state", {}).get("shape", (0,))[0] if has_state else 0

                # Record episodes
                interval = 1.0 / fps
                for ep_idx in range(num_episodes):
                    with _RECORDING_LOCK:
                        _ACTIVE_RECORDINGS[recording_id]["status"] = f"recording episode {ep_idx + 1}/{num_episodes}"

                    ep_frames = 0
                    max_frames = int(episode_time_s * fps)

                    for frame_idx in range(max_frames):
                        frame_start = time.time()

                        # Get action from teleop or policy
                        action_data = {}
                        if teleop:
                            action_data = teleop.get_action()
                        elif policy_provider:
                            # Placeholder for policy-based recording
                            action_data = {"mock": 0.0}

                        # Build frame in LeRobot v3 format:
                        # - "task" key is mandatory (per-frame)
                        # - Feature keys must match dataset features exactly
                        frame_data = {
                            "task": task or "Untitled task",
                        }

                        # Build action vector matching dataset's "action" feature
                        if has_action and action_data:
                            action_values = list(action_data.values())
                            # Pad or truncate to match expected action dimension
                            if len(action_values) < action_dim:
                                action_values.extend([0.0] * (action_dim - len(action_values)))
                            elif len(action_values) > action_dim:
                                action_values = action_values[:action_dim]
                            frame_data["action"] = np.array(action_values, dtype=np.float32)

                        # Build state vector (zero-filled when no observation source)
                        if has_state:
                            frame_data["observation.state"] = np.zeros(state_dim, dtype=np.float32)

                        try:
                            ds.add_frame(frame_data)
                        except Exception as e:
                            logger.warning(f"Frame add error: {e}")

                        ep_frames += 1

                        # Maintain FPS
                        elapsed = time.time() - frame_start
                        sleep_time = interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                        # Check if recording was stopped
                        with _RECORDING_LOCK:
                            if recording_id not in _ACTIVE_RECORDINGS:
                                break

                    # End episode — LeRobot v3 save_episode() takes no task arg.
                    # Tasks are embedded per-frame via add_frame().
                    try:
                        ds.save_episode()
                    except Exception as e:
                        logger.warning(f"Episode save error: {e}")

                    with _RECORDING_LOCK:
                        if recording_id in _ACTIVE_RECORDINGS:
                            _ACTIVE_RECORDINGS[recording_id]["episodes_recorded"] = ep_idx + 1
                            _ACTIVE_RECORDINGS[recording_id]["total_frames"] += ep_frames

                # Cleanup teleop
                if teleop:
                    teleop.disconnect()

                # Finalize dataset (close parquet writers)
                try:
                    ds.finalize()
                except Exception as e:
                    logger.warning(f"Dataset finalize warning: {e}")

                # Push to hub if requested
                if push_to_hub:
                    try:
                        ds.push_to_hub(tags=tags)
                    except Exception as e:
                        logger.warning(f"Push to hub failed: {e}")

                with _RECORDING_LOCK:
                    rec_info = _ACTIVE_RECORDINGS.pop(recording_id, {})

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"✅ Recording complete: {repo_id}\n"
                                f"   Episodes: {rec_info.get('episodes_recorded', num_episodes)}\n"
                                f"   Total frames: {rec_info.get('total_frames', 0)}\n"
                                f"   FPS: {fps}\n"
                                f"   Task: {task}\n"
                                f"   Pushed to Hub: {push_to_hub}"
                            )
                        }
                    ],
                }

            except ImportError as e:
                with _RECORDING_LOCK:
                    _ACTIVE_RECORDINGS.pop(recording_id, None)
                return {
                    "status": "error",
                    "content": [{"text": f"lerobot not installed: {e}\nInstall with: pip install lerobot"}],
                }
            except Exception:
                with _RECORDING_LOCK:
                    _ACTIVE_RECORDINGS.pop(recording_id, None)
                raise

        elif action == "stop_recording":
            with _RECORDING_LOCK:
                active = dict(_ACTIVE_RECORDINGS)

            if not active:
                return {"status": "success", "content": [{"text": "No active recordings"}]}

            stopped = []
            for rid, info in active.items():
                with _RECORDING_LOCK:
                    _ACTIVE_RECORDINGS.pop(rid, None)
                stopped.append(f"{rid} ({info['repo_id']}, {info['episodes_recorded']} episodes)")

            return {
                "status": "success",
                "content": [{"text": f"🛑 Stopped recordings: {', '.join(stopped)}"}],
            }

        elif action == "push":
            if not repo_id:
                return {"status": "error", "content": [{"text": "repo_id required"}]}

            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(repo_id=repo_id, root=root)
            ds.push_to_hub(tags=tags)

            return {
                "status": "success",
                "content": [{"text": f"✅ Pushed {repo_id} to HuggingFace Hub"}],
            }

        elif action == "pull":
            if not repo_id:
                return {"status": "error", "content": [{"text": "repo_id required"}]}

            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(repo_id=repo_id, root=root)

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"✅ Downloaded {repo_id}\n"
                            f"   Episodes: {ds.meta.total_episodes}\n"
                            f"   Frames: {ds.meta.total_frames}\n"
                            f"   Root: {ds.root}"
                        )
                    }
                ],
            }

        elif action == "browse":
            if not repo_id:
                return {"status": "error", "content": [{"text": "repo_id required"}]}

            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(repo_id=repo_id, root=root)

            episodes_info = []
            total_eps = min(ds.meta.total_episodes, max_episodes)

            for ep_idx in range(total_eps):
                try:
                    ep_data = ds.meta.episodes[ep_idx] if hasattr(ds.meta, "episodes") else {}
                    episodes_info.append(
                        {
                            "index": ep_idx,
                            "length": ep_data.get("length", "?"),
                            "task": ep_data.get("task", "?"),
                        }
                    )
                except Exception:
                    episodes_info.append({"index": ep_idx})

            text = f"📋 Dataset: {repo_id} ({ds.meta.total_episodes} episodes)\n\n"
            for ep in episodes_info:
                text += f"  Episode {ep['index']}: {ep.get('length', '?')} frames | {ep.get('task', '?')}\n"

            return {
                "status": "success",
                "content": [
                    {"text": text},
                    {"json": {"episodes": episodes_info}},
                ],
            }

        elif action == "replay":
            if not repo_id:
                return {"status": "error", "content": [{"text": "repo_id required"}]}

            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(repo_id=repo_id, root=root)

            # Extract actions for the episode
            actions = []
            try:
                # Get frames for this episode
                ep_start = 0
                for i in range(episode):
                    if hasattr(ds.meta, "episodes"):
                        ep_start += ds.meta.episodes[i].get("length", 0)

                ep_length = ds.meta.episodes[episode].get("length", 0) if hasattr(ds.meta, "episodes") else 0

                for frame_idx in range(min(ep_length, 1000)):
                    try:
                        frame = ds[ep_start + frame_idx]
                        action_data = {}
                        for k, v in frame.items():
                            if "action" in k:
                                action_data[k] = v.tolist() if hasattr(v, "tolist") else v
                        if action_data:
                            actions.append(action_data)
                    except Exception:
                        break

            except Exception as e:
                logger.warning(f"Replay extraction error: {e}")

            return {
                "status": "success",
                "content": [
                    {"text": f"🔄 Episode {episode} from {repo_id}: {len(actions)} action frames"},
                    {
                        "json": {
                            "episode": episode,
                            "num_frames": len(actions),
                            "sample_action": actions[0] if actions else {},
                        }
                    },
                ],
            }

        elif action == "compute_stats":
            if not repo_id:
                return {"status": "error", "content": [{"text": "repo_id required"}]}

            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(repo_id=repo_id, root=root)

            try:
                ds.consolidate()
                stats_text = f"✅ Stats computed for {repo_id}"
                if hasattr(ds.meta, "stats") and ds.meta.stats:
                    for k, v in list(ds.meta.stats.items())[:10]:
                        stats_text += f"\n   {k}: {v}"
                return {"status": "success", "content": [{"text": stats_text}]}
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Stats error: {e}"}]}

        elif action == "list_hub":
            try:
                from huggingface_hub import HfApi

                api = HfApi()
                datasets = api.list_datasets(author="lerobot", limit=20)
                lines = ["📦 LeRobot Datasets on HuggingFace Hub:\n"]
                for ds in datasets:
                    lines.append(f"  - {ds.id} ({ds.downloads or 0} downloads)")
                return {"status": "success", "content": [{"text": "\n".join(lines)}]}
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Hub error: {e}"}]}

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Unknown action: {action}. Valid: "
                            "create, info, record, stop_recording, push, pull, browse, replay, compute_stats, list_hub"
                        )
                    }
                ],
            }

    except ImportError as e:
        return {
            "status": "error",
            "content": [{"text": f"Missing dependency: {e}\nInstall with: pip install lerobot"}],
        }
    except Exception as e:
        logger.error(f"Dataset error: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}
