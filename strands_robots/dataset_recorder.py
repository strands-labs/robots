"""LeRobotDataset recorder bridge for strands-robots.

Wraps LeRobotDataset so that both robot.py (real hardware) and
MujocoBackend can produce training-ready datasets with
a single add_frame() call per control step.

Usage:
    recorder = DatasetRecorder.create(
        repo_id="user/my_dataset",
        fps=30,
        robot_features=robot.observation_features,
        action_features=robot.action_features,
        task="pick up the red cube",
    )
    # In control loop:
    recorder.add_frame(observation, action, task="pick up the red cube")
    # End of episode:
    recorder.save_episode()
    # Optionally:
    recorder.push_to_hub()
"""

import functools
import logging
import sys
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy check for LeRobot availability ──────────────────────────────
# We must NOT import lerobot at module level because it pulls in
# `datasets` → `pandas`, which can crash with a numpy ABI mismatch on
# systems where the system pandas was compiled against an older numpy
# (e.g. JetPack / Jetson with system pandas 2.1.4 + pip numpy 2.x).


@functools.lru_cache(maxsize=1)
def has_lerobot_dataset() -> bool:
    """Check if lerobot is available. Result is cached after first call."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401

        return True
    except (ImportError, ValueError, RuntimeError) as exc:
        logger.debug("lerobot not available: %s", exc)
        return False


def _get_lerobot_dataset_class():
    """Import and return LeRobotDataset class, or raise ImportError.

    Supports test mocking: if ``strands_robots.dataset_recorder.LeRobotDataset``
    has been set (by a test mock), returns that class directly.
    """
    # Support test mocking: check module-level overrides
    this_module = sys.modules[__name__]

    # If a test injected a mock LeRobotDataset class, use it
    mock_cls = getattr(this_module, "LeRobotDataset", None)
    if mock_cls is not None:
        return mock_cls

    # Actual import
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except (ImportError, ValueError, RuntimeError) as exc:
        raise ImportError(
            f"lerobot not available ({exc}). Install with: pip install lerobot\nRequired for LeRobotDataset recording."
        ) from exc


def _numpy_ify(v):
    """Convert any value to numpy-friendly format for add_frame."""
    if hasattr(v, "numpy"):
        return v.numpy()
    if hasattr(v, "tolist") and isinstance(v, np.ndarray):
        return v
    if isinstance(v, (int, float)):
        return np.array([v], dtype=np.float32)
    if isinstance(v, list):
        return np.array(v, dtype=np.float32)
    return v


class DatasetRecorder:
    """Bridge between strands-robots control loops and LeRobotDataset.

    Handles the full lifecycle:
    1. create() — build LeRobotDataset with correct features
    2. add_frame() — called every control step with obs + action
    3. save_episode() — finalize episode (encodes video, writes parquet)
    4. push_to_hub() — upload to HuggingFace

    Works for both real hardware (robot.py) and simulation backends.
    """

    def __init__(self, dataset, task: str = ""):
        self.dataset = dataset
        self.default_task = task
        self.frame_count = 0
        self.dropped_frame_count = 0
        self.episode_count = 0
        self._closed = False
        self._cached_state_keys: Optional[list[str]] = None
        self._cached_action_keys: Optional[list[str]] = None

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int = 30,
        robot_type: str = "unknown",
        robot_features: Optional[Dict[str, Any]] = None,
        action_features: Optional[Dict[str, Any]] = None,
        camera_keys: Optional[List[str]] = None,
        joint_names: Optional[List[str]] = None,
        task: str = "",
        root: Optional[str] = None,
        use_videos: bool = True,
        vcodec: str = "libsvtav1",
        streaming_encoding: bool = True,
        image_writer_threads: int = 4,
        video_backend: str = "auto",
    ) -> "DatasetRecorder":
        """Create a new DatasetRecorder with auto-detected features.

        Args:
            repo_id: HuggingFace dataset ID (e.g. "user/my_dataset")
            fps: Recording frame rate
            robot_type: Robot type string (e.g. "so100", "panda")
            robot_features: Dict of observation feature names → types
                (from robot.observation_features or sim joint names)
            action_features: Dict of action feature names → types
            camera_keys: List of camera names (images become video features)
            joint_names: List of joint names (alternative to robot_features for sim)
            task: Default task description
            root: Local directory for dataset storage
            use_videos: Encode camera frames as video (True) or keep as images
            vcodec: Video codec (h264, hevc, libsvtav1)
            streaming_encoding: Stream-encode video during capture
            image_writer_threads: Threads for writing image frames
            video_backend: Video backend for encoding ("auto" for HW encoder auto-detect)
        """
        # Lazy import — this is where we actually need lerobot
        LeRobotDatasetCls = _get_lerobot_dataset_class()

        # Build features dict in LeRobot format
        features = cls._build_features(
            robot_features=robot_features,
            action_features=action_features,
            camera_keys=camera_keys,
            joint_names=joint_names,
            use_videos=use_videos,
        )

        logger.info(f"Creating LeRobotDataset: {repo_id} @ {fps}fps, {len(features)} features, robot_type={robot_type}")

        # Build kwargs, skip unsupported params for this LeRobot version
        create_kwargs = dict(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
            vcodec=vcodec,
        )
        # streaming_encoding only in newer LeRobot versions
        import inspect

        create_sig = inspect.signature(LeRobotDatasetCls.create)
        if "streaming_encoding" in create_sig.parameters:
            create_kwargs["streaming_encoding"] = streaming_encoding
        if "video_backend" in create_sig.parameters:
            create_kwargs["video_backend"] = video_backend
        dataset = LeRobotDatasetCls.create(**create_kwargs)

        recorder = cls(dataset=dataset, task=task)
        logger.info("DatasetRecorder ready: %s", repo_id)
        return recorder

    @classmethod
    def _build_features(
        cls,
        robot_features: Optional[Dict] = None,
        action_features: Optional[Dict] = None,
        camera_keys: Optional[List[str]] = None,
        joint_names: Optional[List[str]] = None,
        use_videos: bool = True,
    ) -> Dict[str, Any]:
        """Build LeRobot v3-compatible features dict.

        LeRobot v3 features format:
        {
            "observation.images.camera_name": {"dtype": "video", "shape": (C, H, W), "names": [...]},
            "observation.state": {"dtype": "float32", "shape": (N,), "names": [...]},
            "action": {"dtype": "float32", "shape": (N,), "names": [...]},
        }

        Note: "names" must be a flat list of strings, NOT a dict like {"motors": [...]}.
        """
        features = {}

        # --- Observation: cameras → video/image features ---
        if camera_keys:
            for cam_name in camera_keys:
                key = f"observation.images.{cam_name}"
                dtype = "video" if use_videos else "image"
                features[key] = {
                    "dtype": dtype,
                    "shape": (
                        3,
                        480,
                        640,
                    ),  # CHW default, actual shape set on first frame
                    "names": ["channels", "height", "width"],
                }

        # --- Observation: state (joint positions) ---
        state_dim = 0
        state_names = []
        if robot_features:
            # Count scalar features (exclude cameras)
            state_keys = [
                k
                for k, v in robot_features.items()
                if not isinstance(v, dict) or v.get("dtype") not in ("image", "video")
            ]
            state_dim = len(state_keys)
            state_names = state_keys
        elif joint_names:
            state_dim = len(joint_names)
            state_names = list(joint_names)

        if state_dim > 0:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": state_names,
            }

        # --- Action ---
        action_dim = 0
        action_names = []
        if action_features:
            action_keys = [
                k
                for k, v in action_features.items()
                if not isinstance(v, dict) or v.get("dtype") not in ("image", "video")
            ]
            action_dim = len(action_keys)
            action_names = action_keys
        elif joint_names:
            action_dim = len(joint_names)
            action_names = list(joint_names)
        elif state_dim > 0:
            action_dim = state_dim  # Same dim as state by default
            action_names = state_names[:]

        if action_dim > 0:
            features["action"] = {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": action_names[:action_dim],
            }

        return features

    def add_frame(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        task: Optional[str] = None,
        camera_keys: Optional[List[str]] = None,
    ) -> None:
        """Add a single control-loop frame to the dataset.

        This is the key method — called every step in the control loop.

        Args:
            observation: Raw observation dict from robot/sim
                (joint_name → float, camera_name → np.ndarray)
            action: Action dict (joint_name → float)
            task: Task description (uses default if None)
            camera_keys: Which keys in observation are camera images
        """
        if self._closed:
            return

        frame = {}

        # --- Detect camera vs state keys ---
        if camera_keys is None:
            camera_keys = [k for k, v in observation.items() if isinstance(v, np.ndarray) and v.ndim >= 2]

        state_keys = [k for k in observation.keys() if k not in camera_keys]

        # --- Camera images → observation.images.{name} ---
        for cam_key in camera_keys:
            img = observation[cam_key]
            if isinstance(img, np.ndarray):
                # LeRobot expects HWC uint8 for add_frame
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                frame[f"observation.images.{cam_key}"] = img

        # --- State → observation.state (flattened vector) ---
        # Use feature schema ordering to match the dataset schema declared in _build_features().
        if state_keys:
            state_vals = []
            if self._cached_state_keys is None:
                feat = self.dataset.features.get("observation.state", {})
                state_names = feat.get("names", []) if isinstance(feat, dict) else getattr(feat, "names", [])
                self._cached_state_keys = state_names if state_names else sorted(state_keys)

            for k in self._cached_state_keys:
                v = observation.get(k)
                if v is None:
                    state_vals.append(0.0)
                elif isinstance(v, (int, float)):
                    state_vals.append(float(v))
                elif isinstance(v, np.ndarray) and v.ndim == 0:
                    state_vals.append(float(v))
                elif isinstance(v, (list, np.ndarray)):
                    arr = np.asarray(v, dtype=np.float32).flatten()
                    state_vals.extend(arr.tolist())
            if state_vals:
                frame["observation.state"] = np.array(state_vals, dtype=np.float32)

        # --- Action → flattened vector ---
        # Use feature schema ordering for actions too.
        if action:
            action_vals = []
            if self._cached_action_keys is None:
                feat = self.dataset.features.get("action", {})
                action_names = feat.get("names", []) if isinstance(feat, dict) else getattr(feat, "names", [])
                self._cached_action_keys = action_names if action_names else sorted(action.keys())

            for k in self._cached_action_keys:
                v = action.get(k)
                if v is None:
                    action_vals.append(0.0)
                elif isinstance(v, (int, float)):
                    action_vals.append(float(v))
                elif isinstance(v, np.ndarray) and v.ndim == 0:
                    action_vals.append(float(v))
                elif isinstance(v, (list, np.ndarray)):
                    arr = np.asarray(v, dtype=np.float32).flatten()
                    action_vals.extend(arr.tolist())
            if action_vals:
                frame["action"] = np.array(action_vals, dtype=np.float32)

        # --- Task (mandatory for LeRobot v3) ---
        frame["task"] = task or self.default_task or "untitled"

        # --- Add to dataset ---
        try:
            self.dataset.add_frame(frame)
            self.frame_count += 1
        except Exception as e:
            self.dropped_frame_count += 1
            logger.warning("add_frame failed (frame %d, dropped %d): %s", self.frame_count, self.dropped_frame_count, e)

    def save_episode(self) -> Dict[str, Any]:
        """Finalize current episode — writes parquet, encodes video, computes stats.

        LeRobot v3: save_episode() takes no task argument. Tasks are stored
        per-frame in the episode buffer via add_frame().

        Returns:
            Dict with episode info
        """
        if self._closed:
            return {"status": "error", "message": "Recorder closed"}

        try:
            self.dataset.save_episode()
            self.episode_count += 1
            ep_frames = self.frame_count  # Total frames so far
            logger.info(f"Episode {self.episode_count} saved: {ep_frames} total frames")
            return {
                "status": "success",
                "episode": self.episode_count,
                "total_frames": ep_frames,
            }
        except Exception as e:
            logger.error("save_episode failed: %s", e)
            return {"status": "error", "message": str(e)}

    def finalize(self) -> None:
        """Finalize the dataset (close parquet writers, flush metadata)."""
        if self._closed:
            return
        try:
            self.dataset.finalize()
        except Exception as e:
            logger.warning("finalize warning: %s", e)
        self._closed = True

    def push_to_hub(
        self,
        tags: Optional[List[str]] = None,
        private: bool = False,
    ) -> Dict[str, Any]:
        """Push dataset to HuggingFace Hub.

        Args:
            tags: Optional tags for the dataset
            private: Upload as private dataset

        Returns:
            Dict with push status
        """
        try:
            self.dataset.push_to_hub(tags=tags, private=private)
            logger.info("Dataset pushed to hub: %s", self.dataset.repo_id)
            return {
                "status": "success",
                "repo_id": self.dataset.repo_id,
                "episodes": self.episode_count,
                "frames": self.frame_count,
            }
        except Exception as e:
            logger.error("push_to_hub failed: %s", e)
            return {"status": "error", "message": str(e)}

    @property
    def repo_id(self) -> str:
        return self.dataset.repo_id

    @property
    def root(self) -> str:
        return str(self.dataset.root)

    def __repr__(self) -> str:
        return f"DatasetRecorder(repo_id={self.repo_id}, episodes={self.episode_count}, frames={self.frame_count})"


# ── Shared replay-episode helpers ────────────────────────────────────


def load_lerobot_episode(repo_id: str, episode: int = 0, root: Optional[str] = None):
    """Load a LeRobotDataset and resolve the frame range for an episode.

    Returns:
        Tuple of (dataset, episode_start, episode_length) on success.

    Raises:
        ImportError: If lerobot is not installed.
        ValueError: If the episode is out of range or has no frames.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=root)

    num_episodes = ds.meta.total_episodes if hasattr(ds.meta, "total_episodes") else len(ds.meta.episodes)
    if episode >= num_episodes:
        raise ValueError(f"Episode {episode} out of range (0-{num_episodes - 1})")

    episode_start = 0
    episode_length = 0
    try:
        if hasattr(ds, "episode_data_index"):
            from_idx = ds.episode_data_index["from"][episode].item()
            to_idx = ds.episode_data_index["to"][episode].item()
            episode_start = from_idx
            episode_length = to_idx - from_idx
        else:
            for i in range(episode):
                ep_info = ds.meta.episodes[i] if hasattr(ds.meta, "episodes") else {}
                episode_start += ep_info.get("length", 0)
            ep_info = ds.meta.episodes[episode] if hasattr(ds.meta, "episodes") else {}
            episode_length = ep_info.get("length", 0)
    except Exception:
        # Last resort: scan frames to find episode boundaries
        for idx in range(len(ds)):
            frame = ds[idx]
            frame_ep = frame.get("episode_index", -1) if hasattr(frame, "get") else -1
            if hasattr(frame_ep, "item"):
                frame_ep = frame_ep.item()
            if frame_ep == episode:
                if episode_length == 0:
                    episode_start = idx
                episode_length += 1
            elif episode_length > 0:
                break

    if episode_length == 0:
        raise ValueError(f"Episode {episode} has no frames")

    return ds, episode_start, episode_length
