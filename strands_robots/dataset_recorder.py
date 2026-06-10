"""LeRobotDataset recorder bridge for strands-robots.

Wraps LeRobotDataset so that both robot.py (real hardware) and
simulation.py (MuJoCo) can produce training-ready datasets with
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
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy check for LeRobot availability
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


class DatasetRecorder:
    """Bridge between strands-robots control loops and LeRobotDataset.

    Handles the full lifecycle:
    1. create() - build LeRobotDataset with correct features
    2. add_frame() - called every control step with obs + action
    3. save_episode() - finalize episode (encodes video, writes parquet)
    4. push_to_hub() - upload to HuggingFace

    Works for both real hardware (robot.py) and simulation (simulation.py).
    """

    def __init__(self, dataset, task: str = "", strict: bool = True):
        self.dataset = dataset
        self.default_task = task
        self.frame_count = 0
        self.episode_frame_count = 0  # frames in the CURRENT (unsaved) episode
        self.dropped_frame_count = 0
        self.strict = strict
        self.episode_count = 0
        self._closed = False
        self._cached_state_keys: list[str] | None = None
        self._cached_action_keys: list[str] | None = None

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int = 30,
        robot_type: str = "unknown",
        robot_features: dict[str, Any] | None = None,
        action_features: dict[str, Any] | None = None,
        camera_keys: list[str] | None = None,
        camera_dims: dict[str, tuple[int, int]] | None = None,
        joint_names: list[str] | None = None,
        task: str = "",
        root: str | None = None,
        use_videos: bool = True,
        vcodec: str = "libsvtav1",
        streaming_encoding: bool = True,
        image_writer_threads: int = 4,
        video_backend: str = "auto",
        video_width: int = 640,
        video_height: int = 480,
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
        # Lazy import - this is where we actually need lerobot
        LeRobotDatasetCls = _get_lerobot_dataset_class()

        # Build features dict in LeRobot format
        features = cls._build_features(
            robot_features=robot_features,
            action_features=action_features,
            camera_keys=camera_keys,
            camera_dims=camera_dims,
            joint_names=joint_names,
            use_videos=use_videos,
            video_width=video_width,
            video_height=video_height,
        )

        logger.info(f"Creating LeRobotDataset: {repo_id} @ {fps}fps, {len(features)} features, robot_type={robot_type}")

        # Build kwargs, skip unsupported params for this LeRobot version.
        create_kwargs = dict(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
        )
        import inspect

        create_sig = inspect.signature(LeRobotDatasetCls.create)
        create_params = create_sig.parameters

        # Video codec plumbing drifted across LeRobot versions:
        #   * 0.5.0/0.5.1: create(..., vcodec="libsvtav1")
        #   * 0.5.2+:      create(..., camera_encoder=VideoEncoderConfig(vcodec=...))
        # The flat ``vcodec`` kwarg was removed in 0.5.2 (codec now lives inside
        # VideoEncoderConfig). Detect which surface this LeRobot exposes and route
        # accordingly so the recorder works on both old and new versions.
        if "vcodec" in create_params:
            create_kwargs["vcodec"] = vcodec
        elif "camera_encoder" in create_params:
            try:
                from lerobot.configs.video import VideoEncoderConfig

                create_kwargs["camera_encoder"] = VideoEncoderConfig(vcodec=vcodec)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                # If VideoEncoderConfig can't be built (e.g. unknown codec on this
                # platform), fall back to the codec default rather than crashing.
                logger.warning("VideoEncoderConfig(vcodec=%r) unavailable (%s); using default encoder", vcodec, exc)

        # streaming_encoding / video_backend only in newer LeRobot versions
        if "streaming_encoding" in create_params:
            create_kwargs["streaming_encoding"] = streaming_encoding
        if "video_backend" in create_params:
            create_kwargs["video_backend"] = video_backend
        dataset = LeRobotDatasetCls.create(**create_kwargs)

        recorder = cls(dataset=dataset, task=task)
        logger.info("DatasetRecorder ready: %s", repo_id)
        return recorder

    @classmethod
    def resume(
        cls,
        repo_id: str,
        root: str | None = None,
        task: str = "",
        vcodec: str = "libsvtav1",
        streaming_encoding: bool = True,
        image_writer_threads: int = 4,
        video_backend: str = "auto",
    ) -> "DatasetRecorder":
        """Resume recording into an EXISTING LeRobotDataset (append episodes).

        Unlike :meth:`create` (which calls ``LeRobotDataset.create`` and
        hard-fails with ``FileExistsError`` if the dataset dir already
        exists), this opens an on-disk dataset via ``LeRobotDataset.resume``
        so further ``add_frame``/``save_episode`` calls append new episodes.

        This is the multi-episode data-collection path: ``start_recording``
        with ``overwrite=False`` on an existing dataset routes here instead of
        crashing. The plain ``LeRobotDataset(repo_id, root=...)`` constructor
        returns a READ-ONLY dataset (``add_frame`` raises), so ``resume()`` is
        the only correct append entry point in LeRobot 0.5.2+.

        Feature schema is inherited from the existing dataset on disk — the
        caller's joint/camera layout must match what was originally recorded.

        Args:
            repo_id: HuggingFace dataset ID (same as the original recording).
            root: Local dataset directory (same as the original recording).
            task: Default task description for appended frames.
            vcodec: Video codec (routed into ``camera_encoder`` on 0.5.2+).
            streaming_encoding: Stream-encode video during capture.
            image_writer_threads: Threads for writing image frames.
            video_backend: Video backend for encoding.

        Returns:
            A DatasetRecorder wrapping the resumed dataset.
        """
        import inspect

        LeRobotDatasetCls = _get_lerobot_dataset_class()

        if not hasattr(LeRobotDatasetCls, "resume"):
            # Older LeRobot (0.5.0/0.5.1) has no resume(); the append workflow
            # is unsupported there. Surface a clear error rather than a cryptic
            # read-only add_frame failure downstream.
            raise RuntimeError(
                "This LeRobot version has no LeRobotDataset.resume(); "
                "multi-episode append requires lerobot>=0.5.2. "
                "Use overwrite=True for a fresh single-session dataset."
            )

        resume_sig = inspect.signature(LeRobotDatasetCls.resume).parameters
        resume_kwargs: dict[str, Any] = dict(repo_id=repo_id, root=root)
        # Mirror create()'s version-tolerant codec routing.
        if "vcodec" in resume_sig:
            resume_kwargs["vcodec"] = vcodec
        elif "camera_encoder" in resume_sig:
            try:
                from lerobot.configs.video import VideoEncoderConfig

                resume_kwargs["camera_encoder"] = VideoEncoderConfig(vcodec=vcodec)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                logger.warning("VideoEncoderConfig(vcodec=%r) unavailable on resume (%s)", vcodec, exc)
        if "streaming_encoding" in resume_sig:
            resume_kwargs["streaming_encoding"] = streaming_encoding
        if "image_writer_threads" in resume_sig:
            resume_kwargs["image_writer_threads"] = image_writer_threads
        if "video_backend" in resume_sig:
            resume_kwargs["video_backend"] = video_backend

        dataset = LeRobotDatasetCls.resume(**resume_kwargs)
        recorder = cls(dataset=dataset, task=task)
        # Seed counters from the existing dataset so reporting reflects totals.
        try:
            recorder.episode_count = int(dataset.meta.total_episodes)
            recorder.frame_count = int(dataset.meta.total_frames)
        except Exception:  # noqa: BLE001 - counters are best-effort
            pass
        logger.info(
            "DatasetRecorder resumed: %s (%d existing episodes)",
            repo_id,
            recorder.episode_count,
        )
        return recorder

    @classmethod
    def _build_features(
        cls,
        robot_features: dict | None = None,
        action_features: dict | None = None,
        camera_keys: list[str] | None = None,
        camera_dims: dict[str, tuple[int, int]] | None = None,
        joint_names: list[str] | None = None,
        use_videos: bool = True,
        video_height: int = 480,
        video_width: int = 640,
    ) -> dict[str, Any]:
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

        # Observation: cameras → video/image features
        if camera_keys:
            camera_dims = camera_dims or {}
            for cam_name in camera_keys:
                key = f"observation.images.{cam_name}"
                dtype = "video" if use_videos else "image"
                # Per-camera (height, width). Falls back to the global
                # video_height/width when a camera has no explicit dims, so
                # callers that don't pass camera_dims keep the old behaviour.
                cam_h, cam_w = camera_dims.get(cam_name, (video_height, video_width))
                features[key] = {
                    "dtype": dtype,
                    "shape": (3, cam_h, cam_w),
                    "names": ["channels", "height", "width"],
                }

        # Observation: state (joint positions)
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

        # Action
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
        observation: dict[str, Any],
        action: dict[str, Any],
        task: str | None = None,
        camera_keys: list[str] | None = None,
    ) -> None:
        """Add a single control-loop frame to the dataset.

        This is the key method - called every step in the control loop.

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

        # Detect camera vs state keys
        if camera_keys is None:
            camera_keys = [k for k, v in observation.items() if isinstance(v, np.ndarray) and v.ndim >= 2]

        state_keys = [k for k in observation.keys() if k not in camera_keys]

        # Camera images → observation.images.{name}
        for cam_key in camera_keys:
            img = observation[cam_key]
            if isinstance(img, np.ndarray):
                # LeRobot expects HWC uint8 for add_frame
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                frame[f"observation.images.{cam_key}"] = img

        # State → observation.state (flattened vector)
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

        # Action → flattened vector
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

        # Task (mandatory for LeRobot v3)
        frame["task"] = task or self.default_task or "untitled"

        # Reconcile camera keys between frame and feature schema
        # Normalize namespaced camera keys (e.g. "arm0/wrist_cam" → "arm0__wrist_cam")
        # to match the schema declared in _build_features. MuJoCo uses "/" as a
        # namespace separator for multi-robot cameras, but LeRobot feature names
        # cannot contain "/" (reserved for nested-feature addressing).
        declared_cam_keys = {k for k in self.dataset.features if k.startswith("observation.images.")}
        frame_cam_keys = {k for k in list(frame.keys()) if k.startswith("observation.images.")}
        for cam_key in frame_cam_keys:
            normalized = cam_key.replace("/", "__")
            if normalized != cam_key and normalized in declared_cam_keys:
                frame[normalized] = frame.pop(cam_key)

        # Strip undeclared cameras (keys present in obs but not registered in
        # _build_features). This avoids LeRobot's "Extra features" error.
        # Declared-but-missing cameras (e.g. when a render fails) are left alone -
        # LeRobot tolerates absent columns and the episode simply won't have that
        # camera's data.
        frame_cam_keys_final = {k for k in frame if k.startswith("observation.images.")}
        for extra in frame_cam_keys_final - declared_cam_keys:
            del frame[extra]

        # Add to dataset
        try:
            self.dataset.add_frame(frame)
            self.frame_count += 1
            self.episode_frame_count += 1
        except Exception as e:
            if self.strict:
                raise  # Fail-fast per AGENTS.md convention #5
            self.dropped_frame_count += 1
            n = self.dropped_frame_count
            # Log at 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, then every 1000
            if (n & (n - 1)) == 0 or n % 1000 == 0:
                logger.warning(
                    "add_frame failed (frame %d, dropped %d): %s",
                    self.frame_count,
                    self.dropped_frame_count,
                    e,
                )

    def save_episode(self) -> dict[str, Any]:
        """Finalize current episode - writes parquet, encodes video, computes stats.

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
            # Report frames in THIS episode, not the cumulative total.
            # frame_count is monotonic across all episodes; episode_frame_count
            # is the count since the last save. Reset it after reporting.
            ep_frames = self.episode_frame_count
            total_frames = self.frame_count
            self.episode_frame_count = 0
            logger.info(
                "Episode %d saved: %d frames (%d total across dataset)",
                self.episode_count,
                ep_frames,
                total_frames,
            )
            return {
                "status": "success",
                "episode": self.episode_count,
                "episode_frames": ep_frames,
                "total_frames": total_frames,
            }
        except Exception as e:
            logger.error("save_episode failed: %s", e)
            # Mark recorder as poisoned — the LeRobot episode buffer is in
            # undefined state after a failed save. Subsequent add_frame calls
            # would silently corrupt the dataset. Close to prevent drift.
            self._closed = True
            return {"status": "error", "message": f"save_episode failed (recorder closed): {e}"}

    def clear_episode_buffer(self) -> bool:
        """Discard frames buffered for the current (unsaved) episode.

        After an aborted recording (e.g. a policy returned an empty action
        chunk mid-loop) the open episode buffer still holds the frames written
        so far. Without discarding them, the next ``add_frame`` appends to the
        half-episode and the eventual ``save_episode`` flushes a Frankenstein
        episode that mixes two runs. Call this to start the next episode at
        frame 0.

        LeRobot's buffer-reset surface drifted across 0.5.x, so this routes
        version-tolerantly:
          * ``LeRobotDataset.clear_episode_buffer()`` if exposed (preferred), else
          * reset via ``create_episode_buffer()`` if exposed, else
          * leave the buffer in place and warn (caller must ``stop_recording`` /
            ``save_episode`` to drain it before recording again).

        Returns:
            True if the buffer was actively cleared; False if no clear surface
            was available (a warning is logged in that case).
        """
        cleared = False
        try:
            if hasattr(self.dataset, "clear_episode_buffer"):
                self.dataset.clear_episode_buffer()
                cleared = True
            elif hasattr(self.dataset, "create_episode_buffer"):
                self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                cleared = True
        except Exception as e:  # noqa: BLE001 - best-effort discard; never mask the original abort
            logger.warning("clear_episode_buffer failed: %s", e)
            cleared = False

        # Reset the per-episode frame counter regardless: the next episode
        # reports frames from 0. frame_count (cumulative) is left untouched
        # since those frames were really written to disk only on save_episode.
        self.episode_frame_count = 0

        if not cleared:
            logger.warning(
                "Could not auto-discard the partial episode buffer on this "
                "LeRobot version; call stop_recording()/save_episode() to drain "
                "it before the next recording to avoid a mixed episode."
            )
        return cleared

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
        tags: list[str] | None = None,
        private: bool = False,
    ) -> dict[str, Any]:
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


# Shared replay-episode helpers


def load_lerobot_episode(repo_id: str, episode: int = 0, root: str | None = None):
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
