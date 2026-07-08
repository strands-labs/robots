"""LeRobotDataset recorder bridge for strands-robots.

Wraps LeRobotDataset so that both real hardware (:mod:`strands_robots.robot`)
and simulation (:mod:`strands_robots.simulation`) can produce training-ready
datasets with
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

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Every LeRobot surface in the supported ``>=0.5.0,<0.6.0`` range validates the
# codec against the same codec-name allowlist - ``video_utils.VALID_VIDEO_CODECS
# = {"h264", "hevc", "libsvtav1", "auto"} | HW_ENCODERS`` - and *rejects* the
# ffmpeg library names ("libx264"/"libx265"). This holds for the flat ``vcodec``
# kwarg (0.5.0/0.5.1) just as much as for the ``RGBEncoderConfig`` /
# ``VideoEncoderConfig`` surfaces a later minor may expose. So there is exactly
# one correct normalization direction: ffmpeg name -> codec name, applied to
# whichever surface is present. Callers may pass either spelling.
_ENCODER_CODEC_NAMES = {"libx264": "h264", "libx265": "hevc"}


def _codec_create_kwargs(sig_params: Any, vcodec: str, *, context: str = "create") -> dict[str, Any]:
    """Map a flat ``vcodec`` onto whatever codec surface this LeRobot exposes.

    LeRobot routes the codec differently across releases. Passing the wrong
    kwarg (or none) silently leaves the encoder on its built-in default, so the
    caller's ``vcodec`` is dropped without error - which is how an AV1 default
    sneaks back in even when the caller asked for H.264:

      * 0.5.0 / 0.5.1: a flat ``create/resume(..., vcodec=...)`` kwarg.
      * an interim build briefly exposed
        ``camera_encoder=VideoEncoderConfig(vcodec=...)``.
      * a later minor may move the codec into
        ``rgb_encoder=RGBEncoderConfig(vcodec=...)``.

    Every one of these surfaces validates against LeRobot's codec-name allowlist
    (``{"h264", "hevc", "libsvtav1", "auto"} | HW_ENCODERS``) and rejects the
    ffmpeg library names ("libx264"/"libx265"). So the codec is normalized to its
    codec-name spelling once and routed onto whichever surface is present. The
    caller may pass either spelling ("h264" or "libx264"). An unknown codec
    raises loudly (LeRobot's own ValueError) rather than silently falling back
    to the default.

    Args:
        sig_params: ``inspect.Signature.parameters`` of ``create``/``resume``.
        vcodec: Requested codec, as a codec name ("h264", "hevc", "libsvtav1")
            or an ffmpeg encoder name ("libx264", "libx265").
        context: Label for warning messages ("create" or "resume").

    Returns:
        Mapping with exactly one of ``vcodec`` / ``rgb_encoder`` /
        ``camera_encoder``, or an empty dict if no known codec surface is
        present (recorder falls back to the LeRobot default codec).

    Raises:
        ValueError: When the codec surface rejects the requested codec
            (propagated from LeRobot so an unsupported codec fails loudly
            instead of silently reverting to the default).
    """
    # ffmpeg library name -> codec name; codec names (and HW encoders) pass through.
    codec = _ENCODER_CODEC_NAMES.get(vcodec, vcodec)
    if "vcodec" in sig_params:
        return {"vcodec": codec}
    if "rgb_encoder" in sig_params:
        try:
            from lerobot.configs.video import RGBEncoderConfig
        except ImportError as exc:
            logger.warning("RGBEncoderConfig import failed on %s (%s); using default codec", context, exc)
            return {}
        return {"rgb_encoder": RGBEncoderConfig(vcodec=codec)}
    if "camera_encoder" in sig_params:
        try:
            from lerobot.configs.video import VideoEncoderConfig
        except ImportError as exc:
            logger.warning("VideoEncoderConfig import failed on %s (%s); using default codec", context, exc)
            return {}
        return {"camera_encoder": VideoEncoderConfig(vcodec=codec)}
    return {}


# Allowlist patterns for HF Storage Bucket sync targets. Both `bucket` and
# `run_id` reach the `hf` CLI argv and the `hf://buckets/...` URI; they are
# agent-reachable via stop_recording(bucket=, run_id=) dispatched through the
# simulation action layer, so they MUST be validated before any subprocess /
# URI interpolation (AGENTS.md > LLM Input Safety). `bucket` is "name" or
# "org/name"; `run_id` is a single path segment. Neither may contain shell
# metacharacters, path-traversal (".."), or separators beyond the one allowed
# bucket "org/name" slash.
_BUCKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Lazy check for LeRobot availability.
# We must NOT import lerobot at module level because it pulls in
# `datasets` -> `pandas`, which can crash with a numpy ABI mismatch on
# systems where the system pandas was compiled against an older numpy
# (e.g. JetPack / Jetson with system pandas 2.1.4 + pip numpy 2.x).
#
# Only the POSITIVE result is cached: importing lerobot is expensive (it pulls
# in `datasets` -> `pandas`) and worth doing once, but lerobot availability is
# a process capability that can transiently fail to resolve (a slow/locked
# import, or - in tests - a temporarily monkeypatched `sys.modules`). Caching a
# `False` would permanently disable recording for the rest of the process even
# after the condition clears, so a failed probe is re-attempted on the next
# call.
#
# The cache is a single-cell list rather than a module-level bool: a non-empty
# cell means "probed True". Mutating the cell (append/clear) records the result
# without rebinding a module global, so the memoization needs no ``global``
# statement.
_HAS_LEROBOT_DATASET: list[bool] = []


def has_lerobot_dataset() -> bool:
    """Return True if lerobot's ``LeRobotDataset`` can be imported.

    A successful probe is cached; a failed probe is intentionally re-attempted
    on the next call so a transient import failure does not permanently disable
    recording for the process.
    """
    if _HAS_LEROBOT_DATASET:
        return True
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except (ImportError, ValueError, RuntimeError) as exc:
        logger.debug("lerobot not available: %s", exc)
        return False
    _HAS_LEROBOT_DATASET.append(True)
    return True


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

    Works for both real hardware (:mod:`strands_robots.robot`) and
    simulation (:mod:`strands_robots.simulation`).
    """

    def __init__(
        self,
        dataset,
        task: str = "",
        strict: bool = True,
        camera_key_map: dict[str, str] | None = None,
    ):
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
        # Ordered SOURCE keys to read observation.state from, when they diverge
        # from the flat schema ``names`` (e.g. a floating base's ``base_quat``
        # source key expands to 4 per-component ``base_quat.*`` schema names).
        # ``None`` -> derive the read order from the schema names (scalar-only).
        self._state_source_keys: list[str] | None = None
        # Optional remap of observed camera stream names -> declared schema
        # names. Keys/values are bare camera names (no "observation.images."
        # prefix); a leading prefix on either side is tolerated and stripped.
        # Lets callers reconcile a policy's declared image_keys (e.g. "image",
        # "wrist_image") with differently-named sim/hardware streams (e.g.
        # "front_camera", "wrist_camera") instead of silently dropping frames.
        self.camera_key_map = self._normalize_camera_key_map(camera_key_map)
        # One-shot guard so the camera-key-mismatch diagnostic is logged once
        # per recorder instead of every control step (50Hz would flood logs).
        self._warned_camera_mismatch = False

    @staticmethod
    def _normalize_camera_key_map(camera_key_map: dict[str, str] | None) -> dict[str, str]:
        """Normalize a camera key remap to bare-name -> bare-name form.

        Accepts entries written either as bare camera names ("front_camera")
        or as fully-qualified feature keys ("observation.images.front_camera")
        on EITHER side, and strips the "observation.images." prefix so the map
        can be applied uniformly against bare camera names in add_frame.

        Args:
            camera_key_map: Caller-supplied remap, or None.

        Returns:
            A dict mapping observed bare camera name -> declared bare camera
            name (empty dict when no map was supplied).
        """
        prefix = "observation.images."
        normalized: dict[str, str] = {}
        for src, dst in (camera_key_map or {}).items():
            src_bare = src[len(prefix) :] if src.startswith(prefix) else src
            dst_bare = dst[len(prefix) :] if dst.startswith(prefix) else dst
            normalized[src_bare] = dst_bare
        return normalized

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
        action_names: list[str] | None = None,
        extra_state_specs: list[tuple[str, list[str]]] | None = None,
        task: str = "",
        root: str | None = None,
        use_videos: bool = True,
        vcodec: str = "h264",
        streaming_encoding: bool = True,
        image_writer_threads: int = 4,
        video_backend: str = "auto",
        video_width: int = 640,
        video_height: int = 480,
        camera_key_map: dict[str, str] | None = None,
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
            action_names: Explicit action-column names. Use when the action
                space diverges from the joint names (e.g. actuator short-names
                from SimEngine.robot_action_keys, where a tendon gripper is an
                actuator with no matching joint). Falls back to joint_names.
            extra_state_specs: Optional vector state signals to append to the
                observation.state schema as per-component scalar columns, as
                ``(source_key, [component_suffixes])`` pairs. Each source key is
                read from the observation and flattened in order (e.g. a
                floating base's ``("base_quat", ["w","x","y","z"])`` adds four
                ``base_quat.*`` columns). Scalar joint/action fallbacks ignore
                these; add_frame reads the source keys, not the expanded names.
            task: Default task description
            root: Local directory for dataset storage
            use_videos: Encode camera frames as video (True) or keep as images
            vcodec: Video codec for the per-camera MP4 streams. Defaults to
                "h264" (H.264), which is universally decodable - including
                by OpenCV's VideoCapture / FFmpeg build, used by many
                downstream VLM video readers. Use "libsvtav1" (AV1) for
                smaller files in storage-constrained training pipelines;
                LeRobot read-back (torchcodec/pyav) handles AV1, but OpenCV
                wheels commonly cannot decode it and silently yield 0 frames.
            streaming_encoding: Stream-encode video during capture
            image_writer_threads: Threads for writing image frames
            video_backend: Video backend for encoding ("auto" for HW encoder auto-detect)
            camera_key_map: Optional remap of observed camera stream names to the
                declared schema names (e.g. {"front_camera": "image",
                "wrist_camera": "wrist_image"}). Bare names or fully-qualified
                "observation.images.*" keys are accepted on either side. Use it
                when a policy declares image_keys that differ from the names the
                sim/hardware streams emit, otherwise those frames are dropped.
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
            action_names=action_names,
            extra_state_specs=extra_state_specs,
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

        # Route the requested codec onto whichever surface this LeRobot version
        # exposes (vcodec / rgb_encoder / camera_encoder). The flat ``vcodec``
        # kwarg and ``camera_encoder`` were both removed in 0.5.2; the codec now
        # lives inside ``rgb_encoder=RGBEncoderConfig(vcodec=...)``. Missing this
        # surface silently leaves the encoder on its built-in default.
        create_kwargs.update(_codec_create_kwargs(create_params, vcodec, context="create"))

        # streaming_encoding / video_backend only in newer LeRobot versions
        if "streaming_encoding" in create_params:
            create_kwargs["streaming_encoding"] = streaming_encoding
        if "video_backend" in create_params:
            create_kwargs["video_backend"] = video_backend
        dataset = LeRobotDatasetCls.create(**create_kwargs)

        recorder = cls(dataset=dataset, task=task, camera_key_map=camera_key_map)
        # When vector state specs expand the schema (e.g. a floating base),
        # add_frame must read the SOURCE keys (``base_quat``) rather than the
        # expanded per-component schema names (``base_quat.w``). Record the
        # source order: scalar state keys first, then each vector source key.
        if extra_state_specs:
            scalar_source = list(joint_names) if joint_names else []
            recorder._state_source_keys = scalar_source + [k for k, _ in extra_state_specs]
        logger.info("DatasetRecorder ready: %s", repo_id)
        return recorder

    @classmethod
    def resume(
        cls,
        repo_id: str,
        root: str | None = None,
        task: str = "",
        vcodec: str = "h264",
        streaming_encoding: bool = True,
        image_writer_threads: int = 4,
        video_backend: str = "auto",
        camera_key_map: dict[str, str] | None = None,
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

        Feature schema is inherited from the existing dataset on disk - the
        caller's joint/camera layout must match what was originally recorded.

        Args:
            repo_id: HuggingFace dataset ID (same as the original recording).
            root: Local dataset directory (same as the original recording).
            task: Default task description for appended frames.
            vcodec: Video codec for the per-camera MP4 streams (default
                "h264"; routed into the version-appropriate encoder
                config). See create() for the H.264-vs-AV1 trade-off.
            streaming_encoding: Stream-encode video during capture.
            image_writer_threads: Threads for writing image frames.
            video_backend: Video backend for encoding.
            camera_key_map: Optional remap of observed camera stream names to
                the declared schema names (see create()).

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
        resume_kwargs.update(_codec_create_kwargs(resume_sig, vcodec, context="resume"))
        if "streaming_encoding" in resume_sig:
            resume_kwargs["streaming_encoding"] = streaming_encoding
        if "image_writer_threads" in resume_sig:
            resume_kwargs["image_writer_threads"] = image_writer_threads
        if "video_backend" in resume_sig:
            resume_kwargs["video_backend"] = video_backend

        dataset = LeRobotDatasetCls.resume(**resume_kwargs)
        recorder = cls(dataset=dataset, task=task, camera_key_map=camera_key_map)
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
        action_names: list[str] | None = None,
        extra_state_specs: list[tuple[str, list[str]]] | None = None,
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

        # Preserve additional vector state signals (e.g. a floating base's
        # ``base_quat`` / ``base_ang_vel``) as per-component scalar columns so
        # they are not silently dropped from ``observation.state``. Each spec is
        # ``(source_key, [component_suffixes])``; ``add_frame`` reads the source
        # key from the observation and flattens it in this same order. The
        # scalar joint/action fallbacks below use the pre-expansion dims.
        scalar_state_dim = state_dim
        scalar_state_names = list(state_names)
        if extra_state_specs:
            for src_key, comps in extra_state_specs:
                state_names.extend(f"{src_key}.{c}" for c in comps)
                state_dim += len(comps)

        if state_dim > 0:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": state_names,
            }

        # Action. Prefer an explicit action_names list (the actuator keys the
        # rollout loops emit, which can diverge from the joint names - see
        # SimEngine.robot_action_keys) so the recorded action columns match the
        # keys add_frame receives. Fall back to action_features, then to the
        # joint/state names for robots whose actuators mirror their joints.
        action_dim = 0
        action_col_names: list[str] = []
        if action_features:
            action_keys = [
                k
                for k, v in action_features.items()
                if not isinstance(v, dict) or v.get("dtype") not in ("image", "video")
            ]
            action_dim = len(action_keys)
            action_col_names = action_keys
        elif action_names:
            action_dim = len(action_names)
            action_col_names = list(action_names)
        elif joint_names:
            action_dim = len(joint_names)
            action_col_names = list(joint_names)
        elif scalar_state_dim > 0:
            action_dim = scalar_state_dim  # Same dim as scalar state by default
            action_col_names = scalar_state_names[:]

        if action_dim > 0:
            features["action"] = {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": action_col_names[:action_dim],
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
                if self._state_source_keys is not None:
                    # Vector-expanded schema: read SOURCE keys (e.g. ``base_quat``);
                    # the list/ndarray branch below flattens each in schema order.
                    self._cached_state_keys = list(self._state_source_keys)
                else:
                    feat = self.dataset.features.get("observation.state", {})
                    state_names = feat.get("names", []) if isinstance(feat, dict) else getattr(feat, "names", [])
                    self._cached_state_keys = state_names if state_names else sorted(state_keys)

            for k in self._cached_state_keys:
                v = observation.get(k)
                if v is None:
                    state_vals.append(0.0)
                elif isinstance(v, (int, float)):
                    state_vals.append(float(v))
                elif isinstance(v, (np.generic, np.ndarray)) and v.ndim == 0:
                    # numpy scalars (np.float32/np.int32 from indexing a MuJoCo
                    # qpos/ctrl array) and 0-dim arrays are scalar state values.
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
                elif isinstance(v, (np.generic, np.ndarray)) and v.ndim == 0:
                    # numpy scalars (np.float32/np.int32) and 0-dim arrays are
                    # scalar action values - see state branch above.
                    action_vals.append(float(v))
                elif isinstance(v, (list, np.ndarray)):
                    arr = np.asarray(v, dtype=np.float32).flatten()
                    action_vals.extend(arr.tolist())
            if action_vals:
                frame["action"] = np.array(action_vals, dtype=np.float32)

        # Task (mandatory for LeRobot v3)
        frame["task"] = task or self.default_task or "untitled"

        # Reconcile camera keys between frame and feature schema
        declared_cam_keys = {k for k in self.dataset.features if k.startswith("observation.images.")}

        # Apply the caller-supplied remap FIRST (observed stream name -> declared
        # schema name). This is the explicit escape hatch for the case where a
        # policy declares image_keys (e.g. "image"/"wrist_image") that differ
        # from the names the sim/hardware streams emit (e.g. "front_camera"/
        # "wrist_camera"). Without it those frames are stripped below and the
        # dataset records no image columns.
        if self.camera_key_map:
            prefix = "observation.images."
            for cam_key in [k for k in list(frame.keys()) if k.startswith(prefix)]:
                bare = cam_key[len(prefix) :]
                mapped = self.camera_key_map.get(bare)
                if mapped is not None and mapped != bare:
                    frame[f"{prefix}{mapped}"] = frame.pop(cam_key)

        # Normalize namespaced camera keys (e.g. "arm0/wrist_cam" → "arm0__wrist_cam")
        # to match the schema declared in _build_features. MuJoCo uses "/" as a
        # namespace separator for multi-robot cameras, but LeRobot feature names
        # cannot contain "/" (reserved for nested-feature addressing).
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
        stripped_cam_keys = frame_cam_keys_final - declared_cam_keys
        for extra in stripped_cam_keys:
            del frame[extra]

        # Surface the silent data-loss case: camera frames arrived but NONE of
        # them matched a declared schema key, so every image is being dropped
        # and the dataset will record zero image columns. This is the
        # "image_keys never match the streams" failure mode that otherwise
        # produces episodes with no video and no error. Warn once per recorder
        # (50Hz would flood) with the observed-vs-declared keys and the
        # camera_key_map remedy. A PARTIAL strip (some cameras matched) is left
        # quiet - that is the normal "ignore an extra debug camera" path.
        if (
            not self._warned_camera_mismatch
            and stripped_cam_keys
            and declared_cam_keys
            and not (frame_cam_keys_final & declared_cam_keys)
        ):
            self._warned_camera_mismatch = True
            logger.warning(
                "DatasetRecorder: none of the observed camera streams %s match the "
                "declared image features %s - all image frames are being dropped and "
                "this dataset will have no video. Pass camera_key_map={observed: declared} "
                "to remap (e.g. {%r: %r}), or declare cameras with names matching the streams.",
                sorted(k[len("observation.images.") :] for k in stripped_cam_keys),
                sorted(k[len("observation.images.") :] for k in declared_cam_keys),
                next(iter(sorted(stripped_cam_keys)))[len("observation.images.") :],
                next(iter(sorted(declared_cam_keys)))[len("observation.images.") :],
            )

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
            # Mark recorder as poisoned - the LeRobot episode buffer is in
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

        Refuses to publish an empty dataset (no frames written or no episode
        saved). Pushing then would create a Hub repo containing only
        ``meta/info.json`` (no parquet, no video) and silently pollute the
        namespace. The ``stop_recording`` facade has its own empty-dataset
        guard; this protects the direct-API path (and any caller that reaches
        ``push_to_hub`` after a rollout that never fed the recorder).

        Returns:
            Dict with push status. ``status="error"`` (no Hub call made) when
            the dataset is empty.
        """
        if self.frame_count == 0 or self.episode_count == 0:
            msg = (
                f"refusing to push empty dataset {self.dataset.repo_id} "
                f"({self.frame_count} frames, {self.episode_count} episodes) - "
                "would create a Hub repo with only meta/info.json. Record frames "
                "with add_frame and flush at least one episode with save_episode "
                "before push_to_hub."
            )
            logger.error("push_to_hub aborted: %s", msg)
            return {"status": "error", "message": msg}
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

    def sync_to_bucket(
        self,
        bucket: str,  # "my-org/robot-fave"
        run_id: str | None = None,  # subpath; defaults to dataset name
        *,
        create: bool = True,
        private: bool = True,
        delete: bool = False,
    ) -> dict[str, Any]:
        """Sync the on-disk LeRobotDataset into an HF Storage Bucket (Phase 1/2).

        Mutable, Xet-deduplicated dump target for COLLECTION - avoids git-LFS
        history bloat of push_to_hub during recording. Daily re-sync uploads
        only changed chunks (content-defined chunking). Requires the ``hf`` CLI
        (huggingface_hub>=1.x) and ``hf auth login``.

        ``bucket`` and ``run_id`` are validated against an allowlist before any
        subprocess or URI interpolation: ``bucket`` must be ``"name"`` or
        ``"org/name"`` and ``run_id`` a single path segment, both restricted to
        ``[A-Za-z0-9._-]`` (no path traversal or shell metacharacters). This
        path is agent-reachable via ``stop_recording(bucket=, run_id=)``. A
        rejected value returns ``{"status": "error", ...}`` without running ``hf``.

        The shard layout is already Xet/bucket-friendly at the 100 MB default,
        and ``meta/`` MUST ship or downstream loses normalization stats.
        """
        import shutil
        import subprocess

        if shutil.which("hf") is None:
            return {
                "status": "error",
                "message": "`hf` CLI not found. pip install -U huggingface_hub (>=1.x) and run `hf auth login`.",
            }

        if not _BUCKET_RE.match(bucket):
            return {
                "status": "error",
                "message": f"invalid bucket {bucket!r}: must match "
                "'name' or 'org/name' using [A-Za-z0-9._-] (no path traversal "
                "or shell metacharacters).",
            }

        local_root = str(self.dataset.root)
        # meta/ must ship or downstream loses normalization stats.
        if not (Path(local_root) / "meta").exists():
            return {
                "status": "error",
                "message": f"No meta/ under {local_root}; call finalize() before "
                "sync_to_bucket (stats/info required for streaming/training).",
            }

        run_id = run_id or self.dataset.repo_id.split("/")[-1]
        if not _RUN_ID_RE.match(run_id):
            return {
                "status": "error",
                "message": f"invalid run_id {run_id!r}: must be a single path "
                "segment using [A-Za-z0-9._-] (no '/', path traversal, or shell "
                "metacharacters).",
            }
        dest = f"hf://buckets/{bucket}/{run_id}"

        if create:
            cp = subprocess.run(
                ["hf", "buckets", "create", bucket] + (["--private"] if private else []),
                capture_output=True,
                text=True,
            )
            blob = (cp.stderr + cp.stdout).lower()
            if cp.returncode != 0 and "exist" not in blob:
                return {
                    "status": "error",
                    "message": f"bucket create failed: {cp.stderr.strip()}",
                }

        cmd = ["hf", "sync", local_root, dest]
        if delete:
            cmd.append("--delete")
        logger.info("Syncing %s -> %s", local_root, dest)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return {
                "status": "error",
                "message": proc.stderr.strip() or proc.stdout.strip(),
            }

        return {
            "status": "success",
            "bucket_uri": dest,
            "episodes": self.episode_count,
            "frames": self.frame_count,
        }

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
        ValueError: If the episode index is negative, out of range, or the
            resolved episode has no frames.
    """
    if episode < 0:
        raise ValueError(f"Episode index must be non-negative, got {episode}")

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


def read_dataset_episode_indices(root: str | Path) -> dict[str, Any]:
    """Read episode-level ground truth from a LeRobot v3 dataset on disk.

    Parses every ``meta/episodes/**/*.parquet`` file under ``root`` and returns
    the recorded episode index set plus per-episode frame counts. This is the
    parquet source of truth used by :meth:`SimEngine.verify_dataset_episodes`
    to confirm a recording session produced the number of distinct episodes the
    caller intended (rather than one merged ``episode_index=0`` mega-episode).

    Pure ``pyarrow`` read - it does NOT import ``lerobot`` or instantiate a
    ``LeRobotDataset`` (which would re-validate/scan the whole dataset). Reads
    only the lightweight episode metadata parquet.

    Args:
        root: Dataset root directory (the dir that contains ``meta/``).

    Returns:
        Dict with:
          - ``episode_indices``: sorted list of distinct ``episode_index`` values.
          - ``total_episodes``: number of distinct episodes (``len`` of above).
          - ``total_frames``: sum of per-episode ``length`` (0 if unavailable).
          - ``frames_per_episode``: per-episode frame counts aligned to
            ``episode_indices`` (empty list if the ``length`` column is absent).
          - ``info_total_episodes``: the ``total_episodes`` recorded in
            ``meta/info.json`` (``None`` if that file is absent or unreadable).
            Returned alongside the parquet truth so callers can cross-check the
            two metadata sources for agreement - a healthy dataset has
            ``info_total_episodes == total_episodes``.

    Raises:
        ImportError: If ``pyarrow`` is not installed.
        FileNotFoundError: If no ``meta/episodes`` parquet exists under ``root``
            (no episode was ever flushed - the dataset is empty/unfinalized).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - pyarrow ships with lerobot
        raise ImportError("read_dataset_episode_indices requires pyarrow (installed with the lerobot extra).") from e

    root_path = Path(root)
    parquet_files = sorted((root_path / "meta" / "episodes").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No meta/episodes parquet under {root_path}. The dataset is empty or was "
            "never finalized (episodes are flushed to parquet at stop_recording/finalize)."
        )

    pairs: list[tuple[int, int]] = []
    seen: set[int] = set()
    for pf in parquet_files:
        table = pq.read_table(pf)
        cols = table.column_names
        if "episode_index" not in cols:
            continue
        data = table.to_pydict()
        ep_indices = data["episode_index"]
        lengths = data.get("length")
        for i, ep in enumerate(ep_indices):
            ep_int = int(ep)
            if ep_int in seen:
                continue
            seen.add(ep_int)
            length = int(lengths[i]) if lengths is not None and lengths[i] is not None else 0
            pairs.append((ep_int, length))

    pairs.sort(key=lambda p: p[0])
    episode_indices = [p[0] for p in pairs]
    frames_per_episode = [p[1] for p in pairs]
    has_lengths = any(f > 0 for f in frames_per_episode)

    # Read meta/info.json total_episodes as a second, independent metadata
    # source. A healthy LeRobot dataset has info.json.total_episodes equal to
    # the distinct episode count in the parquet; a mismatch means the dataset
    # is internally inconsistent (e.g. an interrupted finalize), which
    # verify_dataset_episodes surfaces. Absent/corrupt info.json -> None (the
    # parquet remains the ground truth and is still reported).
    info_total_episodes: int | None = None
    info_path = root_path / "meta" / "info.json"
    if info_path.is_file():
        try:
            with info_path.open() as f:
                info_total_episodes = int(json.load(f)["total_episodes"])
        except (OSError, ValueError, KeyError, TypeError):
            info_total_episodes = None

    return {
        "episode_indices": episode_indices,
        "total_episodes": len(episode_indices),
        "total_frames": sum(frames_per_episode) if has_lengths else 0,
        "frames_per_episode": frames_per_episode if has_lengths else [],
        "info_total_episodes": info_total_episodes,
    }
