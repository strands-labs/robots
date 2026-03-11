"""GR00T Policy Implementation — supports both N1.5 (ZMQ service) and N1.6 (local inference).

Two modes:
  1. Service mode (N1.5/N1.6): Connect to a running GR00T inference service via ZMQ/HTTP
  2. Local mode (N1.5/N1.6):   Load model directly on GPU — no Docker/service needed

Service mode uses ExternalRobotInferenceClient (ZMQ).
Local mode uses Isaac-GR00T's Gr00tPolicy directly on device.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .. import Policy
from .data_config import DATA_CONFIG_MAP, BaseDataConfig, load_data_config

# Convenience aliases for common import patterns
Gr00tDataConfig = BaseDataConfig


def _get_client_class():
    from .client import ExternalRobotInferenceClient

    return ExternalRobotInferenceClient


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect which Isaac-GR00T version is available
# ---------------------------------------------------------------------------
_GROOT_VERSION: Optional[str] = None  # "n1.5", "n1.6", or None


def _detect_groot_version() -> Optional[str]:
    """Auto-detect installed Isaac-GR00T version."""
    global _GROOT_VERSION
    if _GROOT_VERSION is not None:
        return _GROOT_VERSION

    # Try N1.6 first (newer)
    try:
        from gr00t.data.embodiment_tags import (
            EmbodimentTag as _EmbodimentTag,
        )
        from gr00t.policy.gr00t_policy import Gr00tPolicy as _Gr00tPolicy

        _GROOT_VERSION = "n1.6"
        logger.info("Detected GR00T N1.6 (gr00t.policy.gr00t_policy)")
        return _GROOT_VERSION
    except ImportError:
        pass

    # Try N1.5
    try:
        from gr00t.model.policy import Gr00tPolicy as _

        _GROOT_VERSION = "n1.5"
        logger.info("Detected GR00T N1.5 (gr00t.model.policy)")
        return _GROOT_VERSION
    except ImportError:
        pass

    logger.debug("No local Isaac-GR00T found — service mode only")
    return None


class Gr00tPolicy(Policy):
    """GR00T policy supporting N1.5, N1.6, service mode, and local inference.

    Args:
        data_config:      String name (e.g. "so100_dualcam") or BaseDataConfig object.
        host/port:        For service (ZMQ) mode — connects to running inference server.
        model_path:       For local mode — loads model on GPU directly.
        embodiment_tag:   Embodiment tag (N1.5: string, N1.6: EmbodimentTag enum).
        denoising_steps:  Diffusion denoising steps (default 4).
        device:           "cuda" or "cpu".
        groot_version:    Force "n1.5" or "n1.6" (auto-detected if None).

    Examples:
        # Service mode (works without Isaac-GR00T installed)
        policy = Gr00tPolicy(data_config="so100_dualcam", host="localhost", port=5555)

        # Local N1.5
        policy = Gr00tPolicy(
            data_config="so100_dualcam",
            model_path="/data/checkpoints/gr00t-fruit-6k/checkpoint-6000",
        )

        # Local N1.6 (base model)
        policy = Gr00tPolicy(
            data_config="so100_dualcam",
            model_path="nvidia/GR00T-N1-2B",
            groot_version="n1.6",
        )
    """

    def __init__(
        self,
        data_config: Union[str, BaseDataConfig] = "so100_dualcam",
        host: str = "localhost",
        port: int = 5555,
        model_path: Optional[str] = None,
        embodiment_tag: str = "new_embodiment",
        denoising_steps: int = 4,
        device: str = "cuda",
        groot_version: Optional[str] = None,
        **kwargs,
    ):
        # Load data config
        self.data_config = load_data_config(data_config)
        self.data_config_name = (
            data_config if isinstance(data_config, str) else type(data_config).__name__
        )

        # Extract modality keys
        self.camera_keys = self.data_config.video_keys
        self.state_keys = self.data_config.state_keys
        self.action_keys = self.data_config.action_keys
        self.language_keys = self.data_config.language_keys
        self.robot_state_keys: List[str] = []

        # Decide mode: local vs service
        self._local_policy = None
        self._client = None
        self._mode = "service"
        self._groot_version = groot_version or _detect_groot_version()

        if model_path is not None:
            # Local mode — load model on GPU
            self._mode = "local"
            self._load_local_policy(model_path, embodiment_tag, denoising_steps, device)
        else:
            # Service mode — ZMQ client
            self._mode = "service"
            try:
                self._client = _get_client_class()(host=host, port=port)
            except Exception as e:
                raise ImportError(
                    f"GR00T service client init failed: {e}. "
                    f"Install: pip install msgpack pyzmq"
                ) from e

        logger.info(
            f"🧠 GR00T Policy [{self._mode}] v={self._groot_version or 'service-only'}"
        )
        logger.info("   config=%s cameras=%s", self.data_config_name, self.camera_keys)

    # ------------------------------------------------------------------
    # Local model loading
    # ------------------------------------------------------------------

    def _load_local_policy(
        self, model_path: str, embodiment_tag: str, denoising_steps: int, device: str
    ):
        """Load GR00T model locally (N1.5 or N1.6)."""
        ver = self._groot_version

        if ver == "n1.6":
            self._load_n16(model_path, embodiment_tag, denoising_steps, device)
        elif ver == "n1.5":
            self._load_n15(model_path, embodiment_tag, denoising_steps, device)
        else:
            raise ImportError(
                "Isaac-GR00T not installed. Cannot use local mode. "
                "Either install Isaac-GR00T or use service mode (host/port)."
            )

    def _load_n15(
        self, model_path: str, embodiment_tag: str, denoising_steps: int, device: str
    ):
        """Load N1.5 model using gr00t.model.policy.Gr00tPolicy."""
        from gr00t.experiment.data_config import DATA_CONFIG_MAP as N15_CONFIGS
        from gr00t.model.policy import Gr00tPolicy as N15Policy

        # Try to get native N1.5 config for proper transforms
        cfg_name = (
            self.data_config_name
            if isinstance(self.data_config_name, str)
            else "so100_dualcam"
        )
        native_cfg = N15_CONFIGS.get(cfg_name)

        if native_cfg:
            mc = native_cfg.modality_config()
            mt = native_cfg.transform()
        else:
            # Fallback: build from our data_config
            mc = self.data_config.modality_config()
            mt = None
            logger.warning(
                f"No native N1.5 config for '{cfg_name}', transforms may be missing"
            )

        kwargs = {
            "model_path": model_path,
            "embodiment_tag": embodiment_tag,
            "modality_config": mc,
            "modality_transform": mt,
            "denoising_steps": denoising_steps,
            "device": device,
        }
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        self._local_policy = N15Policy(**kwargs)
        logger.info("✅ GR00T N1.5 loaded from %s", model_path)

    def _load_n16(
        self, model_path: str, embodiment_tag: str, denoising_steps: int, device: str
    ):
        # NOTE: GR00T N1.6 requires Python 3.10 + the gr00t package from NVIDIA Isaac-GR00T.
        # The Eagle-Block2A-2B-v2 vision backbone needs a config.json that isn't shipped
        # in the pip package. Generate it with these verified dimensions (from L40S testing):
        #   Vision: 27 layers, hidden=1152, intermediate=4304, SigLIP2, image=224, patch=14
        #   LLM: 16 layers, hidden=2048, intermediate=6144, Qwen3 (qk_norm), vocab=151680
        # Also needs: flash-attn, tokenizer files (Qwen2), chat_template
        # See docs/policies/groot.md for full setup instructions.
        """Load N1.6 model using gr00t.policy.gr00t_policy.Gr00tPolicy."""
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy as N16Policy

        # Map string to EmbodimentTag enum
        tag = getattr(
            EmbodimentTag, embodiment_tag.upper(), EmbodimentTag.NEW_EMBODIMENT
        )

        self._local_policy = N16Policy(
            embodiment_tag=tag,
            model_path=model_path,
            device=device,
        )
        logger.info("✅ GR00T N1.6 loaded from %s", model_path)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "groot"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(
        self, observation_dict: Dict[str, Any], instruction: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Get actions from GR00T (service or local mode)."""

        # Build observation in GR00T format
        obs_dict = self._build_observation(observation_dict, instruction)

        if self._mode == "local":
            action_chunk = self._local_inference(obs_dict)
        else:
            action_chunk = self._client.get_action(obs_dict)

        return self._convert_to_robot_actions(action_chunk)

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_observation(
        self, observation_dict: Dict[str, Any], instruction: str
    ) -> dict:
        """Build GR00T-formatted observation dict."""
        obs = {}

        # Camera observations
        for video_key in self.camera_keys:
            camera_key = self._map_video_key_to_camera(video_key, observation_dict)
            if camera_key and camera_key in observation_dict:
                obs[video_key] = observation_dict[camera_key]

        # State observations
        robot_state = np.array(
            [observation_dict.get(k, 0.0) for k in self.robot_state_keys]
        )
        self._map_robot_state_to_gr00t_state(obs, robot_state)

        # Language
        if self.language_keys:
            obs[self.language_keys[0]] = instruction

        # Add batch dim for service mode
        if self._mode == "service":
            for k in obs:
                if isinstance(obs[k], np.ndarray):
                    obs[k] = obs[k][np.newaxis, ...]
                else:
                    obs[k] = [obs[k]]

        return obs

    def _local_inference(self, obs_dict: dict) -> dict:
        """Run local inference (N1.5 or N1.6)."""
        ver = self._groot_version

        if ver == "n1.5":
            # N1.5: flat dict with batch dim, returns dict of arrays
            batched = {}
            for k, v in obs_dict.items():
                if isinstance(v, np.ndarray):
                    batched[k] = (
                        v[np.newaxis, ...] if v.ndim < 4 else v[np.newaxis, ...]
                    )
                else:
                    batched[k] = [v] if not isinstance(v, list) else v
            return self._local_policy.get_action(batched)

        elif ver == "n1.6":
            # N1.6: nested dict {video: {key: arr}, state: {key: arr}, language: {key: [[str]]}}
            # Vendor expects shapes: video=(B,T,H,W,C), state=(B,T,D), language=[[str]]
            # We add both batch (B=1) and temporal (T=1) dimensions.
            mc = self._local_policy.get_modality_config()
            nested = {"video": {}, "state": {}, "language": {}}

            video_mc = getattr(mc, "video", None)
            video_keys = getattr(video_mc, "modality_keys", []) if video_mc else []

            for key in video_keys:
                short = key.split(".")[-1] if "." in key else key
                # Try multiple source key patterns
                src_key = None
                for candidate in [f"video.{short}", key, short]:
                    if candidate in obs_dict:
                        src_key = candidate
                        break
                if src_key:
                    arr = obs_dict[src_key]
                    if isinstance(arr, np.ndarray):
                        # Ensure (B=1, T=1, H, W, C) — add batch AND temporal dims
                        if arr.ndim == 3:  # (H, W, C) → (1, 1, H, W, C)
                            arr = arr[np.newaxis, np.newaxis, ...]
                        elif arr.ndim == 4:  # (B, H, W, C) → (B, 1, H, W, C)
                            arr = arr[:, np.newaxis, ...]
                        nested["video"][key] = arr
                    else:
                        nested["video"][key] = arr

            state_mc = getattr(mc, "state", None)
            state_keys = getattr(state_mc, "modality_keys", []) if state_mc else []

            for key in state_keys:
                short = key.split(".")[-1] if "." in key else key
                # Try multiple source key patterns
                src_key = None
                for candidate in [f"state.{short}", key, short]:
                    if candidate in obs_dict:
                        src_key = candidate
                        break
                if src_key:
                    arr = obs_dict[src_key]
                    if isinstance(arr, np.ndarray):
                        arr = arr.astype(np.float32)
                        # Ensure (B=1, T=1, D) — add batch AND temporal dims
                        if arr.ndim == 1:  # (D,) → (1, 1, D)
                            arr = arr[np.newaxis, np.newaxis, ...]
                        elif arr.ndim == 2:  # (B, D) → (B, 1, D)
                            arr = arr[:, np.newaxis, ...]
                        nested["state"][key] = arr
                    else:
                        nested["state"][key] = arr

            lang_mc = getattr(mc, "language", None)
            lang_keys = getattr(lang_mc, "modality_keys", []) if lang_mc else []

            for key in lang_keys:
                short = key.split(".")[-1] if "." in key else key
                lang_key = self.language_keys[0] if self.language_keys else key
                # Look for lang value in obs_dict by full key, short key, or our config key
                lang_value = None
                for candidate in [lang_key, key, short, f"language.{short}"]:
                    if candidate in obs_dict:
                        lang_value = obs_dict[candidate]
                        break
                if lang_value is None:
                    # Default instruction
                    lang_value = "Perform the task"
                # N1.6 expects [[str]] format
                if isinstance(lang_value, str):
                    nested["language"][key] = [[lang_value]]
                elif isinstance(lang_value, list):
                    nested["language"][key] = (
                        lang_value if isinstance(lang_value[0], list) else [lang_value]
                    )
                else:
                    nested["language"][key] = [[str(lang_value)]]

            actions, _ = self._local_policy.get_action(nested)
            # N1.6 returns {key: arr} without "action." prefix — add it
            return {
                f"action.{k}" if not k.startswith("action.") else k: v
                for k, v in actions.items()
            }

        raise RuntimeError(f"Unknown GR00T version: {ver}")

    # ------------------------------------------------------------------
    # Mapping helpers (unchanged)
    # ------------------------------------------------------------------

    def _map_video_key_to_camera(self, video_key: str, observation_dict: dict) -> str:
        camera_name = video_key.replace("video.", "")
        if camera_name in observation_dict:
            return camera_name
        mapping = {
            "webcam": ["webcam", "front", "wrist", "main"],
            "front": ["front", "webcam", "top", "ego_view", "main"],
            "wrist": ["wrist", "hand", "end_effector", "gripper"],
            "ego_view": ["front", "ego_view", "webcam", "main"],
            "top": ["top", "overhead", "front"],
            "side": ["side", "lateral", "left", "right"],
        }
        for name in mapping.get(camera_name, [camera_name]):
            if name in observation_dict:
                return name
        camera_keys = [k for k in observation_dict if not k.startswith("state")]
        return camera_keys[0] if camera_keys else None

    def _map_robot_state_to_gr00t_state(self, obs_dict: dict, robot_state: np.ndarray):
        name = (
            self.data_config_name.lower()
            if isinstance(self.data_config_name, str)
            else ""
        )
        # N1.6 requires float32 (vendor asserts dtype); N1.5 used float64.
        _dtype = np.float32
        if "so100" in name or "so101" in name:
            if len(robot_state) >= 6:
                obs_dict["state.single_arm"] = robot_state[:5].astype(_dtype)
                obs_dict["state.gripper"] = robot_state[5:6].astype(_dtype)
        elif "fourier_gr1" in name:
            if len(robot_state) >= 14:
                obs_dict["state.left_arm"] = robot_state[:7].astype(_dtype)
                obs_dict["state.right_arm"] = robot_state[7:14].astype(_dtype)
        elif "unitree_g1" in name:
            # G1 full body: left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + right_arm(7)
            #               + left_hand(7) + right_hand(7) = 43 DOF total
            # N1.6 verified DOF layout from UNITREE_G1 EmbodimentTag
            g1_layout = [
                ("state.left_leg", 6),
                ("state.right_leg", 6),
                ("state.waist", 3),
                ("state.left_arm", 7),
                ("state.right_arm", 7),
                ("state.left_hand", 7),
                ("state.right_hand", 7),
            ]
            idx = 0
            for key, dim in g1_layout:
                if idx + dim <= len(robot_state):
                    obs_dict[key] = robot_state[idx : idx + dim].astype(_dtype)
                else:
                    # Pad with zeros if robot_state is shorter (e.g. arms-only mode)
                    obs_dict[key] = np.zeros(dim, dtype=_dtype)
                idx += dim
        elif self.state_keys and len(robot_state) > 0:
            obs_dict[self.state_keys[0]] = robot_state.astype(_dtype)

    def _convert_to_robot_actions(self, action_chunk: dict) -> List[Dict[str, Any]]:
        """Convert raw GR00T action output to robot action dicts.

        N1.6 returns: {group_name: (1, horizon, dim)} or {group_name: (horizon, dim)}
        e.g. {"left_leg": (1, 16, 6), "right_leg": (1, 16, 6), "waist": (1, 16, 3), ...}

        We flatten across groups and map back to robot_state_keys per timestep.
        """
        # Normalize keys: strip "action." prefix if present
        normalized = {}
        for k, v in action_chunk.items():
            clean_key = k.replace("action.", "") if k.startswith("action.") else k
            if hasattr(v, "shape"):
                # Remove batch dim if present: (1, H, D) → (H, D)
                arr = v
                while arr.ndim > 2:
                    arr = arr[0]
                normalized[clean_key] = arr  # (horizon, dim)
            else:
                normalized[clean_key] = np.atleast_2d(v)

        if not normalized:
            return []

        # Get action horizon from first array
        first = next(iter(normalized.values()))
        action_horizon = first.shape[0] if first.ndim >= 1 else 1

        # Build per-timestep action dicts
        robot_actions = []
        for i in range(action_horizon):
            # Concatenate all groups at timestep i
            parts = []
            for ak in normalized:
                v = normalized[ak]
                parts.append(np.atleast_1d(v[i] if v.ndim >= 1 else v))
            concat = np.concatenate(parts, axis=0) if parts else np.zeros(6)

            # Map to robot_state_keys
            action_dict = {}
            if self.robot_state_keys:
                for j, key in enumerate(self.robot_state_keys):
                    action_dict[key] = float(concat[j]) if j < len(concat) else 0.0
            else:
                # No robot_state_keys set — return per-group dict
                for ak in normalized:
                    v = normalized[ak]
                    row = v[i] if v.ndim >= 1 else v
                    action_dict[ak] = (
                        row.tolist() if hasattr(row, "tolist") else list(row)
                    )

            robot_actions.append(action_dict)
        return robot_actions


__all__ = ["Gr00tPolicy", "Gr00tDataConfig", "BaseDataConfig"]
