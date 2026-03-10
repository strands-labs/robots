#!/usr/bin/env python3
"""
DreamGen / GR00T-Dreams Policy Integration

Integrates NVIDIA GR00T-Dreams (DreamGen) models as strands-robots policies.
Supports both the IDM (Inverse Dynamics Model) for real-time inference from
frame pairs, and the full Gr00tPolicy from GR00T-Dreams for VLA inference.

DreamGen pipeline:
1. Video world model generates synthetic robot videos
2. IDM extracts pseudo-actions from consecutive frames
3. Neural trajectories (video + pseudo-actions) train downstream policies

This policy wraps:
- IDM: Given two frames, predicts action chunks between them (real-time use)
- Gr00tPolicy (Dreams): Full VLA with GR00T N1 model + transforms

Usage:
    # IDM-based inference (from frame pairs)
    policy = create_policy(
        provider="dreamgen",
        mode="idm",
        model_path="nvidia/gr00t-idm-so100",
    )

    # Full GR00T-Dreams VLA
    policy = create_policy(
        provider="dreamgen",
        mode="vla",
        model_path="nvidia/GR00T-N1.6-3B",
        embodiment_tag="new_embodiment",
        modality_config={...},
    )
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .. import Policy

logger = logging.getLogger(__name__)


class DreamgenPolicy(Policy):
    """DreamGen policy supporting IDM and full VLA inference modes.

    This policy integrates the GR00T-Dreams pipeline into strands-robots,
    enabling:
    - IDM inference: Predict action chunks from consecutive camera frames
    - VLA inference: Full vision-language-action model from GR00T-Dreams
    - Neural trajectory generation: Create synthetic training data

    The IDM mode is particularly useful for:
    - Real-time action prediction from video observations
    - Generating pseudo-actions for neural trajectory pipelines
    - Lightweight inference without full VLA overhead
    """

    def __init__(
        self,
        model_path: str = "nvidia/gr00t-idm-so100",
        mode: str = "idm",
        embodiment_tag: Optional[str] = None,
        modality_config: Optional[Dict] = None,
        modality_transform: Optional[Any] = None,
        device: str = "cuda",
        action_horizon: int = 16,
        action_dim: int = 6,
        denoising_steps: Optional[int] = None,
        sliding_window: bool = True,
        **kwargs,
    ):
        """Initialize DreamGen policy.

        Args:
            model_path: HuggingFace model ID or local path to model checkpoint
            mode: Inference mode - "idm" (inverse dynamics) or "vla" (full VLA)
            embodiment_tag: Robot embodiment tag (required for VLA mode)
            modality_config: Modality configuration dict (required for VLA mode)
            modality_transform: Modality transform pipeline (required for VLA mode)
            device: Inference device (cuda, cpu)
            action_horizon: Number of actions to predict per inference
            action_dim: Action dimensionality
            denoising_steps: Number of denoising steps (overrides model default)
            sliding_window: Use sliding window for IDM action extraction
        """
        self.model_path = model_path
        self.mode = mode
        self.embodiment_tag = embodiment_tag
        self.modality_config_dict = modality_config
        self.modality_transform = modality_transform
        self.device = device
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.sliding_window = sliding_window
        self.robot_state_keys: List[str] = []

        # Model state (lazy loaded)
        self._model = None
        self._policy = None
        self._previous_frame = None

        logger.info(f"🎬 DreamGen Policy: mode={mode}")
        logger.info(f"🧠 Model: {model_path}")
        logger.info(f"⚡ Action horizon: {action_horizon}, dim: {action_dim}")

    @property
    def provider_name(self) -> str:
        return "dreamgen"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        """Set robot state keys from connected robot."""
        self.robot_state_keys = robot_state_keys
        logger.info(f"🔧 DreamGen state keys: {self.robot_state_keys}")

    def _load_idm(self):
        """Lazy-load IDM model."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
            from transformers import AutoConfig, AutoModel  # noqa: F401

            # Register IDM model type (GR00T-Dreams registers this)
            try:
                from gr00t.model.idm import IDM, IDMConfig  # noqa: F401
            except ImportError:
                logger.warning("GR00T-Dreams IDM not importable, trying AutoModel")

            logger.info(f"🔄 Loading IDM from {self.model_path}...")
            self._model = AutoModel.from_pretrained(self.model_path)
            self._model.eval()
            self._model.to(self.device)
            logger.info(f"✅ IDM loaded on {self.device}")

        except Exception as e:
            raise ImportError(
                f"Failed to load DreamGen IDM model: {e}. " f"Install GR00T-Dreams: pip install gr00t-dreams"
            ) from e

    def _load_vla(self):
        """Lazy-load full VLA policy from GR00T-Dreams."""
        if self._policy is not None:
            return

        try:
            from gr00t.model.policy import Gr00tPolicy as DreamsGr00tPolicy

            if not self.embodiment_tag:
                raise ValueError("embodiment_tag required for VLA mode")
            if not self.modality_config_dict:
                raise ValueError("modality_config required for VLA mode")
            if not self.modality_transform:
                raise ValueError("modality_transform required for VLA mode")

            logger.info(f"🔄 Loading GR00T-Dreams VLA from {self.model_path}...")
            self._policy = DreamsGr00tPolicy(
                model_path=self.model_path,
                embodiment_tag=self.embodiment_tag,
                modality_config=self.modality_config_dict,
                modality_transform=self.modality_transform,
                denoising_steps=self.denoising_steps,
                device=self.device,
            )
            logger.info(f"✅ GR00T-Dreams VLA loaded on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"GR00T-Dreams not available: {e}. " f"Install: pip install gr00t-dreams or clone GR00T-Dreams repo"
            ) from e

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from DreamGen model.

        For IDM mode: Uses current + previous frame to predict action chunk.
        For VLA mode: Uses full observation + instruction for VLA inference.

        Args:
            observation_dict: Robot observations (cameras + state)
            instruction: Natural language instruction
            **kwargs: Additional parameters

        Returns:
            List of action dictionaries for robot execution
        """
        if self.mode == "idm":
            return await self._get_idm_actions(observation_dict, instruction)
        elif self.mode == "vla":
            return await self._get_vla_actions(observation_dict, instruction)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'idm' or 'vla'")

    async def _get_idm_actions(self, observation_dict: Dict[str, Any], instruction: str) -> List[Dict[str, Any]]:
        """Get actions using IDM (Inverse Dynamics Model).

        The IDM takes two consecutive frames and predicts the action chunk
        that transitions between them. Uses a sliding window approach.
        """
        import torch

        self._load_idm()

        # Extract current camera frame
        current_frame = self._extract_frame(observation_dict)
        if current_frame is None:
            logger.warning("⚠️ No camera frame found for IDM")
            return self._generate_zero_actions()

        # Need previous frame for IDM (frame pair → actions)
        if self._previous_frame is None:
            self._previous_frame = current_frame
            logger.debug("📸 First frame captured, waiting for next frame")
            return self._generate_zero_actions()

        try:
            # Build IDM input: two frames as video tensor
            # IDM expects: video shape (B, 2, H, W, C) uint8
            video = np.stack([self._previous_frame, current_frame], axis=0)
            video = video[np.newaxis, ...]  # Add batch dim: (1, 2, H, W, C)

            inputs = {"video": video}

            # Run IDM inference
            with torch.inference_mode():
                outputs = self._model.get_action(inputs)

            # Extract predicted actions: (B, horizon, action_dim)
            action_pred = outputs["action_pred"]
            if hasattr(action_pred, "cpu"):
                action_pred = action_pred.cpu().numpy()

            # Remove batch dim
            if action_pred.ndim == 3:
                action_pred = action_pred[0]  # (horizon, action_dim)

            # Update sliding window
            self._previous_frame = current_frame

            # Convert to robot action dicts
            return self._actions_to_dicts(action_pred)

        except Exception as e:
            logger.error(f"❌ IDM inference error: {e}")
            self._previous_frame = current_frame
            return self._generate_zero_actions()

    async def _get_vla_actions(self, observation_dict: Dict[str, Any], instruction: str) -> List[Dict[str, Any]]:
        """Get actions using full GR00T-Dreams VLA policy."""
        self._load_vla()

        try:
            # Build observation dict for GR00T-Dreams
            obs = self._build_dreams_observation(observation_dict, instruction)

            # Run VLA inference
            action_dict = self._policy.get_action(obs)

            # Convert GR00T-Dreams output format to robot actions
            return self._convert_dreams_actions(action_dict)

        except Exception as e:
            logger.error(f"❌ GR00T-Dreams VLA inference error: {e}")
            return self._generate_zero_actions()

    def _extract_frame(self, observation_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract a camera frame from the observation dict."""
        for key, value in observation_dict.items():
            if key not in self.robot_state_keys and isinstance(value, np.ndarray):
                if value.ndim == 3 and value.shape[-1] == 3:  # (H, W, C) RGB
                    return value.astype(np.uint8)
                elif value.ndim == 3 and value.shape[0] == 3:  # (C, H, W)
                    return np.transpose(value, (1, 2, 0)).astype(np.uint8)
        return None

    def _build_dreams_observation(self, observation_dict: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Build observation dict for GR00T-Dreams VLA."""
        obs = {}

        # Add camera frames
        for key, value in observation_dict.items():
            if key not in self.robot_state_keys and isinstance(value, np.ndarray):
                if value.ndim >= 2:
                    # Map to video.* key format
                    video_key = f"video.{key}" if not key.startswith("video.") else key
                    obs[video_key] = value

        # Add state
        for key in self.robot_state_keys:
            if key in observation_dict:
                state_key = f"state.{key}" if not key.startswith("state.") else key
                obs[state_key] = np.array([observation_dict[key]], dtype=np.float64)

        # Add language
        obs["annotation.human.task_description"] = instruction

        return obs

    def _convert_dreams_actions(self, action_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert GR00T-Dreams action output to robot action list."""
        # Concatenate all action modalities
        action_parts = []
        for key, value in sorted(action_dict.items()):
            if key.startswith("action."):
                arr = np.array(value)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                action_parts.append(arr)

        if not action_parts:
            return self._generate_zero_actions()

        # Stack: (horizon, total_action_dim)
        concat_actions = np.concatenate(action_parts, axis=-1)
        return self._actions_to_dicts(concat_actions)

    def _actions_to_dicts(self, actions: np.ndarray) -> List[Dict[str, Any]]:
        """Convert action array to list of robot action dicts."""
        robot_actions = []
        for i in range(len(actions)):
            action_dict = {}
            for j, key in enumerate(self.robot_state_keys):
                if j < actions.shape[-1]:
                    action_dict[key] = float(actions[i, j])
                else:
                    action_dict[key] = 0.0
            robot_actions.append(action_dict)
        return robot_actions

    def _generate_zero_actions(self) -> List[Dict[str, Any]]:
        """Generate zero actions as fallback."""
        return [{key: 0.0 for key in self.robot_state_keys} for _ in range(self.action_horizon)]

    def reset(self):
        """Reset policy state (clear previous frame for IDM)."""
        self._previous_frame = None
        logger.info("🔄 DreamGen policy reset")


__all__ = ["DreamgenPolicy"]
