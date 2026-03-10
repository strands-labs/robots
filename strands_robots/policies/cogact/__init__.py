#!/usr/bin/env python3
"""
CogACT Policy Provider — Cognitive Action model with diffusion-based decoding.

CogACT uses a VLM backbone (CogVLM2) with a diffusion action decoder for
manipulation. Key innovation: instead of tokenizing actions as text,
it uses a diffusion-based action head that generates continuous action
trajectories via iterative denoising.

Architecture: CogVLM2 vision-language backbone → Diffusion Action Decoder → Actions
Pre-trained on the Open X-Embodiment (OXE) dataset.

Checkpoints:
    - CogACT/CogACT-Base (base, 17⭐, 6.3K downloads)

Usage:
    policy = create_policy("cogact", model_id="CogACT/CogACT-Base")
    actions = await policy.get_actions(obs, "pick up the red block")

Reference: Li et al., "CogACT: A Foundational Vision-Language-Action Model
           for Synergizing Cognition and Action in Robotic Manipulation",
           arXiv:2411.19650, 2024
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy
from strands_robots.policies._utils import detect_device, extract_pil_image, parse_numbers_from_text

logger = logging.getLogger(__name__)


class CogactPolicy(Policy):
    """CogACT — Cognitive Action model with diffusion-based action decoding.

    Uses CogVLM2 backbone for visual understanding and a diffusion head
    to predict continuous action trajectories.
    """

    def __init__(
        self,
        model_id: str = "CogACT/CogACT-Base",
        device: Optional[str] = None,
        action_dim: int = 7,
        action_horizon: int = 4,
        num_diffusion_steps: int = 10,
        unnorm_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CogACT policy.

        Args:
            model_id: HuggingFace model ID
            device: Inference device
            action_dim: Action dimensionality (default: 7 for EEF)
            action_horizon: Number of future actions to predict per step
            num_diffusion_steps: Number of denoising steps for action generation
            unnorm_key: Action unnormalization key (embodiment-specific, e.g., "bridge_orig")
        """
        self._model_id = model_id
        self._requested_device = device
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._num_diffusion_steps = num_diffusion_steps
        self._unnorm_key = unnorm_key
        self._robot_state_keys: List[str] = []

        self._model = None
        self._processor = None
        self._loaded = False
        self._device = None
        self._step = 0

        logger.info(f"⚡ CogACT policy: {model_id}")

    @property
    def provider_name(self) -> str:
        return "cogact"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys
        self._action_dim = max(len(robot_state_keys), self._action_dim)

    def _ensure_loaded(self):
        if self._loaded:
            return

        import torch

        logger.info(f"⚡ Loading CogACT from {self._model_id}...")
        start = time.time()

        self._device = detect_device(self._requested_device)

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()

            elapsed = time.time() - start
            n_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"⚡ CogACT loaded: {n_params/1e9:.1f}B in {elapsed:.1f}s on {self._device}")

        except Exception as e:
            raise ImportError(
                f"Failed to load CogACT from {self._model_id}.\n"
                f"Install: pip install transformers>=4.42.0\n"
                f"Repo: https://github.com/microsoft/CogACT\nError: {e}"
            )

        self._loaded = True

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from CogACT.

        CogACT outputs action trajectories via diffusion denoising.
        Returns multiple future actions (action_horizon) per call.
        """
        self._ensure_loaded()
        import torch

        image = extract_pil_image(observation_dict)

        # CogACT prompt format
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        inputs = self._processor(prompt, image, return_tensors="pt").to(self._device, dtype=torch.bfloat16)

        with torch.no_grad():
            # Try CogACT's direct action prediction
            if hasattr(self._model, "predict_action"):
                predict_kwargs = {}
                unnorm_key = kwargs.get("unnorm_key", self._unnorm_key)
                if unnorm_key:
                    predict_kwargs["unnorm_key"] = unnorm_key
                action = self._model.predict_action(**inputs, **predict_kwargs)
                action_np = np.asarray(action)
            elif hasattr(self._model, "generate_actions"):
                action = self._model.generate_actions(
                    **inputs,
                    num_diffusion_steps=self._num_diffusion_steps,
                )
                action_np = np.asarray(action)
            else:
                # Fallback: text generation
                outputs = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
                text = self._processor.decode(outputs[0], skip_special_tokens=True)
                action_np = parse_numbers_from_text(text, action_dim=self._action_dim, take_last=True).reshape(1, -1)

        # Handle multi-step action output
        if action_np.ndim == 1:
            action_np = action_np.reshape(1, -1)

        eef_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        results = []
        for i in range(min(self._action_horizon, len(action_np))):
            action_dict = {}
            keys = self._robot_state_keys if len(self._robot_state_keys) >= len(action_np[i]) else eef_keys
            for j in range(min(len(keys), len(action_np[i]))):
                action_dict[keys[j]] = float(action_np[i][j])
            results.append(action_dict)

        self._step += 1
        return results if results else [{k: 0.0 for k in (self._robot_state_keys or eef_keys)}]


__all__ = ["CogactPolicy"]
