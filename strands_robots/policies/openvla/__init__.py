#!/usr/bin/env python3
"""
OpenVLA Policy Provider — Stanford's Open Vision-Language-Action Model.

The most downloaded robotics model on HuggingFace (1M+ downloads, 176⭐).
7B params, trained on 970K episodes from Open X-Embodiment.
Outputs 7-DoF end-effector deltas (x, y, z, roll, pitch, yaw, gripper).

Supports zero-shot control for embodiments in the OXE pretraining mix
(e.g., WidowX/BridgeV2, RT-2 robots) and efficient fine-tuning for new setups.

Also supports OpenVLA-OFT (Optimized Fine-Tuning) variants for LIBERO benchmarks.

Usage:
    policy = create_policy("openvla", model_id="openvla/openvla-7b")
    actions = await policy.get_actions(obs, "pick up the red block")

Checkpoints:
    - openvla/openvla-7b (base, 1M+ downloads)
    - openvla/openvla-7b-finetuned-libero-spatial
    - openvla/openvla-7b-finetuned-libero-10
    - moojink/openvla-7b-oft-finetuned-libero-spatial (OFT variant)
    - Embodied-CoT/ecot-openvla-7b-bridge (chain-of-thought)

Reference: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", 2024
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy
from strands_robots.policies._utils import detect_device, extract_pil_image

logger = logging.getLogger(__name__)


class OpenvlaPolicy(Policy):
    """OpenVLA — HuggingFace-native VLA for manipulation.

    Architecture: DINOv2 + SigLIP vision → Llama-2 LLM → 7-DoF action tokens
    Input: language instruction + single camera image
    Output: (x, y, z, roll, pitch, yaw, gripper) end-effector deltas
    """

    def __init__(
        self,
        model_id: str = "openvla/openvla-7b",
        unnorm_key: Optional[str] = None,
        device: Optional[str] = None,
        use_flash_attention: bool = True,
        do_sample: bool = False,
        **kwargs,
    ):
        self._model_id = model_id
        self._unnorm_key = unnorm_key
        self._requested_device = device
        self._use_flash_attn = use_flash_attention
        self._do_sample = do_sample
        self._robot_state_keys: List[str] = []

        self._vla = None
        self._processor = None
        self._loaded = False
        self._step = 0

        logger.info(f"🔮 OpenVLA policy: {model_id}")

    @property
    def provider_name(self) -> str:
        return "openvla"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys

    def _ensure_loaded(self):
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info(f"🔮 Loading OpenVLA from {self._model_id}...")
        start = time.time()

        self._device = detect_device(self._requested_device)

        attn_impl = "flash_attention_2" if self._use_flash_attn and self._device.startswith("cuda") else None
        kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True, "trust_remote_code": True}
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        self._vla = AutoModelForVision2Seq.from_pretrained(self._model_id, **kwargs).to(self._device)

        elapsed = time.time() - start
        n_params = sum(p.numel() for p in self._vla.parameters())
        logger.info(f"🔮 OpenVLA loaded: {n_params/1e9:.1f}B params in {elapsed:.1f}s on {self._device}")
        self._loaded = True

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        import torch

        # Extract camera image
        image = extract_pil_image(observation_dict)

        # Format prompt per OpenVLA convention
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # Predict
        inputs = self._processor(prompt, image).to(self._device, dtype=torch.bfloat16)
        predict_kwargs = {"do_sample": self._do_sample}
        if self._unnorm_key:
            predict_kwargs["unnorm_key"] = self._unnorm_key
        unnorm_key = kwargs.get("unnorm_key", self._unnorm_key)
        if unnorm_key:
            predict_kwargs["unnorm_key"] = unnorm_key

        action = self._vla.predict_action(**inputs, **predict_kwargs)

        # action is (7,) numpy: x, y, z, roll, pitch, yaw, gripper
        action_np = np.asarray(action).flatten()

        # Map to robot state keys or default EEF naming
        action_dict = {}
        eef_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        keys = self._robot_state_keys if len(self._robot_state_keys) >= 7 else eef_keys
        for i, key in enumerate(keys[: len(action_np)]):
            action_dict[key] = float(action_np[i])

        self._step += 1
        return [action_dict]


__all__ = ["OpenvlaPolicy"]
