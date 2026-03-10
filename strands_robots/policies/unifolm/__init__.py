#!/usr/bin/env python3
"""
UnifoLM Policy Provider — Unitree's Unified Foundation Language Model for robotics.

Unitree's proprietary VLA foundation model for their robot ecosystem
(G1 humanoid, H1 humanoid, Go2 quadruped, etc.).

Variants:
    - unitreerobotics/UnifoLM-WMA-0-Base: World Model Agent (base)
    - unitreerobotics/UnifoLM-WMA-0-Dual: Dual-arm variant
    - unitreerobotics/UnifoLM-VLA-Base: Vision-Language-Action base

Note: Limited public documentation as of Feb 2026. This provider
supports both HuggingFace loading and Unitree's native API.

Usage:
    policy = create_policy("unifolm", model_id="unitreerobotics/UnifoLM-VLA-Base")
    actions = await policy.get_actions(obs, "walk forward")
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy
from strands_robots.policies._utils import detect_device, extract_pil_image, parse_numbers_from_text

logger = logging.getLogger(__name__)


class UnifolmPolicy(Policy):
    """UnifoLM — Unitree's Foundation Language Model for robotics.

    Supports loading from HuggingFace or connecting to Unitree's
    native inference server.
    """

    def __init__(
        self,
        model_id: str = "unitreerobotics/UnifoLM-VLA-Base",
        server_url: Optional[str] = None,
        device: Optional[str] = None,
        action_dim: int = 12,
        **kwargs,
    ):
        self._model_id = model_id
        self._server_url = server_url
        self._requested_device = device
        self._action_dim = action_dim
        self._robot_state_keys: List[str] = []

        self._model = None
        self._processor = None
        self._loaded = False
        self._step = 0

        logger.info(f"🦾 UnifoLM policy: {model_id}")

    @property
    def provider_name(self) -> str:
        return "unifolm"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys
        self._action_dim = len(robot_state_keys) or self._action_dim

    def _ensure_loaded(self):
        if self._loaded:
            return

        if self._server_url:
            logger.info(f"🦾 UnifoLM server mode: {self._server_url}")
            self._loaded = True
            return

        import torch

        logger.info(f"🦾 Loading UnifoLM from {self._model_id}...")
        start = time.time()

        self._device = detect_device(self._requested_device)

        try:
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self._model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()

            elapsed = time.time() - start
            n_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"🦾 UnifoLM loaded: {n_params/1e9:.1f}B in {elapsed:.1f}s")

        except Exception as e:
            logger.warning(
                f"UnifoLM loading failed (model may require special access): {e}\n"
                f"Try: policy = create_policy('unifolm', server_url='http://...')"
            )

        self._loaded = True

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        self._ensure_loaded()

        if self._server_url:
            return await self._infer_server(observation_dict, instruction)

        if self._model is None:
            logger.warning("UnifoLM model not loaded — returning zeros")
            return [{k: 0.0 for k in self._robot_state_keys}]

        return self._infer_local(observation_dict, instruction)

    async def _infer_server(self, observation_dict: Dict[str, Any], instruction: str) -> List[Dict[str, Any]]:
        """Inference via Unitree's server API."""
        import base64
        import io

        import requests

        image = extract_pil_image(observation_dict)
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        state = [float(observation_dict.get(k, 0.0)) for k in self._robot_state_keys]

        payload = {
            "instruction": instruction,
            "image": base64.b64encode(buf.getvalue()).decode("ascii"),
            "state": state,
        }

        resp = requests.post(f"{self._server_url}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        actions = np.array(result.get("actions", result.get("action", [0.0] * self._action_dim)))
        return [self._array_to_dict(actions.flatten())]

    def _infer_local(self, observation_dict: Dict[str, Any], instruction: str) -> List[Dict[str, Any]]:
        """Local inference via HuggingFace model."""
        import torch

        image = extract_pil_image(observation_dict)
        state = np.array(
            [float(observation_dict.get(k, 0.0)) for k in self._robot_state_keys],
            dtype=np.float32,
        )

        # Try model-specific API first
        if hasattr(self._model, "predict_action"):
            with torch.no_grad():
                action = self._model.predict_action(image=image, instruction=instruction, state=state)
            return [self._array_to_dict(np.asarray(action).flatten())]

        # Fallback: generic generate
        if self._processor:
            inputs = self._processor(instruction, image, return_tensors="pt").to(self._device, dtype=torch.bfloat16)
            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_new_tokens=256)
            text = self._processor.decode(outputs[0], skip_special_tokens=True)
            action = parse_numbers_from_text(text, action_dim=self._action_dim, take_last=False)
            return [self._array_to_dict(action)]

        return [{k: 0.0 for k in self._robot_state_keys}]

    def _array_to_dict(self, arr: np.ndarray) -> Dict[str, Any]:
        keys = self._robot_state_keys or [f"joint_{i}" for i in range(len(arr))]
        return {keys[i]: float(arr[i]) for i in range(min(len(keys), len(arr)))}


__all__ = ["UnifolmPolicy"]
