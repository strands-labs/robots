#!/usr/bin/env python3
"""
RoboBrain Policy Provider — BAAI's Embodied Brain Model.

RoboBrain 2.0 is an open-source embodied brain model from Beijing Academy of
Artificial Intelligence (BAAI) that unifies perception, reasoning, and planning
for complex embodied tasks.

Available in 3 variants:
    - BAAI/RoboBrain2.0-3B: Ultra-lightweight, edge-friendly
    - BAAI/RoboBrain2.0-7B: Balanced performance (default)
    - BAAI/RoboBrain2.0-32B: Full-scale, SOTA on spatial/temporal benchmarks

Built on Qwen2.5-VL backbone. Supports:
    - Spatial understanding (affordance prediction, spatial referring, trajectory forecasting)
    - Temporal decision-making (closed-loop interaction, multi-agent planning, scene memory)
    - Multi-image, long video, and high-resolution visual inputs
    - Structured scene graph reasoning

Outperforms Gemini 2.5 Pro, o4-mini, and Claude Sonnet 4 on spatial benchmarks.

Usage:
    policy = create_policy("robobrain", model_id="BAAI/RoboBrain2.0-7B")
    actions = await policy.get_actions(obs, "pick up the red cup from the shelf")

Reference: BAAI RoboBrain Team, "RoboBrain 2.0 Technical Report", arXiv:2507.02029, 2025
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy
from strands_robots.policies._utils import detect_device, extract_pil_image, parse_numbers_from_text

logger = logging.getLogger(__name__)


class RobobrainPolicy(Policy):
    """RoboBrain 2.0 — BAAI's Embodied Brain Model.

    Qwen2.5-VL backbone with MLP projector for multi-modal embodied reasoning.
    Outputs structured plans with spatial coordinates and action sequences.

    Features:
    - Spatial understanding: affordance, pointing, bbox prediction
    - Temporal reasoning: trajectory forecasting, closed-loop interaction
    - Scene graph construction and reasoning
    - Multi-image and video input support
    """

    def __init__(
        self,
        model_id: str = "BAAI/RoboBrain2.0-7B",
        device: Optional[str] = None,
        action_dim: int = 7,
        max_new_tokens: int = 512,
        enable_scene_memory: bool = False,
        **kwargs,
    ):
        """Initialize RoboBrain policy.

        Args:
            model_id: HuggingFace model ID (3B, 7B, or 32B variant)
            device: Inference device (auto-detected if None)
            action_dim: Action dimensionality for manipulation tasks
            max_new_tokens: Max generation length
            enable_scene_memory: Whether to maintain a scene graph memory
        """
        self._model_id = model_id
        self._requested_device = device
        self._action_dim = action_dim
        self._max_new_tokens = max_new_tokens
        self._enable_scene_memory = enable_scene_memory
        self._robot_state_keys: List[str] = []

        self._model = None
        self._processor = None
        self._loaded = False
        self._device = None
        self._step = 0

        # Scene memory (optional)
        self._scene_graph: Dict[str, Any] = {}
        self._scene_history: List[str] = []

        logger.info(f"🧠 RoboBrain policy: {model_id}")

    @property
    def provider_name(self) -> str:
        return "robobrain"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys
        self._action_dim = len(robot_state_keys) or self._action_dim

    def _ensure_loaded(self):
        if self._loaded:
            return

        import torch

        logger.info(f"🧠 Loading RoboBrain from {self._model_id}...")
        start = time.time()

        self._device = detect_device(self._requested_device)

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()

            elapsed = time.time() - start
            n_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"🧠 RoboBrain loaded: {n_params/1e9:.1f}B in {elapsed:.1f}s on {self._device}")

        except Exception as e:
            raise ImportError(
                f"Failed to load RoboBrain. Requires Qwen2.5-VL architecture support.\n"
                f"Install: pip install transformers>=4.45.0\n"
                f"Repo: https://github.com/FlagOpen/RoboBrain2.0\nError: {e}"
            )

        self._loaded = True

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from RoboBrain.

        RoboBrain can output:
        - Manipulation actions (joint positions or EEF deltas)
        - Spatial predictions (point coordinates, bounding boxes)
        - Action plans (multi-step structured plans)

        The output format depends on the instruction.
        """
        self._ensure_loaded()
        import torch

        image = extract_pil_image(observation_dict)

        # Build RoboBrain-style prompt
        # RoboBrain uses Qwen2.5-VL chat format
        prompt = self._build_prompt(instruction, observation_dict)

        inputs = self._processor(prompt, image, return_tensors="pt").to(self._device, dtype=torch.bfloat16)

        with torch.no_grad():
            # Try direct action prediction API
            if hasattr(self._model, "predict_action"):
                action = self._model.predict_action(**inputs)
                action_np = np.asarray(action).flatten()
            else:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
                text = self._processor.decode(outputs[0], skip_special_tokens=True)
                action_np = parse_numbers_from_text(text, action_dim=self._action_dim, take_last=True)

                # Update scene memory if enabled
                if self._enable_scene_memory:
                    self._update_scene_memory(text)

        action_dict = self._array_to_dict(action_np)
        self._step += 1
        return [action_dict]

    def _build_prompt(self, instruction: str, observation_dict: Dict[str, Any]) -> str:
        """Build RoboBrain-style prompt with optional scene context."""
        parts = ["<image>"]

        # Add scene memory context if available
        if self._enable_scene_memory and self._scene_history:
            recent = self._scene_history[-3:]
            parts.append("Scene context: " + "; ".join(recent))

        # Add proprioceptive state if available
        if self._robot_state_keys:
            state = [
                f"{k}={observation_dict.get(k, 0.0):.3f}" for k in self._robot_state_keys[:6] if k in observation_dict
            ]
            if state:
                parts.append(f"Robot state: {', '.join(state)}")

        parts.append(f"Task: {instruction}")
        parts.append("Predict the robot action as numerical values.")

        return "\n".join(parts)

    def _array_to_dict(self, arr: np.ndarray) -> Dict[str, Any]:
        eef_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        keys = self._robot_state_keys if self._robot_state_keys else eef_keys
        return {keys[i]: float(arr[i]) for i in range(min(len(keys), len(arr)))}

    def _update_scene_memory(self, text: str):
        """Update internal scene memory from model output."""
        # Extract spatial information from the response
        self._scene_history.append(text[:200])
        if len(self._scene_history) > 10:
            self._scene_history = self._scene_history[-10:]

    # ==== Bonus capabilities ====

    def spatial_refer(self, observation_dict: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Predict spatial location from referring expression.

        Returns point coordinates or bounding box.
        """
        self._ensure_loaded()
        import torch

        image = extract_pil_image(observation_dict)
        prompt = f"<image>\nPoint to: {query}\nOutput the (x, y) coordinates."

        inputs = self._processor(prompt, image, return_tensors="pt").to(self._device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = self._processor.decode(outputs[0], skip_special_tokens=True)

        coords = parse_numbers_from_text(text, action_dim=4, take_last=False)
        return {"text": text, "coordinates": coords.tolist()}

    def predict_trajectory(self, observation_dict: Dict[str, Any], instruction: str) -> List[Dict[str, float]]:
        """Predict future trajectory waypoints.

        Uses RoboBrain's temporal reasoning to forecast object/robot motion.
        """
        self._ensure_loaded()
        import torch

        image = extract_pil_image(observation_dict)
        prompt = f"<image>\nPredict the trajectory for: {instruction}\nOutput (x, y) waypoints."

        inputs = self._processor(prompt, image, return_tensors="pt").to(self._device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = self._processor.decode(outputs[0], skip_special_tokens=True)

        import re

        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        waypoints = []
        for i in range(0, len(numbers) - 1, 2):
            waypoints.append({"x": float(numbers[i]), "y": float(numbers[i + 1])})
        return waypoints

    def reason_about_scene(self, observation_dict: Dict[str, Any], question: str) -> str:
        """Use RoboBrain as a VLM for embodied scene reasoning."""
        self._ensure_loaded()
        import torch

        image = extract_pil_image(observation_dict)
        prompt = f"<image>\n{question}"

        inputs = self._processor(prompt, image, return_tensors="pt").to(self._device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return self._processor.decode(outputs[0], skip_special_tokens=True)


__all__ = ["RobobrainPolicy"]
