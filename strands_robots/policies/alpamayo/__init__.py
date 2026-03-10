#!/usr/bin/env python3
"""
Alpamayo Policy Provider — NVIDIA's Autonomous Driving Reasoning VLA.

Alpamayo 1 (formerly Alpamayo-R1) is a 10B-param Vision-Language-Action model
for autonomous driving that integrates Chain-of-Causation (CoC) reasoning with
diffusion-based trajectory planning.

Architecture: Cosmos-Reason VLM backbone (8.2B) + Diffusion Action Expert (2.3B)

Input:
    - Multi-camera images (4 cameras: front-wide, front-tele, cross-left, cross-right)
    - 0.4s history window at 10Hz (4 frames per camera)
    - Egomotion history (16 waypoints at 10Hz)
    - Text commands

Output:
    - Chain-of-Causation reasoning traces (text)
    - 6.4s future trajectory (64 waypoints at 10Hz)
    - Position (x, y, z) + rotation matrix in ego vehicle frame

Note: Non-commercial license. For research and development only.

Usage:
    policy = create_policy("alpamayo",
        model_id="nvidia/Alpamayo-R1-10B")

    actions = await policy.get_actions(obs, "turn left at the intersection")
    # Returns: [{"trajectory": [...], "reasoning": "...", "x": ..., "y": ..., "yaw": ...}]

Reference: "Alpamayo-R1: Bridging Reasoning and Action Prediction for
           Generalizable Autonomous Driving in the Long Tail", arXiv:2511.00088
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

import numpy as np

from strands_robots.policies import Policy
from strands_robots.policies._utils import detect_device, extract_pil_image

logger = logging.getLogger(__name__)


class AlpamayoPolicy(Policy):
    """Alpamayo — NVIDIA's Autonomous Driving Reasoning VLA.

    Cosmos-Reason VLM (8.2B) + Diffusion Action Expert (2.3B).
    Outputs 64-waypoint future trajectories with Chain-of-Causation reasoning.

    Supports two modes:
    1. Local mode: Load model directly from HuggingFace (requires 24GB+ VRAM)
    2. Server mode: Connect to running Alpamayo inference server
    """

    # Camera configuration matching Alpamayo's expected input
    CAMERA_NAMES = ["front_wide", "front_tele", "cross_left", "cross_right"]
    HISTORY_HZ = 10
    HISTORY_WINDOW_SEC = 0.4
    TRAJECTORY_HZ = 10
    TRAJECTORY_DURATION_SEC = 6.4
    NUM_TRAJECTORY_WAYPOINTS = 64
    IMAGE_RESOLUTION = (320, 576)  # Model's internal resolution

    def __init__(
        self,
        model_id: str = "nvidia/Alpamayo-R1-10B",
        server_url: Optional[str] = None,
        device: Optional[str] = None,
        generate_reasoning: bool = True,
        max_new_tokens: int = 512,
        waypoint_select: int = 10,
        max_linear_vel: float = 15.0,
        max_angular_vel: float = 0.5,
        **kwargs,
    ):
        """Initialize Alpamayo policy.

        Args:
            model_id: HuggingFace model ID
            server_url: URL of running inference server (bypasses local loading)
            device: Inference device (auto-detected if None)
            generate_reasoning: Whether to generate CoC reasoning traces
            max_new_tokens: Max tokens for reasoning generation
            waypoint_select: Which waypoint index to use for immediate control (0-63)
            max_linear_vel: Maximum linear velocity (m/s) for control output
            max_angular_vel: Maximum angular velocity (rad/s) for control output
        """
        self._model_id = model_id
        self._server_url = server_url
        self._requested_device = device
        self._generate_reasoning = generate_reasoning
        self._max_new_tokens = max_new_tokens
        self._waypoint_select = waypoint_select
        self._max_linear_vel = max_linear_vel
        self._max_angular_vel = max_angular_vel
        self._robot_state_keys: List[str] = []

        # Model components (lazy loaded)
        self._model = None
        self._processor = None
        self._action_decoder = None
        self._loaded = False
        self._device = None
        self._step = 0

        # Egomotion history buffer
        self._ego_history: List[np.ndarray] = []

        mode = "server" if server_url else "local"
        logger.info(f"🏔️ Alpamayo policy: {model_id} (mode={mode})")

    @property
    def provider_name(self) -> str:
        return "alpamayo"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys

    def _ensure_loaded(self):
        if self._loaded:
            return

        if self._server_url:
            # Server mode — verify connectivity
            try:
                import requests

                requests.get(f"{self._server_url}/health", timeout=5)
                logger.info(f"🏔️ Alpamayo server connected: {self._server_url}")
            except Exception as e:
                logger.warning(f"Alpamayo server not reachable: {e}")
            self._loaded = True
            return

        # Local mode — load from HuggingFace
        import torch

        logger.info(f"🏔️ Loading Alpamayo from {self._model_id}...")
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
            logger.info(f"🏔️ Alpamayo loaded: {n_params/1e9:.1f}B in {elapsed:.1f}s on {self._device}")

        except Exception as e:
            raise ImportError(
                f"Failed to load Alpamayo. Requires 24GB+ VRAM.\n"
                f"For server mode: create_policy('alpamayo', server_url='http://...')\n"
                f"Repo: https://github.com/NVlabs/alpamayo\nError: {e}"
            )

        self._loaded = True

    async def get_actions(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Get driving trajectory from Alpamayo.

        Args:
            observation_dict: Robot/vehicle observation containing:
                - Camera images as numpy arrays (keys matching CAMERA_NAMES or any image keys)
                - Optional egomotion: "ego_x", "ego_y", "ego_z", "ego_yaw" or "egomotion" array
                - Optional velocity: "velocity", "speed"
            instruction: Driving command (e.g., "turn left", "follow the road")

        Returns:
            List with one action dict containing:
                - trajectory: Full 64-waypoint trajectory [(x, y, z, yaw), ...]
                - reasoning: Chain-of-Causation text (if generate_reasoning=True)
                - x, y, z: Immediate position target from selected waypoint
                - yaw: Immediate heading target
                - linear_vel: Forward velocity command (m/s)
                - angular_vel: Yaw rate command (rad/s)
        """
        self._ensure_loaded()

        if self._server_url:
            return await self._infer_server(observation_dict, instruction)
        return self._infer_local(observation_dict, instruction)

    async def _infer_server(self, observation_dict: Dict[str, Any], instruction: str) -> List[Dict[str, Any]]:
        """Inference via HTTP server."""
        import base64
        import io

        import requests

        # Extract camera images
        images_b64 = {}
        for cam_name in self.CAMERA_NAMES:
            image = self._find_camera_image(observation_dict, cam_name)
            if image is not None:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                images_b64[cam_name] = base64.b64encode(buf.getvalue()).decode("ascii")

        # Extract egomotion history
        ego = self._extract_egomotion(observation_dict)

        payload = {
            "instruction": instruction,
            "images": images_b64,
            "egomotion_history": ego.tolist() if ego is not None else [],
        }

        resp = requests.post(f"{self._server_url}/predict", json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        trajectory = np.array(result.get("trajectory", []))
        reasoning = result.get("reasoning", "")

        return [self._build_action_dict(trajectory, reasoning)]

    def _infer_local(self, observation_dict: Dict[str, Any], instruction: str) -> List[Dict[str, Any]]:
        """Local inference via HuggingFace model."""
        import torch
        from PIL import Image

        # Collect camera images
        images = []
        for cam_name in self.CAMERA_NAMES:
            img = self._find_camera_image(observation_dict, cam_name)
            if img is None:
                img = Image.new("RGB", (1920, 1080))
            # Resize to model's expected resolution
            img = img.resize(self.IMAGE_RESOLUTION[::-1])
            images.append(img)

        # Build prompt
        prompt = f"<image><image><image><image>\n{instruction}"

        # Process inputs
        if len(images) == 1:
            inputs = self._processor(prompt, images[0], return_tensors="pt")
        else:
            inputs = self._processor(prompt, images, return_tensors="pt")

        inputs = {k: v.to(self._device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        reasoning = ""
        trajectory = np.zeros((self.NUM_TRAJECTORY_WAYPOINTS, 4))

        with torch.no_grad():
            # Try direct trajectory prediction API
            if hasattr(self._model, "predict_trajectory"):
                ego = self._extract_egomotion(observation_dict)
                result = self._model.predict_trajectory(
                    **inputs,
                    egomotion_history=torch.from_numpy(ego).unsqueeze(0).to(self._device) if ego is not None else None,
                )
                trajectory = np.asarray(result.get("trajectory", trajectory))
                reasoning = result.get("reasoning", "")
            else:
                # Fallback: generate text and parse
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
                text = self._processor.decode(outputs[0], skip_special_tokens=True)
                reasoning = text
                trajectory = self._parse_trajectory_from_text(text)

        self._step += 1
        return [self._build_action_dict(trajectory, reasoning)]

    def _find_camera_image(self, observation_dict: Dict[str, Any], cam_name: str):
        """Find a camera image in observation dict by name or fallback."""
        from PIL import Image

        # Direct match
        for key in observation_dict:
            if cam_name in key.lower():
                val = observation_dict[key]
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    return Image.fromarray(val[:, :, :3].astype(np.uint8))

        # Fallback: any image via shared utility
        return extract_pil_image(observation_dict) if cam_name == self.CAMERA_NAMES[0] else None

    def _extract_egomotion(self, observation_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract egomotion history from observation."""
        if "egomotion" in observation_dict:
            return np.asarray(observation_dict["egomotion"], dtype=np.float32)

        # Build from individual keys
        ego_keys = ["ego_x", "ego_y", "ego_z", "ego_yaw"]
        if all(k in observation_dict for k in ego_keys):
            ego = np.array(
                [float(observation_dict[k]) for k in ego_keys],
                dtype=np.float32,
            )
            self._ego_history.append(ego)
            # Keep last 16 waypoints (0.4s at 10Hz = 4 frames, but keep more for safety)
            if len(self._ego_history) > 16:
                self._ego_history = self._ego_history[-16:]
            return np.stack(self._ego_history)

        return None

    def _parse_trajectory_from_text(self, text: str) -> np.ndarray:
        """Parse trajectory waypoints from generated text."""
        import re

        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        values = [float(n) for n in numbers]

        # Group into (x, y, z, yaw) tuples
        trajectory = []
        for i in range(0, len(values) - 3, 4):
            trajectory.append(values[i : i + 4])

        # Pad to expected length
        while len(trajectory) < self.NUM_TRAJECTORY_WAYPOINTS:
            if trajectory:
                trajectory.append(trajectory[-1])
            else:
                trajectory.append([0.0, 0.0, 0.0, 0.0])

        return np.array(trajectory[: self.NUM_TRAJECTORY_WAYPOINTS], dtype=np.float32)

    def _build_action_dict(self, trajectory: np.ndarray, reasoning: str = "") -> Dict[str, Any]:
        """Build action dict from trajectory + reasoning."""
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 4)

        # Select immediate control waypoint
        idx = min(self._waypoint_select, len(trajectory) - 1)
        waypoint = trajectory[idx] if len(trajectory) > 0 else np.zeros(4)

        x, y = float(waypoint[0]), float(waypoint[1])
        z = float(waypoint[2]) if len(waypoint) > 2 else 0.0
        yaw = float(waypoint[3]) if len(waypoint) > 3 else 0.0

        # Compute velocity commands from trajectory
        if len(trajectory) >= 2:
            dt = 1.0 / self.TRAJECTORY_HZ
            dx = float(trajectory[1][0] - trajectory[0][0])
            dy = float(trajectory[1][1] - trajectory[0][1])
            linear_vel = math.sqrt(dx**2 + dy**2) / dt
            angular_vel = yaw / dt if abs(yaw) > 1e-6 else 0.0
        else:
            linear_vel = 0.0
            angular_vel = 0.0

        linear_vel = np.clip(linear_vel, 0, self._max_linear_vel)
        angular_vel = np.clip(angular_vel, -self._max_angular_vel, self._max_angular_vel)

        action_dict = {
            "x": x,
            "y": y,
            "z": z,
            "yaw": yaw,
            "linear_vel": float(linear_vel),
            "angular_vel": float(angular_vel),
            "trajectory": trajectory.tolist(),
        }

        if reasoning:
            action_dict["reasoning"] = reasoning

        # Map to robot state keys if configured
        if self._robot_state_keys:
            mapped = {}
            for key in self._robot_state_keys:
                if key in action_dict:
                    mapped[key] = action_dict[key]
            if mapped:
                action_dict.update(mapped)

        return action_dict

    def get_reasoning(self, observation_dict: Dict[str, Any], situation: str) -> str:
        """Get Chain-of-Causation reasoning about a driving situation.

        Uses Alpamayo as a VLM to reason about the driving scene, identifying
        causal factors, risks, and decision rationale.

        Args:
            observation_dict: Camera images + vehicle state
            situation: Description of the driving situation to reason about

        Returns:
            Chain-of-Causation reasoning text
        """
        self._ensure_loaded()
        if self._model is None:
            return "Model not loaded"

        import torch
        from PIL import Image

        image = self._find_camera_image(observation_dict, "front_wide")
        if image is None:
            image = Image.new("RGB", (576, 320))

        prompt = f"<image>\nAnalyze this driving scene: {situation}\nProvide Chain-of-Causation reasoning."

        inputs = self._processor(prompt, image, return_tensors="pt").to(self._device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        return self._processor.decode(outputs[0], skip_special_tokens=True)

    def reset(self):
        """Reset internal state."""
        self._step = 0
        self._ego_history = []
        logger.info("🏔️ Alpamayo policy reset")


__all__ = ["AlpamayoPolicy"]
