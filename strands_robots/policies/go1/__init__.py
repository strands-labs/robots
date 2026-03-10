#!/usr/bin/env python3
"""
GO-1 Policy Provider — AgiBot World's Foundation VLA Model.

GO-1 is a Vision-Language-Action model pretrained on the AgiBot World dataset
(1M+ trajectories, 100+ robots). It combines InternVL2.5 vision encoder,
InternLM2 language model, and a DDPM-based ActionExpert with latent planning.

Architecture:
    InternVL2.5 Vision (InternViT-6B-448px) → MLP connector →
    InternLM2 LLM (2B) → KV-cache cross-attention →
    Latent Planner (optional) → ActionExpert (DDPM diffusion) →
    16-dim action chunks @ 30Hz

Key specs:
    - action_dim=16 (7 left arm joints + 1 left gripper + 7 right arm joints + 1 right gripper)
    - action_chunk_size=30 (1 second at 30Hz)
    - state_dim=16 (proprioceptive input, same layout as action)
    - 3 cameras: head, left_hand, right_hand (448×448 InternVL format)
    - ~5.6GB model weights (~2.6B total params)
    - Supports normalization via dataset_stats.json

Variants:
    - GO-1: Full model with latent planner (DDPM + discrete latent codes)
    - GO-1-Air: Lightweight variant without latent planner

Two inference modes:
    1. Server mode (FastAPI): Connect to a running GO-1 inference server
    2. Local mode: Load model directly from HuggingFace (requires GPU, ~7GB VRAM)

Usage:
    # Server mode (recommended for deployment)
    policy = create_policy("go1", server_url="http://localhost:9000")

    # Local mode (requires GPU + go1/agibot-world packages)
    policy = create_policy("go1", model_id="agibot-world/GO-1")

    # GO-1 Air variant (no latent planner, lighter)
    policy = create_policy("go1", model_id="agibot-world/GO-1-Air")

Server API (FastAPI at /act endpoint):
    POST /act
    Request body (JSON):
    {
        "instruction": "pick up the red cup",
        "top": <np.ndarray HxWx3>,        # head camera
        "right": <np.ndarray HxWx3>,       # right hand camera
        "left": <np.ndarray HxWx3>,        # left hand camera
        "state": <np.ndarray (16,)>,       # proprioceptive state
        "ctrl_freqs": <np.ndarray (1,)>    # control frequency (e.g., 30.0)
    }
    Response: list of 30 actions, each (16,) array

Reference:
    Bu et al., "AgiBot World Colosseo: A Large-Scale Manipulation Platform
    for Scalable and Intelligent Embodied Systems", arXiv:2503.06669, 2025
    Project: https://agibot-world.com/
    Repo: https://github.com/OpenDriveLab/Agibot-World
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Default dataset normalization stats from agibot-world/GO-1
# These are baked in from dataset_stats.json for convenience.
# ─────────────────────────────────────────────────────────────────────

_DEFAULT_STATS = {
    "state": {
        "mean": np.array(
            [
                -1.0137895,
                0.6469389,
                0.61943513,
                -0.9803505,
                0.5940099,
                0.9803505,
                0.00141076,
                0.0,
                1.0361551,
                -0.6618055,
                -0.79369825,
                0.8947412,
                -0.5716255,
                -0.9803505,
                -0.03847568,
                0.0,
            ],
            dtype=np.float32,
        ),
        "std": np.array(
            [
                0.54978794,
                0.3524604,
                0.53088915,
                0.36732885,
                0.47180042,
                0.4573713,
                0.6099905,
                1.0,
                0.55983984,
                0.3511852,
                0.54829943,
                0.39228913,
                0.5156797,
                0.44490147,
                0.629012,
                1.0,
            ],
            dtype=np.float32,
        ),
    },
    "action": {
        "mean": np.array(
            [
                -1.0137895,
                0.6469389,
                0.61943513,
                -0.9803505,
                0.5940099,
                0.9803505,
                0.00141076,
                0.0,
                1.0361551,
                -0.6618055,
                -0.79369825,
                0.8947412,
                -0.5716255,
                -0.9803505,
                -0.03847568,
                0.0,
            ],
            dtype=np.float32,
        ),
        "std": np.array(
            [
                0.54978794,
                0.3524604,
                0.53088915,
                0.36732885,
                0.47180042,
                0.4573713,
                0.6099905,
                1.0,
                0.55983984,
                0.3511852,
                0.54829943,
                0.39228913,
                0.5156797,
                0.44490147,
                0.629012,
                1.0,
            ],
            dtype=np.float32,
        ),
    },
}

# Camera key mapping: GO-1 server convention → observation dict fallbacks
_CAMERA_ALIASES = {
    "cam_head_color": ["head", "top", "front", "ego_view", "webcam", "main"],
    "cam_hand_right_color": ["right_hand", "right", "right_wrist", "wrist_right"],
    "cam_hand_left_color": ["left_hand", "left", "left_wrist", "wrist_left"],
}

# AgiBot Genie1 joint layout: 7 left arm + 1 left gripper + 7 right arm + 1 right gripper
_DEFAULT_JOINT_KEYS = [
    "left_arm_joint_0",
    "left_arm_joint_1",
    "left_arm_joint_2",
    "left_arm_joint_3",
    "left_arm_joint_4",
    "left_arm_joint_5",
    "left_arm_joint_6",
    "left_gripper",
    "right_arm_joint_0",
    "right_arm_joint_1",
    "right_arm_joint_2",
    "right_arm_joint_3",
    "right_arm_joint_4",
    "right_arm_joint_5",
    "right_arm_joint_6",
    "right_gripper",
]


class Go1Policy(Policy):
    """GO-1 — AgiBot World Foundation VLA Model.

    Supports two inference modes:
    1. Server mode: connects to a FastAPI GO-1 inference server (POST /act)
    2. Local mode: loads GO-1 model directly from HuggingFace (requires GPU)

    The model predicts action_chunk_size (default 30) action steps at once,
    each with action_dim (default 16) dimensions covering bimanual arm joints
    and grippers.

    Args:
        model_id: HuggingFace model ID or local path (default: "agibot-world/GO-1")
        server_url: FastAPI server URL for server mode (e.g., "http://localhost:9000")
        device: Inference device for local mode (auto-detected if None)
        action_dim: Action dimensionality (default: 16)
        action_chunk_size: Number of action steps per prediction (default: 30)
        ctrl_freq: Control frequency in Hz (default: 30.0)
        normalize: Whether to normalize state/unnormalize actions (default: False)
        data_stats_path: Path to dataset_stats.json for normalization
        camera_keys: Camera key names to use (default: head + both hands)
        num_inference_timesteps: DDPM denoising steps for local inference (default: 5)
    """

    def __init__(
        self,
        model_id: str = "agibot-world/GO-1",
        pretrained_name_or_path: Optional[str] = None,
        server_url: Optional[str] = None,
        device: Optional[str] = None,
        action_dim: int = 16,
        action_chunk_size: int = 30,
        ctrl_freq: float = 30.0,
        normalize: bool = False,
        data_stats_path: Optional[str] = None,
        camera_keys: Optional[List[str]] = None,
        num_inference_timesteps: int = 5,
        **kwargs,
    ):
        # Accept pretrained_name_or_path as alias for model_id (policy_resolver compat)
        if pretrained_name_or_path is not None:
            model_id = pretrained_name_or_path
        self._model_id = model_id
        self._server_url = server_url
        self._device = device
        self._action_dim = action_dim
        self._action_chunk_size = action_chunk_size
        self._ctrl_freq = ctrl_freq
        self._normalize = normalize
        self._data_stats_path = data_stats_path
        self._camera_keys = camera_keys or ["cam_head_color", "cam_hand_right_color", "cam_hand_left_color"]
        self._num_inference_timesteps = num_inference_timesteps
        self._robot_state_keys: List[str] = []

        # Model state (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._img_transform = None
        self._config = None
        self._data_stats = None
        self._loaded = False
        self._step = 0

        mode = "server" if server_url else "local"
        logger.info(
            f"🤖 GO-1 policy: {model_id} (mode={mode}, action_dim={action_dim}, "
            f"chunk_size={action_chunk_size}, ctrl_freq={ctrl_freq}Hz)"
        )

    @property
    def provider_name(self) -> str:
        return "go1"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys

    # ─────────────────────────────────────────────────────────────────
    # Lazy loading
    # ─────────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Lazy-load model or verify server connectivity."""
        if self._loaded:
            return

        if self._server_url:
            self._ensure_server_loaded()
        else:
            self._ensure_local_loaded()

        self._loaded = True

    def _ensure_server_loaded(self):
        """Verify server connectivity (non-blocking on failure)."""
        try:
            import requests

            resp = requests.get(f"{self._server_url}/docs", timeout=5)
            logger.info(f"🤖 GO-1 server connected: {self._server_url} (status={resp.status_code})")
        except Exception as e:
            logger.warning(f"GO-1 server not reachable at {self._server_url}: {e}")

    def _ensure_local_loaded(self):
        """Load GO-1 model locally from HuggingFace."""
        import torch

        logger.info(f"🤖 Loading GO-1 from {self._model_id}...")
        start = time.time()

        device = self._device
        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device

        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            # Load config to get model parameters
            self._config = AutoConfig.from_pretrained(self._model_id, trust_remote_code=True)

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id, add_eos_token=False, trust_remote_code=True, use_fast=False
            )

            # Load model
            self._model = AutoModel.from_pretrained(
                self._model_id,
                config=self._config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device)
            self._model.eval()

            elapsed = time.time() - start
            n_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"🤖 GO-1 loaded: {n_params / 1e9:.1f}B params in {elapsed:.1f}s on {device}")

        except Exception as e:
            raise ImportError(
                f"Failed to load GO-1 model from '{self._model_id}'.\n"
                f"For server mode: policy = create_policy('go1', server_url='http://...')\n"
                f"For local mode: pip install transformers>=4.46.0 torch\n"
                f"Repo: https://github.com/OpenDriveLab/Agibot-World\nError: {e}"
            )

        # Load normalization stats if needed
        self._load_data_stats()

    def _load_data_stats(self):
        """Load normalization statistics from file or use defaults."""
        if self._data_stats_path:
            import json

            import torch

            with open(self._data_stats_path, "r") as f:
                stats_json = json.load(f)
            self._data_stats = {}
            for name in ["state", "action"]:
                self._data_stats[name] = {
                    "mean": torch.from_numpy(np.array(stats_json[name]["mean"], dtype=np.float32)),
                    "std": torch.from_numpy(np.array(stats_json[name]["std"], dtype=np.float32)),
                }
            logger.info(f"🤖 Loaded normalization stats from {self._data_stats_path}")
        elif self._normalize:
            import torch

            self._data_stats = {
                name: {
                    "mean": torch.from_numpy(_DEFAULT_STATS[name]["mean"]),
                    "std": torch.from_numpy(_DEFAULT_STATS[name]["std"]),
                }
                for name in ["state", "action"]
            }
            logger.info("🤖 Using default AgiBot World normalization stats")

    # ─────────────────────────────────────────────────────────────────
    # Main inference entry point
    # ─────────────────────────────────────────────────────────────────

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from GO-1 model.

        Args:
            observation_dict: Must contain camera images and optionally robot state.
                Camera keys searched: head/top/front, right_hand/right, left_hand/left
                State: either individual joint keys or "observation.state" array
            instruction: Natural language task description
            **kwargs: Optional overrides (ctrl_freq, action_chunk_size)

        Returns:
            List of action dicts, one per step in the action chunk.
            Each dict maps robot_state_keys to float values.
        """
        self._ensure_loaded()

        if self._server_url:
            return await self._infer_server(observation_dict, instruction, **kwargs)
        return self._infer_local(observation_dict, instruction, **kwargs)

    # ─────────────────────────────────────────────────────────────────
    # Server mode inference
    # ─────────────────────────────────────────────────────────────────

    async def _infer_server(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Inference via FastAPI GO-1 server (POST /act)."""
        import requests

        payload = self._build_server_payload(observation_dict, instruction, **kwargs)

        resp = requests.post(f"{self._server_url}/act", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        # Server returns list of action arrays or single array
        actions = np.array(result, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        self._step += 1
        return self._actions_to_dicts(actions)

    def _build_server_payload(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> Dict[str, Any]:
        """Build the JSON payload for the GO-1 FastAPI server."""
        payload: Dict[str, Any] = {"instruction": instruction}

        # Extract camera images
        cameras = self._extract_cameras(observation_dict)
        for go1_key, image in cameras.items():
            # Map to server convention: cam_head_color → "top", etc.
            server_key = self._go1_key_to_server_key(go1_key)
            if image is not None:
                payload[server_key] = image.tolist() if isinstance(image, np.ndarray) else image

        # Extract state
        state = self._extract_state(observation_dict)
        payload["state"] = state.tolist()

        # Control frequency
        ctrl_freq = kwargs.get("ctrl_freq", self._ctrl_freq)
        payload["ctrl_freqs"] = [float(ctrl_freq)]

        return payload

    @staticmethod
    def _go1_key_to_server_key(go1_key: str) -> str:
        """Map internal camera key to server POST payload key."""
        mapping = {
            "cam_head_color": "top",
            "cam_hand_right_color": "right",
            "cam_hand_left_color": "left",
        }
        return mapping.get(go1_key, go1_key)

    # ─────────────────────────────────────────────────────────────────
    # Local mode inference
    # ─────────────────────────────────────────────────────────────────

    def _infer_local(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Direct local GPU inference (requires GO-1 model loaded)."""
        import torch
        from PIL import Image

        if self._model is None:
            logger.warning("GO-1 model not loaded, returning zero actions")
            return self._generate_zero_actions()

        # Build inputs for GO-1 model
        cameras = self._extract_cameras(observation_dict)
        state = self._extract_state(observation_dict)

        # Normalize state if needed
        state_tensor = torch.from_numpy(state.copy()).float()
        if self._data_stats and "state" in self._data_stats:
            state_tensor = self._normalize_tensor(state_tensor, self._data_stats["state"])

        ctrl_freq = kwargs.get("ctrl_freq", self._ctrl_freq)
        ctrl_freqs = torch.tensor([ctrl_freq], dtype=torch.float32)

        # Process images through InternVL pipeline
        raw_target = {
            "final_prompt": f"What action should the robot take to {instruction}?",
        }
        for go1_key, image_array in cameras.items():
            if image_array is not None:
                raw_target[go1_key] = Image.fromarray(image_array.astype(np.uint8))

        # Try to use the model's native preprocessing if available
        try:
            inputs = self._preprocess_inputs(raw_target)
        except Exception as e:
            logger.warning(f"GO-1 preprocessing failed: {e}, returning zero actions")
            return self._generate_zero_actions()

        inputs["state"] = state_tensor
        inputs["ctrl_freqs"] = ctrl_freqs

        # Run inference
        device = self._device
        with torch.no_grad():
            action = self._model(
                pixel_values=inputs["pixel_values"].to(dtype=torch.bfloat16, device=device),
                input_ids=inputs["input_ids"].to(device).unsqueeze(0),
                attention_mask=inputs["attention_mask"].to(device).unsqueeze(0),
                position_ids=inputs["position_ids"].to(device).unsqueeze(0),
                image_flags=inputs["image_flags"].to(device),
                state=inputs["state"].to(dtype=torch.bfloat16, device=device).unsqueeze(0),
                ctrl_freqs=inputs["ctrl_freqs"].to(dtype=torch.bfloat16, device=device).unsqueeze(0),
            )

        # action[1] is the predicted actions, shape: (batch, chunk_size, action_dim)
        outputs = action[1][0].float().cpu()

        # Unnormalize if needed
        if self._data_stats and "action" in self._data_stats:
            outputs = self._unnormalize_tensor(outputs, self._data_stats["action"])

        actions_np = outputs.numpy()
        self._step += 1
        return self._actions_to_dicts(actions_np)

    def _preprocess_inputs(self, raw_target: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess inputs using InternVL2.5 image/text pipeline.

        This replicates the multi_image_get_item() from the GO-1 deploy.py.
        Falls back to a simpler path if full preprocessing isn't available.
        """
        import torch
        from PIL import Image

        # Collect camera images
        images = []
        num_tiles = []
        for cam_key in self._camera_keys:
            if cam_key in raw_target and raw_target[cam_key] is not None:
                img = raw_target[cam_key]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.asarray(img).astype(np.uint8))
                images.append(img)
                num_tiles.append(1)

        if not images:
            # No cameras — create placeholder
            images = [Image.new("RGB", (448, 448))]
            num_tiles = [1]

        # Try to use the model's image transform
        if self._img_transform is not None:
            pixel_values = torch.stack([self._img_transform(img) for img in images])
        else:
            # Fallback: basic resize + normalize
            pixel_values = self._basic_image_preprocess(images)

        num_image = len(images)

        # Build conversation-style input
        image_markers = "<image>" * num_image
        prompt = raw_target.get("final_prompt", "What action should the robot take?")
        full_prompt = f"{image_markers}{prompt}"

        # Tokenize
        if self._tokenizer is not None:
            tokens = self._tokenizer(full_prompt, return_tensors="pt")
            input_ids = tokens["input_ids"][0]
            attention_mask = tokens["attention_mask"][0]
        else:
            # Fallback: dummy tokens
            input_ids = torch.zeros(128, dtype=torch.long)
            attention_mask = torch.ones(128, dtype=torch.long)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        image_flags = torch.ones(pixel_values.size(0), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }

    @staticmethod
    def _basic_image_preprocess(images: list) -> "torch.Tensor":  # noqa: F821
        """Fallback image preprocessing (resize + normalize to tensor)."""
        import torch

        tensors = []
        for img in images:
            img = img.convert("RGB").resize((448, 448))
            arr = np.array(img, dtype=np.float32) / 255.0
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            # HWC → CHW
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
            tensors.append(tensor)
        return torch.stack(tensors)

    @staticmethod
    def _normalize_tensor(data, stats):
        """Normalize tensor using mean/std stats."""
        return (data - stats["mean"]) / (stats["std"] + 1e-6)

    @staticmethod
    def _unnormalize_tensor(data, stats):
        """Unnormalize tensor using mean/std stats."""
        return data * stats["std"] + stats["mean"]

    # ─────────────────────────────────────────────────────────────────
    # Observation extraction utilities
    # ─────────────────────────────────────────────────────────────────

    def _extract_cameras(self, observation_dict: Dict[str, Any]) -> Dict[str, Optional[np.ndarray]]:
        """Extract camera images from observation dict, mapping aliases.

        Returns dict with GO-1 camera keys → numpy arrays (HxWx3).
        """
        cameras: Dict[str, Optional[np.ndarray]] = {}

        for go1_key, aliases in _CAMERA_ALIASES.items():
            image = None
            # Try the GO-1 key directly first
            if go1_key in observation_dict:
                image = self._to_numpy_image(observation_dict[go1_key])
            else:
                # Try aliases
                for alias in aliases:
                    if alias in observation_dict:
                        image = self._to_numpy_image(observation_dict[alias])
                        break
            cameras[go1_key] = image

        # Fallback: if no cameras matched, try to find any image-like array
        if all(v is None for v in cameras.values()):
            for key in sorted(observation_dict.keys()):
                val = observation_dict[key]
                if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] in (3, 4):
                    cameras["cam_head_color"] = val[:, :, :3].astype(np.uint8)
                    break

        return cameras

    @staticmethod
    def _to_numpy_image(val: Any) -> Optional[np.ndarray]:
        """Convert various image formats to numpy HxWx3 uint8."""
        if isinstance(val, np.ndarray):
            if val.ndim == 3 and val.shape[-1] in (3, 4):
                return val[:, :, :3].astype(np.uint8)
            if val.ndim == 3 and val.shape[0] in (3, 4):
                # CHW → HWC
                return val[:3].transpose(1, 2, 0).astype(np.uint8)
        try:
            from PIL import Image

            if isinstance(val, Image.Image):
                return np.array(val.convert("RGB"), dtype=np.uint8)
        except ImportError:
            pass
        return None

    def _extract_state(self, observation_dict: Dict[str, Any]) -> np.ndarray:
        """Extract proprioceptive state vector (16-dim for GO-1).

        Looks for:
        1. Individual robot_state_keys in observation_dict
        2. "observation.state" array key
        3. "state" array key
        4. Falls back to zeros
        """
        # Try robot state keys
        if self._robot_state_keys:
            state = np.array(
                [float(observation_dict.get(k, 0.0)) for k in self._robot_state_keys],
                dtype=np.float32,
            )
            # Pad or truncate to action_dim
            if len(state) < self._action_dim:
                state = np.pad(state, (0, self._action_dim - len(state)))
            elif len(state) > self._action_dim:
                state = state[: self._action_dim]
            return state

        # Try array keys
        for key in ["observation.state", "state", "robot_state", "qpos"]:
            if key in observation_dict:
                val = observation_dict[key]
                state = np.asarray(val, dtype=np.float32).flatten()
                if len(state) < self._action_dim:
                    state = np.pad(state, (0, self._action_dim - len(state)))
                elif len(state) > self._action_dim:
                    state = state[: self._action_dim]
                return state

        return np.zeros(self._action_dim, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────
    # Action output conversion
    # ─────────────────────────────────────────────────────────────────

    def _actions_to_dicts(self, actions: np.ndarray) -> List[Dict[str, Any]]:
        """Convert (chunk_size, action_dim) array to list of action dicts.

        Each dict maps robot_state_keys (or default joint keys) to float values.
        """
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        keys = self._robot_state_keys if self._robot_state_keys else _DEFAULT_JOINT_KEYS
        result = []
        for i in range(actions.shape[0]):
            action_dict = {}
            for j, key in enumerate(keys):
                if j < actions.shape[1]:
                    action_dict[key] = float(actions[i, j])
                else:
                    action_dict[key] = 0.0
            result.append(action_dict)
        return result

    def _generate_zero_actions(self) -> List[Dict[str, Any]]:
        """Generate zero actions for error/fallback cases."""
        keys = self._robot_state_keys if self._robot_state_keys else _DEFAULT_JOINT_KEYS
        zero_action = {k: 0.0 for k in keys}
        return [zero_action.copy() for _ in range(self._action_chunk_size)]

    # ─────────────────────────────────────────────────────────────────
    # Bonus: extract single image (for compatibility)
    # ─────────────────────────────────────────────────────────────────

    def _extract_image(self, observation_dict: Dict[str, Any]):
        """Extract a single PIL Image from observation dict (compatibility)."""
        from PIL import Image

        cameras = self._extract_cameras(observation_dict)
        for img in cameras.values():
            if img is not None:
                return Image.fromarray(img)
        return Image.new("RGB", (448, 448))

    def reset(self):
        """Reset policy state for a new episode."""
        self._step = 0


__all__ = ["Go1Policy"]
