#!/usr/bin/env python3
"""
RDT Policy Provider — Robotics Diffusion Transformer (Tsinghua TSAIL).

1B-param diffusion transformer pre-trained on 1M+ multi-robot episodes.
Predicts 64 future actions via iterative denoising.
Supports single-arm, dual-arm, joint/EEF, position/velocity, and wheeled locomotion
via a unified 128-dim action space.

Checkpoints:
    - robotics-diffusion-transformer/rdt-1b (base, 100⭐)
    - robotics-diffusion-transformer/RDT2-VQ (v2 with VQ tokenization)

Architecture: SigLIP vision + T5-XXL language → Diffusion Transformer → 64 actions

Reference: Liu et al., "RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation", 2024
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy
from strands_robots.policies._utils import detect_device, extract_pil_image

logger = logging.getLogger(__name__)


class RdtPolicy(Policy):
    """RDT — Robotics Diffusion Transformer.

    Uses diffusion denoising to predict action chunks (64 steps).
    Supports the unified 128-dim action space covering all manipulator types.
    """

    # T5 encoder model ID — this is ~22GB and will be downloaded on first use.
    # Override with a smaller encoder via the `t5_model_id` constructor param.
    DEFAULT_T5_MODEL = "google/t5-v1_1-xxl"

    def __init__(
        self,
        model_id: str = "robotics-diffusion-transformer/rdt-1b",
        repo_path: Optional[str] = None,
        state_dim: int = 14,
        chunk_size: int = 64,
        camera_names: Optional[List[str]] = None,
        control_frequency: int = 25,
        device: Optional[str] = None,
        actions_per_step: int = 1,
        t5_model_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize RDT policy.

        Args:
            model_id: HuggingFace model ID for the RDT checkpoint.
            repo_path: Path to cloned RDT repo (required for full inference).
                This path is added to sys.path at load time — only use trusted sources.
            state_dim: Robot proprioceptive state dimension.
            chunk_size: Number of future actions to predict per inference.
            camera_names: Camera key names in the observation dict.
            control_frequency: Robot control frequency in Hz.
            device: Inference device (auto-detected if None).
            actions_per_step: How many actions from the chunk to return per call.
            t5_model_id: Override for the T5 encoder model. Defaults to
                google/t5-v1_1-xxl (~22GB). Set to a smaller T5 variant
                (e.g. "google/t5-v1_1-base") for faster setup at the cost
                of language understanding quality.
        """
        self._model_id = model_id
        self._repo_path = repo_path
        self._state_dim = state_dim
        self._chunk_size = chunk_size
        self._camera_names = camera_names or ["cam_high", "cam_right_wrist", "cam_left_wrist"]
        self._control_frequency = control_frequency
        self._requested_device = device
        self._actions_per_step = actions_per_step
        self._t5_model_id = t5_model_id or self.DEFAULT_T5_MODEL
        self._robot_state_keys: List[str] = []

        self._model = None
        self._loaded = False
        self._step = 0
        self._lang_embeddings = None

        logger.info(f"🌊 RDT policy: {model_id} (state_dim={state_dim}, chunk={chunk_size})")

    @property
    def provider_name(self) -> str:
        return "rdt"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys
        self._state_dim = len(robot_state_keys)

    def _ensure_loaded(self):
        if self._loaded:
            return

        import torch

        logger.info(f"🌊 Loading RDT from {self._model_id}...")
        start = time.time()

        self._device = detect_device(self._requested_device)

        try:
            # RDT uses its own model creation from the repo.
            # Security note: repo_path is inserted into sys.path so that
            # RDT's scripts.agilex_model can be imported. This affects all
            # subsequent imports in the process. Only use repo_path values
            # from trusted sources — a malicious directory could shadow
            # standard library or third-party modules.
            import sys

            # NOTE(security): This inserts user-provided repo_path into sys.path.
            # The repo directory must be trusted — it can contain arbitrary Python
            # modules that will be importable by the entire process.
            if self._repo_path and self._repo_path not in sys.path:
                sys.path.insert(0, self._repo_path)

            from scripts.agilex_model import create_model

            config = {
                "episode_len": 1000,
                "state_dim": self._state_dim,
                "chunk_size": self._chunk_size,
                "camera_names": self._camera_names,
            }

            self._model = create_model(
                args=config,
                dtype=torch.bfloat16,
                pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384",
                pretrained=self._model_id,
                control_frequency=self._control_frequency,
            )

            elapsed = time.time() - start
            logger.info(f"🌊 RDT loaded in {elapsed:.1f}s on {self._device}")

        except ImportError:
            logger.warning(
                "RDT requires cloning the repo: "
                "git clone https://github.com/thu-ml/RoboticsDiffusionTransformer\n"
                "Then pass repo_path= to the policy."
            )
            # Fallback: try loading as generic HF model
            self._load_hf_fallback()

        self._loaded = True

    def _load_hf_fallback(self):
        """Fallback: load RDT weights directly from HF without the repo."""
        import torch  # noqa: F401
        from huggingface_hub import hf_hub_download

        logger.info("🌊 RDT fallback: downloading weights from HF (repo not available)")
        # Just download and mark as loaded — actual inference needs the repo
        try:
            hf_hub_download(self._model_id, "config.json")
            logger.info(f"🌊 RDT config found at {self._model_id}")
        except Exception as e:
            logger.warning(f"Could not access RDT model: {e}")

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        self._ensure_loaded()

        if self._model is None:
            logger.error(
                "RDT model not loaded — need the RDT repo. "
                "git clone https://github.com/thu-ml/RoboticsDiffusionTransformer"
            )
            return [{k: 0.0 for k in self._robot_state_keys}]

        import torch

        # Encode language instruction
        if self._lang_embeddings is None or self._step == 0:
            self._lang_embeddings = self._encode_instruction(instruction)

        # Extract images for each camera
        images = []
        for cam_name in self._camera_names:
            found = False
            for key in observation_dict:
                if cam_name in key.lower():
                    val = observation_dict[key]
                    if isinstance(val, np.ndarray):
                        from PIL import Image

                        images.append(Image.fromarray(val[:, :, :3].astype(np.uint8)))
                        found = True
                        break
            if not found:
                # Fallback to shared utility for first available image, or blank
                images.append(extract_pil_image(observation_dict, fallback_size=(384, 384)))

        # Proprioception
        proprio = np.array(
            [float(observation_dict.get(k, 0.0)) for k in self._robot_state_keys],
            dtype=np.float32,
        )

        # Predict action chunk
        with torch.no_grad():
            actions = self._model.step(
                proprio=proprio,
                images=images,
                text_embeds=self._lang_embeddings,
            )

        # Convert chunk to action dicts
        actions_np = np.asarray(actions)
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(1, -1)

        result = []
        for i in range(min(self._actions_per_step, len(actions_np))):
            action_dict = {}
            for j, key in enumerate(self._robot_state_keys):
                if j < actions_np.shape[-1]:
                    action_dict[key] = float(actions_np[i, j])
            result.append(action_dict)

        self._step += 1
        return result if result else [{k: 0.0 for k in self._robot_state_keys}]

    def _encode_instruction(self, instruction: str):
        """Encode language instruction using T5.

        Warning: Downloads the T5 encoder model on first use. The default
        (google/t5-v1_1-xxl) is ~22GB. Override via t5_model_id constructor
        param to use a smaller variant.
        """
        import torch

        try:
            from transformers import T5EncoderModel, T5Tokenizer

            t5_model_id = self._t5_model_id
            logger.warning(
                f"🌊 Loading T5 encoder: {t5_model_id}. "
                + (
                    "This will download ~22GB on first use. "
                    "Set t5_model_id='google/t5-v1_1-base' for a smaller (~1GB) alternative."
                    if "xxl" in t5_model_id
                    else ""
                )
            )

            tokenizer = T5Tokenizer.from_pretrained(t5_model_id)
            encoder = T5EncoderModel.from_pretrained(t5_model_id, torch_dtype=torch.bfloat16)
            encoder = encoder.to(self._device).eval()

            tokens = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(self._device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = encoder(**tokens)
            embeddings = outputs.last_hidden_state

            del encoder  # Free memory
            return embeddings

        except Exception as e:
            logger.warning(f"T5 encoding failed: {e}. Using zeros.")
            return torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device=self._device)


__all__ = ["RdtPolicy"]
