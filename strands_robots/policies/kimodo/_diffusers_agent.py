"""DiffusersKimodoAgent - real HF diffusers-backed sampler for Kimodo.

Kept in a separate module so ``from strands_robots.policies.kimodo import
KimodoPolicy`` NEVER pulls torch/diffusers at import time. The policy lazily
imports this on the first sample call when no ``motion_agent`` was injected.

The upstream Kimodo checkpoint (``nvidia/Kimodo-G1-RP-v1``) ships a custom
``DiffusionPipeline`` class that requires ``trust_remote_code=True``. That's
gated at the factory layer (:mod:`strands_robots.policies.factory`) via the
``STRANDS_TRUST_REMOTE_CODE`` env var so the user opts in explicitly.

The agent implements :class:`~strands_robots.policies.kimodo.policy.KimodoMotionAgent`
and returns a numpy ``(num_frames, 7+29)`` qpos array. Frame layout matches
the sampler's native output: root position (3) + root quaternion (4) + 29
joint angles.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from .config import KimodoConfig

logger = logging.getLogger(__name__)


class DiffusersKimodoAgent:
    """HuggingFace diffusers-backed sampler agent."""

    def __init__(self, config: KimodoConfig) -> None:
        self.config = config
        # Deferred imports - only runs when someone actually samples.
        import torch
        from diffusers import DiffusionPipeline

        self._torch = torch
        self._np_dtype = np.float32

        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[config.dtype]

        logger.info(
            "Loading Kimodo pipeline: %s (device=%s dtype=%s)",
            config.model_id,
            device,
            config.dtype,
        )
        self._pipe = DiffusionPipeline.from_pretrained(
            config.model_id,
            torch_dtype=torch_dtype,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        self._pipe = self._pipe.to(device)
        self._device = device

    def sample(
        self,
        prompt: str,
        num_frames: int,
        diffusion_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> NDArray[np.float32]:
        """Run diffusion and return a ``(num_frames, 7+29)`` qpos array."""
        torch = self._torch
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self._device).manual_seed(int(seed))

        output = self._pipe(
            prompt=prompt,
            num_inference_steps=diffusion_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            generator=gen,
        )
        # Upstream returns `output.motion` as a torch tensor of shape
        # (num_frames, 7+29) already in qpos convention.
        motion = getattr(output, "motion", None)
        if motion is None:
            # Fallback: some versions use `output["motion"]`.
            motion = output["motion"] if isinstance(output, dict) else None
        if motion is None:
            raise RuntimeError("Kimodo pipeline output missing 'motion' field; is the checkpoint version compatible?")
        arr = motion.detach().to("cpu").float().numpy().astype(self._np_dtype)
        # Some checkpoints return (B, T, D) with B=1 - squeeze.
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr
