"""DiffusersKimodoAgent - real HF diffusers-backed sampler for Kimodo.

Kept in a separate module so ``from strands_robots.policies.kimodo import
KimodoPolicy`` NEVER pulls torch/diffusers at import time. The policy lazily
imports this on the first sample call when no ``motion_agent`` was injected.

This agent loads a checkpoint published in *diffusers pipeline layout* - one
carrying a ``model_index.json`` index. ``trust_remote_code`` is forwarded to
``from_pretrained`` for pipelines that ship custom code, gated at the factory
layer (:mod:`strands_robots.policies.factory`) via the
``STRANDS_TRUST_REMOTE_CODE`` env var so the user opts in explicitly.

NVIDIA's own Kimodo checkpoints are NOT in that layout. ``nvidia/Kimodo-G1-RP-v1``
publishes bare weights (``config.yaml`` plus ``model.safetensors``) consumed by
its own runtime, and the Hub declares ``library_name: kimodo``. There is no
``model_index.json``, so ``from_pretrained`` cannot load it at all. This agent
refuses such a target at construction, naming the remedy rather than surfacing a
bare 404 for a file that will never exist: pass ``motion_agent=`` with a sampler
that loads the checkpoint through its own runtime.

The agent implements :class:`~strands_robots.policies.kimodo.policy.KimodoMotionAgent`
and returns a numpy ``(num_frames, 7+29)`` qpos array. Frame layout matches
the sampler's native output: root position (3) + root quaternion (4) + 29
joint angles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import KimodoConfig

logger = logging.getLogger(__name__)


def _output_field_names(output: Any) -> str:
    """Name the fields a pipeline output carries, for a refusal message.

    Args:
        output: Whatever the diffusion pipeline returned.

    Returns:
        A comma-separated field listing, or a short phrase when none is
        readable. Never raises - it only ever builds an error message.
    """
    if isinstance(output, dict):
        names = sorted(str(key) for key in output)
    else:
        names = sorted(str(name) for name in getattr(output, "__dict__", {}))
    return "fields " + ", ".join(names) if names else "no readable fields"


_PIPELINE_INDEX_FILE = "model_index.json"


def _carries_no_pipeline_index(model_id: str, error: Exception) -> bool:
    """Decide whether *error* means *model_id* has no diffusers pipeline index.

    ``from_pretrained`` reports a missing index differently per target, and both
    shapes are ``OSError`` - as are genuine transport failures, which must NOT be
    reported as a layout problem. So this narrows structurally:

    * Hub repo: ``huggingface_hub`` raises ``EntryNotFoundError`` for the entry it
      could not resolve. The filename guard keeps the refusal honest when the
      absent entry is some *other* file (a pipeline component), in which case the
      original error is the accurate one and is re-raised untouched.
    * Local directory: diffusers raises a plain ``OSError``, so the index is
      probed on disk directly.

    Args:
        model_id: The configured Hub repo id or local checkpoint directory.
        error: The error ``from_pretrained`` raised. Deliberately not narrowed to
            ``OSError``: only ``RemoteEntryNotFoundError`` inherits it, while the
            bare ``EntryNotFoundError`` subclasses ``Exception`` alone.

    Returns:
        True when the target demonstrably carries no ``model_index.json``.
    """
    from huggingface_hub.errors import EntryNotFoundError

    if isinstance(error, EntryNotFoundError):
        return _PIPELINE_INDEX_FILE in str(error)
    directory = Path(model_id)
    return directory.is_dir() and not (directory / _PIPELINE_INDEX_FILE).is_file()


def _not_a_pipeline_error(model_id: str) -> str:
    """Build the refusal for a target that is not a diffusers pipeline.

    Args:
        model_id: The configured Hub repo id or local checkpoint directory.

    Returns:
        A message naming the missing index, why the target cannot be loaded, and
        the two remedies.
    """
    return (
        f"Kimodo model_id '{model_id}' is not a diffusers pipeline: it carries no "
        f"{_PIPELINE_INDEX_FILE}, so DiffusionPipeline.from_pretrained cannot load it. NVIDIA's "
        "Kimodo checkpoints publish bare weights (config.yaml plus model.safetensors) for their "
        "own runtime - the Hub declares library_name 'kimodo', not 'diffusers'. Pass motion_agent= "
        "with a sampler that loads this checkpoint through its own runtime and returns a "
        "(num_frames, 7+29) qpos array, or point model_id at a checkpoint published in diffusers "
        "pipeline layout."
    )


class DiffusersKimodoAgent:
    """HuggingFace diffusers-backed sampler agent."""

    def __init__(self, config: KimodoConfig) -> None:
        self.config = config
        # Deferred imports - only runs when someone actually samples.
        import torch
        from diffusers import DiffusionPipeline
        from huggingface_hub.errors import EntryNotFoundError

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
        try:
            self._pipe = DiffusionPipeline.from_pretrained(
                config.model_id,
                torch_dtype=torch_dtype,
                cache_dir=config.cache_dir,
                trust_remote_code=config.trust_remote_code,
            )
        except (OSError, EntryNotFoundError) as err:
            if not _carries_no_pipeline_index(config.model_id, err):
                raise
            raise RuntimeError(_not_a_pipeline_error(config.model_id)) from err
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
        # (num_frames, 7+29) already in qpos convention. Read the mapping form
        # with ``get`` rather than a subscript: a diffusers pipeline output is a
        # ``BaseOutput``, which subclasses ``OrderedDict``, so a subscript for a
        # field the checkpoint does not carry raises ``KeyError`` before the
        # refusal below can name the model and the remedy.
        motion = getattr(output, "motion", None)
        if motion is None and isinstance(output, dict):
            motion = output.get("motion")
        if motion is None:
            raise RuntimeError(
                f"Kimodo pipeline output for model_id '{self.config.model_id}' carries no 'motion' "
                f"field: got {type(output).__name__} with {_output_field_names(output)}. Kimodo emits "
                "per-frame qpos under 'motion' - point model_id at a Kimodo checkpoint, or pass "
                "motion_agent= to adapt a sampler that names its output differently."
            )
        arr = motion.detach().to("cpu").float().numpy().astype(self._np_dtype)
        # Some checkpoints return (B, T, D) with B=1 - squeeze.
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr
