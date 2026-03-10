#!/usr/bin/env python3
"""
Shared utilities for policy providers.

Consolidates common patterns that were copy-pasted across 10+ policy modules:
- Image extraction from observation dicts → PIL Image
- Device auto-detection (CUDA → MPS → CPU)
- Number parsing from VLM text output → numpy arrays

These are intentionally simple, stateless helper functions — no classes, no state.
Import them directly into your policy module.

Usage:
    from strands_robots.policies._utils import extract_pil_image, detect_device, parse_numbers_from_text

    image = extract_pil_image(observation_dict)
    device = detect_device(requested_device=None)
    actions = parse_numbers_from_text(text, action_dim=7)
"""

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_pil_image(
    observation_dict: Dict[str, Any],
    preferred_key: Optional[str] = None,
    fallback_size: tuple = (224, 224),
):
    """Extract a PIL Image from an observation dictionary.

    Searches for the first 3D numpy array with 3 or 4 channels (HWC format).
    If `preferred_key` is given, checks that key first before scanning others.
    Returns a blank image if nothing found (never raises).

    Args:
        observation_dict: Robot observation containing camera images as numpy arrays.
        preferred_key: Optional key to check first (e.g. "primary_image", "wrist").
        fallback_size: Size of blank fallback image if no camera found.

    Returns:
        PIL.Image.Image in RGB mode.

    Examples:
        >>> image = extract_pil_image(obs)
        >>> image = extract_pil_image(obs, preferred_key="wrist_image")
    """
    import numpy as np
    from PIL import Image

    # Check preferred key first
    if preferred_key and preferred_key in observation_dict:
        val = observation_dict[preferred_key]
        if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] in (3, 4):
            return Image.fromarray(val[:, :, :3].astype(np.uint8))

    # Scan all keys alphabetically for any image-like array
    for key in sorted(observation_dict.keys()):
        val = observation_dict[key]
        if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] in (3, 4):
            return Image.fromarray(val[:, :, :3].astype(np.uint8))

    logger.debug("No camera image found in observation, returning blank image")
    return Image.new("RGB", fallback_size)


def detect_device(requested_device: Optional[str] = None) -> str:
    """Auto-detect the best available compute device.

    Priority: explicit request → CUDA → Apple MPS → CPU.

    Args:
        requested_device: If provided, returned as-is (user override).

    Returns:
        Device string suitable for torch.device() — e.g. "cuda:0", "mps", "cpu".

    Examples:
        >>> detect_device()          # Auto: "cuda:0" if GPU available
        >>> detect_device("cpu")     # Explicit: "cpu"
        >>> detect_device("cuda:1")  # Explicit: "cuda:1"
    """
    if requested_device:
        return requested_device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def parse_numbers_from_text(
    text: str,
    action_dim: int = 7,
    take_last: bool = True,
):
    """Parse floating-point numbers from VLM-generated text.

    VLMs output action values as text (e.g. "Action: 0.12, -0.34, 0.56, ...").
    This extracts all numbers and returns the last `action_dim` values,
    padding with zeros if not enough numbers are found.

    Args:
        text: Raw text output from a VLM/LLM.
        action_dim: Expected number of action dimensions.
        take_last: If True, take the last N numbers (common for VLMs that
                   echo the prompt before outputting actions). If False,
                   take the first N.

    Returns:
        numpy float32 array of shape (action_dim,).

    Examples:
        >>> parse_numbers_from_text("Action: 0.1, -0.2, 0.3", action_dim=3)
        array([ 0.1, -0.2,  0.3], dtype=float32)

        >>> parse_numbers_from_text("The answer is 42", action_dim=3)
        array([ 0., 0., 42.], dtype=float32)
    """
    import numpy as np

    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    values = [float(n) for n in numbers]

    if take_last:
        values = values[-action_dim:]
    else:
        values = values[:action_dim]

    # Pad with zeros if fewer numbers found than action_dim
    while len(values) < action_dim:
        values.append(0.0)

    return np.array(values[:action_dim], dtype=np.float32)


__all__ = ["extract_pil_image", "detect_device", "parse_numbers_from_text"]
