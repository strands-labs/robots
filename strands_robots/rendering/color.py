# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Color-space utilities and the hybrid-render color-pipeline contract.

The hybrid composite blends layers produced by unrelated renderers, so the
color space each layer arrives in is part of the compositing contract. This
module is the single written-down decision (issue #2323, stage 3):

**What space each layer is in**

* *Simulation foreground* -- MuJoCo's offscreen renderer writes an sRGB-encoded
  framebuffer; Isaac's RTX real-time path emits tonemapped display-referred
  output. Both arrive as display-space (gamma-encoded) ``uint8``.
* *3DGS background* -- splat DC colors are optimized so the rasterized image
  reproduces the *training photographs*, which are display-space sRGB. The
  rasterizer's output is therefore display-referred too (see
  :mod:`strands_robots.rendering.backgrounds`).
* *Panorama background* -- an ordinary ``.jpg``/``.png``, i.e. sRGB.

**The one decision, applied once**

Every layer is nominally display-space sRGB, so the compositor's default
byte-for-byte blend keeps layers aligned *without* any conversion -- converting
only some layers would be the actual bug. The two operations that are *light
arithmetic* rather than layer passthrough must run in linear space, and both
go through the helpers here:

* seam blending (``HybridCompositor(blend_in_linear=True)``) -- averaging
  gamma-encoded bytes darkens the fg/bg seam; averaging linear light does not.
* shadow-catcher modulation (``HybridCompositor(shadow_plane_z=...)``) -- a
  shadow is a multiplicative ratio of received light, which is only a
  multiplication in linear space.

Transfer function: IEC 61966-2-1 (sRGB), the piecewise curve -- not a bare 2.2
power -- so round trips are exact to float precision. Luminance weights are
Rec. 709 / sRGB primaries.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["linear_to_srgb", "relative_luminance", "srgb_to_linear"]


def srgb_to_linear(rgb: npt.ArrayLike) -> np.ndarray:
    """Decode sRGB-encoded values to linear light.

    Args:
        rgb: array of sRGB-encoded values, either ``uint8`` in ``[0, 255]`` or
            float in ``[0, 1]``. Any shape.

    Returns:
        ``float32`` array of the same shape, linear light in ``[0, 1]``.
    """
    x = np.asarray(rgb)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    # IEC 61966-2-1: linear segment below the knee, power segment above.
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(linear: npt.ArrayLike) -> np.ndarray:
    """Encode linear-light values to sRGB.

    Args:
        linear: float array of linear-light values in ``[0, 1]``. Any shape.
            Values outside the range are clipped (display-referred output has
            nowhere else to put them).

    Returns:
        ``float32`` array of the same shape, sRGB-encoded in ``[0, 1]``.
    """
    x = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055).astype(np.float32)


def relative_luminance(linear_rgb: npt.ArrayLike) -> np.ndarray:
    """Rec. 709 relative luminance of *linear-light* RGB.

    Args:
        linear_rgb: ``(..., 3)`` float array of linear-light RGB in ``[0, 1]``
            (decode display-space input with :func:`srgb_to_linear` first --
            luminance weights are only meaningful on linear values).

    Returns:
        ``(...)`` ``float32`` array of luminance in ``[0, 1]``.
    """
    x = np.asarray(linear_rgb, dtype=np.float32)
    result: np.ndarray = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).astype(np.float32)
    return result
