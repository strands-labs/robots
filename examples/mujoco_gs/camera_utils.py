# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera intrinsics / extrinsics / depth helpers -- now provided by the library.

The camera math this module used to implement by reaching through
``sim.mj_model`` / ``sim.mj_data`` is now a public backend API (issue #1537):

* ``sim.get_camera_params(name, width=..., height=...)`` returns the
  :class:`strands_robots.rendering.CameraParams` (intrinsic ``K``,
  world-from-camera pose in the OpenGL optical convention, clip planes).
* ``sim.get_frame(name, width=..., height=...)`` returns the raw
  ``(rgb_uint8, depth_float32_meters)`` pair, with the GL renderer cached
  per-thread and the benchmark viz option applied inside the backend.

The thin wrappers below keep this example's historical call shape
(``fn(sim, camera, w, h)``); new code should call the ``sim`` methods
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from strands_robots.rendering import CameraParams

if TYPE_CHECKING:  # avoid hard import at module load -- sim deps are optional
    from strands_robots.simulation import Simulation

__all__ = ["CameraParams", "get_camera_params", "render_rgb_and_depth"]


def get_camera_params(sim: "Simulation", camera_name: str, width: int, height: int) -> CameraParams:
    """Return :class:`CameraParams` for ``camera_name`` (library delegation)."""
    return sim.get_camera_params(camera_name, width=width, height=height)


def render_rgb_and_depth(
    sim: "Simulation", camera_name: str, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Render one frame as ``(rgb_uint8, depth_metric_float32)`` (library delegation)."""
    rgb, depth = sim.get_frame(camera_name, width=width, height=height)
    assert depth is not None  # MuJoCo always produces depth
    return rgb, depth
