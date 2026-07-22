# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic pinhole camera parameters.

:class:`CameraParams` is the shared currency between the simulation
backends (``SimEngine.get_camera_params``) and the hybrid-render layer
(:mod:`strands_robots.rendering.backgrounds`,
:mod:`strands_robots.rendering.compositor`): a plain-numpy description of a
pinhole camera at a given image resolution, with the pose expressed in the
OpenGL optical convention every background renderer in this package assumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraParams:
    """Pinhole camera parameters at a given image resolution.

    Attributes:
        K: ``(3, 3)`` intrinsic matrix in pixels.
        T_world_cam: ``(4, 4)`` SE(3) pose. ``T_world_cam @ [x_cam; 1]`` gives
            the world-frame coordinates of a camera-frame point. The camera
            frame follows the OpenGL / MuJoCo optical convention: **+X right,
            +Y up, -Z forward**. Backends whose native camera frame differs
            (e.g. Isaac's USD camera prim) apply the fixed basis correction in
            their ``get_camera_params`` so consumers never see a
            backend-specific frame.
        width: image width in pixels.
        height: image height in pixels.
        znear: near-plane distance in meters.
        zfar: far-plane distance in meters. Background renderers report
            "at infinity" pixels as ``depth >= zfar`` so the compositor's
            depth test always picks a finite foreground.
    """

    K: np.ndarray
    T_world_cam: np.ndarray
    width: int
    height: int
    znear: float
    zfar: float

    @property
    def fovy_rad(self) -> float:
        """Vertical field-of-view in radians, recovered from K and image height."""
        fy = float(self.K[1, 1])
        return float(2.0 * np.arctan(0.5 * self.height / fy))
