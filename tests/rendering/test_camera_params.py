# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CameraParams contract: intrinsics round-trip + immutability."""

import numpy as np
import pytest

from strands_robots.rendering import CameraParams


def _params(width: int = 640, height: int = 480, fovy_deg: float = 45.0) -> CameraParams:
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    K = np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
    return CameraParams(K=K, T_world_cam=np.eye(4), width=width, height=height, znear=0.01, zfar=100.0)


def test_fovy_rad_round_trips_through_k() -> None:
    for fovy_deg in (30.0, 45.0, 60.0, 90.0):
        cam = _params(fovy_deg=fovy_deg)
        assert cam.fovy_rad == pytest.approx(np.deg2rad(fovy_deg), rel=1e-9)


def test_camera_params_is_frozen() -> None:
    cam = _params()
    with pytest.raises(AttributeError):
        cam.width = 100  # type: ignore[misc]
