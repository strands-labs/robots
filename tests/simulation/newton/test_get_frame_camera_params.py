# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw-frame API contract on the Newton backend (issue #1537).

Newton renders RGB only: ``get_frame`` must return ``(rgb, None)`` -- never a
zero-filled depth buffer -- and ``get_camera_params`` reports the look-at pose
in the OpenGL optical convention. Skipped when Newton/Warp are not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


@pytest.fixture
def engine_with_camera():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="mujoco")
    sim.create_world()
    sim.add_robot("so101")
    sim.add_camera("front", position=[0.5, -0.5, 0.4], target=[0.0, 0.0, 0.1], width=64, height=48)
    yield sim
    sim.destroy()


def test_get_frame_returns_rgb_and_none_depth(engine_with_camera) -> None:
    rgb, depth = engine_with_camera.get_frame("front")
    assert rgb.shape == (48, 64, 3)
    assert rgb.dtype == np.uint8
    assert depth is None  # Newton has no depth path -- explicit, not zeros


def test_get_frame_unknown_camera_raises(engine_with_camera) -> None:
    with pytest.raises(KeyError, match="not found"):
        engine_with_camera.get_frame("nope")


def test_get_camera_params_lookat_pose_and_pinhole_k(engine_with_camera) -> None:
    cam = engine_with_camera.get_camera_params("front")
    assert cam.width == 64 and cam.height == 48
    fy_expected = 0.5 * 48 / np.tan(np.deg2rad(60.0) / 2.0)  # add_camera default fov
    assert cam.K[1, 1] == pytest.approx(fy_expected, rel=1e-6)
    R = cam.T_world_cam[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(cam.T_world_cam[:3, 3], [0.5, -0.5, 0.4], atol=1e-9)
    fwd = -R[:, 2]  # -Z forward (OpenGL optical convention)
    expected = np.array([0.0, 0.0, 0.1]) - np.array([0.5, -0.5, 0.4])
    expected /= np.linalg.norm(expected)
    assert np.allclose(fwd, expected, atol=1e-9)


def test_compositor_refuses_newton_missing_depth(engine_with_camera) -> None:
    from strands_robots.rendering import HybridCompositor

    with pytest.raises(RuntimeError, match="depth"):
        HybridCompositor(engine_with_camera).render("front")
