# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public raw-frame + camera-params APIs on the MuJoCo backend (issue #1537).

``get_camera_params`` is pure model math (no GL); ``get_frame`` and the
end-to-end HybridCompositor test need an offscreen GL context and are gated
behind the shared runtime probe.
"""

import os

import numpy as np
import pytest

pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl


def _make_sim(width: int = 64, height: int = 48):
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("front", position=[0.4, -0.5, 0.3], target=[0.0, 0.0, 0.1], width=width, height=height)
    sim.step(n_steps=5)
    return sim


# ----- get_camera_params (no GL required) ----- #


def test_get_camera_params_math_and_conventions() -> None:
    sim = _make_sim()
    try:
        cam = sim.get_camera_params("front")
        assert cam.width == 64 and cam.height == 48
        # K: square pixels, centered principal point, fy from the camera's
        # vertical FOV (add_camera default fov=60 deg).
        assert cam.K.shape == (3, 3)
        fy_expected = 0.5 * 48 / np.tan(np.deg2rad(60.0) / 2.0)
        assert cam.K[1, 1] == pytest.approx(fy_expected, rel=1e-6)
        assert cam.K[0, 0] == pytest.approx(cam.K[1, 1])
        assert cam.K[0, 2] == pytest.approx(32.0)
        assert cam.K[1, 2] == pytest.approx(24.0)
        # Pose: rotation is orthonormal, translation is the camera position.
        R = cam.T_world_cam[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.allclose(cam.T_world_cam[:3, 3], [0.4, -0.5, 0.3], atol=1e-6)
        # OpenGL optical convention: -Z (third column negated) points from
        # the eye towards the look-at target.
        fwd = -R[:, 2]
        expected = np.array([0.0, 0.0, 0.1]) - np.array([0.4, -0.5, 0.3])
        expected /= np.linalg.norm(expected)
        assert np.allclose(fwd, expected, atol=1e-6)
        assert 0.0 < cam.znear < cam.zfar
    finally:
        sim.destroy()


def test_get_camera_params_width_height_override() -> None:
    sim = _make_sim()
    try:
        cam = sim.get_camera_params("front", width=128, height=96)
        assert cam.width == 128 and cam.height == 96
        assert cam.K[0, 2] == pytest.approx(64.0)
    finally:
        sim.destroy()


def test_get_camera_params_rejects_free_camera_and_unknown_names() -> None:
    sim = _make_sim()
    try:
        with pytest.raises(ValueError, match="free camera"):
            sim.get_camera_params("default")
        with pytest.raises(KeyError, match="not found"):
            sim.get_camera_params("nope")
    finally:
        sim.destroy()


def test_get_camera_params_requires_world() -> None:
    from strands_robots.simulation import Simulation

    sim = Simulation()
    with pytest.raises(RuntimeError):
        sim.get_camera_params("front")


# ----- get_frame + compositor (GL required) ----- #


@requires_gl
def test_get_frame_returns_raw_rgb_and_metric_depth() -> None:
    sim = _make_sim()
    try:
        rgb, depth = sim.get_frame("front")
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8
        assert depth is not None
        assert depth.shape == (48, 64)
        assert depth.dtype == np.float32
        assert np.isfinite(depth).all()  # sanitized: no NaN/inf
        assert (depth >= 0).all()
        # The scene has geometry: some pixel is nearer than the far clip.
        cam = sim.get_camera_params("front")
        assert float(depth.min()) < cam.zfar * 0.999
    finally:
        sim.destroy()


@requires_gl
def test_get_frame_unknown_camera_raises() -> None:
    sim = _make_sim()
    try:
        with pytest.raises(KeyError, match="not found"):
            sim.get_frame("nope")
    finally:
        sim.destroy()


@requires_gl
def test_get_frame_rejects_bad_dimensions() -> None:
    sim = _make_sim()
    try:
        with pytest.raises(ValueError):
            sim.get_frame("front", width=0, height=48)
    finally:
        sim.destroy()


@requires_gl
def test_hybrid_compositor_end_to_end_over_mujoco() -> None:
    from strands_robots.rendering import HybridCompositor

    sim = _make_sim()
    try:
        frame = HybridCompositor(sim, feather_pixels=0).render("front")
        assert frame.rgb.shape == (48, 64, 3)
        assert frame.rgb.dtype == np.uint8
        # Both regimes present: some robot pixels won, some background shows.
        assert bool(frame.foreground_mask.any())
        assert not bool(frame.foreground_mask.all())
    finally:
        sim.destroy()
