# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Background renderer contracts (panorama path: zero ML deps)."""

import numpy as np
import pytest

from strands_robots.rendering import (
    GSPLAT_SCENES,
    CameraParams,
    PanoramaBackground,
    download_gsplat_scene,
    gsplat_scene_names,
    gsplat_skybox_align_for,
    gsplat_skybox_scene_names,
)


def _cam(width: int = 64, height: int = 48) -> CameraParams:
    fy = 0.5 * height / np.tan(np.deg2rad(45.0) / 2.0)
    K = np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
    return CameraParams(K=K, T_world_cam=np.eye(4), width=width, height=height, znear=0.01, zfar=100.0)


def test_procedural_panorama_renders_at_camera_resolution() -> None:
    cam = _cam()
    rgb, depth = PanoramaBackground().render(cam)
    assert rgb.shape == (48, 64, 3)
    assert rgb.dtype == np.uint8
    assert depth.shape == (48, 64)
    # Panorama sits at infinity: every pixel reports the far plane, so any
    # finite foreground wins the compositor's depth test.
    assert np.all(depth == np.float32(cam.zfar))


def test_procedural_panorama_is_deterministic() -> None:
    cam = _cam(32, 24)
    a, _ = PanoramaBackground().render(cam)
    b, _ = PanoramaBackground().render(cam)
    assert np.array_equal(a, b)


def test_panorama_missing_image_path_falls_back_to_procedural(tmp_path) -> None:
    bg = PanoramaBackground(image_path=tmp_path / "nope.jpg")
    rgb, _ = bg.render(_cam(16, 12))
    assert rgb.shape == (12, 16, 3)


def test_download_gsplat_scene_rejects_unknown_name(tmp_path) -> None:
    with pytest.raises(KeyError, match="Unknown scene"):
        download_gsplat_scene("not-a-scene", cache_dir=tmp_path)


def test_scene_preset_registries_are_consistent() -> None:
    names = gsplat_scene_names()
    assert set(names) == set(GSPLAT_SCENES)
    # Every curated skybox scene is a real preset.
    assert set(gsplat_skybox_scene_names()) <= set(names)
    # Curated alignment comes back as a mutable copy (callers **kwargs it).
    align = gsplat_skybox_align_for("tabletop")
    assert align["up_axis"] == (0.0, 1.0, 0.0)
    align["up_axis"] = None
    assert gsplat_skybox_align_for("tabletop")["up_axis"] == (0.0, 1.0, 0.0)
    # Uncurated scenes get an empty dict (best-effort auto alignment).
    assert gsplat_skybox_align_for("someone_uploaded.ply") == {}
