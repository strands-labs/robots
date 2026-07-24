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


def _write_solid_panorama(path, color, width: int = 256, height: int = 128) -> None:
    """Write a solid-color equirectangular image the loader can read."""
    from PIL import Image

    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[:] = np.asarray(color, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_panorama_loads_equirectangular_image_from_disk(tmp_path) -> None:
    # A real image on disk takes the load path (not the procedural fallback):
    # a solid-color equirectangular panorama must render to that exact color
    # at every pixel, whatever direction each ray points.
    magenta = (255, 0, 255)
    img = tmp_path / "sky.png"
    _write_solid_panorama(img, magenta)
    rgb, _ = PanoramaBackground(image_path=img).render(_cam())
    # Bilinear resampling can floor a uniform texel by 1 LSB; allow that.
    assert np.all(np.abs(rgb.astype(int) - np.array(magenta, dtype=int)) <= 1)


def test_panorama_loads_image_once_and_caches(tmp_path) -> None:
    # The panorama is read from disk on first render and cached: editing the
    # file afterwards must not change subsequent renders (the loader is not
    # re-hit). Proves the cache-return path, observable as a stable render.
    img = tmp_path / "sky.png"
    _write_solid_panorama(img, (10, 200, 30))
    bg = PanoramaBackground(image_path=img)
    first, _ = bg.render(_cam())
    _write_solid_panorama(img, (200, 10, 30))  # overwrite on disk after load
    second, _ = bg.render(_cam())
    assert np.array_equal(first, second)
    assert np.all(np.abs(first.astype(int) - np.array((10, 200, 30), dtype=int)) <= 1)


def test_panorama_yaw_rotation_changes_the_background() -> None:
    # rotation_deg spins the panorama around world +Z, so a non-zero yaw must
    # remap texels to different pixels than the unrotated render. The
    # procedural panorama has azimuthal variation (the window-light lobe), so
    # a 90-degree yaw is observable.
    cam = _cam()
    unrotated, _ = PanoramaBackground(rotation_deg=0.0).render(cam)
    rotated, _ = PanoramaBackground(rotation_deg=90.0).render(cam)
    assert not np.array_equal(unrotated, rotated)


def test_gsplat_rasterizer_available_is_a_nonraising_capability_probe() -> None:
    # The probe must never raise -- callers rely on it to decide up front
    # whether to fall back to PanoramaBackground. It always returns
    # (ok: bool, reason: str) with a non-empty reason.
    from strands_robots.rendering.backgrounds import gsplat_rasterizer_available

    ok, reason = gsplat_rasterizer_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str) and reason
