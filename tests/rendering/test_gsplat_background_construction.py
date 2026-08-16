# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construction + lazy-load contracts for ``GsplatBackground`` (zero ML deps).

``GsplatBackground`` is the CUDA-only 3D Gaussian Splatting backdrop, but its
constructor is deliberately dependency-free: the ``gsplat``/``torch``/scene
imports and the ``.ply``/``.spz`` read are all deferred to the first
:meth:`~GsplatBackground.render` (via ``_load``). That lazy design lets a caller
construct and configure a backdrop on a CPU-only host (or in CI without the
``sim-gs`` extra) and only pay the heavy dependency at render time. These tests
pin that contract plus the documented constructor semantics (mode gating on an
explicit transform, per-mode ``bg_fill`` default, the public ``own_floor`` flag)
and the fail-loud behaviour when the extra is genuinely absent.
"""

import importlib

import numpy as np
import pytest

from strands_robots.rendering import GsplatBackground, bake_gsplat_panorama


def _gsplat_importable() -> bool:
    try:
        importlib.import_module("gsplat")
    except ImportError:
        return False
    return True


def test_construction_does_not_require_gsplat_or_touch_the_scene_file(tmp_path) -> None:
    # A non-existent .ply path must not raise at construction time: the file is
    # only read in _load() on first render. This is the lazy-load contract that
    # keeps configuration cheap and CPU-safe.
    missing = tmp_path / "does_not_exist.ply"
    bg = GsplatBackground(ply_path=missing)

    assert bg.name == "gsplat"
    assert bg.own_floor is False
    # Nothing loaded yet - splats stay lazy until the first render.
    assert bg._splats is None


def test_own_floor_flag_is_public_and_round_trips() -> None:
    # The compositor reads ``own_floor`` to decide whether to hide the MuJoCo
    # grid ground, so it is part of the public surface.
    assert GsplatBackground("scene.ply", own_floor=True).own_floor is True
    assert GsplatBackground("scene.ply", own_floor=False).own_floor is False


def test_explicit_transform_disables_skybox_and_backdrop_auto_alignment() -> None:
    # Documented contract: skybox / auto_backdrop only fit a world_from_gs when
    # no explicit transform is supplied. Passing one pins the pose verbatim.
    world_from_gs = np.diag([2.0, 2.0, 2.0, 1.0])
    bg = GsplatBackground(
        "scene.ply",
        transform=world_from_gs,
        skybox=True,
        auto_backdrop=True,
    )

    assert bg._explicit_transform is True
    assert bg._skybox is False
    assert bg._auto_backdrop is False
    np.testing.assert_allclose(bg._transform, world_from_gs)


def test_skybox_and_backdrop_modes_activate_without_an_explicit_transform() -> None:
    skybox = GsplatBackground("scene.ply", skybox=True)
    assert skybox._skybox is True

    backdrop = GsplatBackground("scene.ply", auto_backdrop=True)
    assert backdrop._auto_backdrop is True


def test_bg_fill_defaults_to_neutral_grey_in_skybox_and_black_otherwise() -> None:
    # Skybox voids read as a light-grey ceiling/sky; the plain backdrop leaves
    # uncovered pixels black. An explicit bg_fill overrides either default.
    np.testing.assert_array_equal(
        GsplatBackground("scene.ply", skybox=True)._bg_fill,
        np.array([188.0, 188.0, 192.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        GsplatBackground("scene.ply")._bg_fill,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        GsplatBackground("scene.ply", bg_fill=(10, 20, 30))._bg_fill,
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )


def test_explicit_up_and_major_axis_are_stored_as_tuples() -> None:
    bg = GsplatBackground("scene.ply", up_axis=(0.0, 0.0, 1.0), major_axis=(1.0, 0.0, 0.0))
    assert bg._up_axis == (0.0, 0.0, 1.0)
    assert bg._major_axis == (1.0, 0.0, 0.0)

    pca = GsplatBackground("scene.ply")
    assert pca._up_axis is None
    assert pca._major_axis is None


def test_render_without_the_sim_gs_extra_fails_loud_with_an_install_hint() -> None:
    # First render triggers _load(), which require_optional()s gsplat. When the
    # extra is absent the caller must get an actionable ImportError, not a
    # cryptic failure deep inside rasterization.
    if _gsplat_importable():
        pytest.skip("gsplat is installed; cannot exercise the missing-extra path")

    bg = GsplatBackground("scene.ply", device="cpu")
    with pytest.raises(ImportError, match="sim-gs"):
        bg._load()


def test_bake_panorama_returns_cached_image_without_loading_splats(tmp_path) -> None:
    # The bake helper is CUDA-heavy, but a warm cache hit (an existing non-empty
    # panorama next to the .ply for the *same* requested geometry) must
    # short-circuit before any gsplat load. The default path encodes the
    # geometry, so the cache file is named for the request it answers.
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"placeholder")
    cached = ply.with_name("scene_pano_64x32_f16.jpg")
    cached.write_bytes(b"\xff\xd8\xff\xe0cached-jpeg")

    out = bake_gsplat_panorama(ply, face_size=16, equi_w=64, equi_h=32)
    assert out == cached
