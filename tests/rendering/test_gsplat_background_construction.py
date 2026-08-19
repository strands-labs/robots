# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construction + lazy-load contracts for ``GsplatBackground`` (zero ML deps).

``GsplatBackground`` is the CUDA-only 3D Gaussian Splatting backdrop, but its
constructor is deliberately dependency-free: the ``gsplat``/``torch``/scene
imports and the ``.ply``/``.spz`` decode are all deferred to the first
:meth:`~GsplatBackground.render` (via ``_load``). That lazy design lets a caller
construct and configure a backdrop on a CPU-only host (or in CI without the
``sim-gs`` extra) and only pay the heavy dependency at render time.

The scene *path*, however, is validated eagerly (issue #2321): a nonexistent
``ply_path`` raises ``FileNotFoundError`` at construction, where the caller
supplied it -- not at the first frame inside an app's catch-all, which is what
silently demoted the photoreal background to the procedural fallback. These
tests pin both halves of that contract plus the documented constructor
semantics (mode gating on an explicit transform, per-mode ``bg_fill`` default,
the public ``own_floor`` flag) and the fail-loud behaviour when the extra is
genuinely absent.
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


@pytest.fixture
def scene_ply(tmp_path):
    """An existing placeholder scene file (construction validates existence,
    but only ``_load`` -- deferred to first render -- reads the bytes)."""
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"placeholder")
    return ply


def test_missing_scene_file_raises_at_construction(tmp_path) -> None:
    # A wrong path is a configuration error and must surface where it happens
    # (issue #2321) -- not at first render() inside an app's catch-all, which
    # is how the photoreal path silently demoted to the procedural gradient.
    missing = tmp_path / "does_not_exist.ply"
    with pytest.raises(FileNotFoundError, match="Gaussian Splat not found"):
        GsplatBackground(ply_path=missing)


def test_construction_does_not_require_gsplat_or_read_the_scene_file(scene_ply) -> None:
    # With an existing file, construction stays dependency-free and lazy: no
    # gsplat/torch import, no decode -- splats load on the first render.
    bg = GsplatBackground(ply_path=scene_ply)

    assert bg.name == "gsplat"
    assert bg.own_floor is False
    # Nothing loaded yet - splats stay lazy until the first render.
    assert bg._splats is None


def test_own_floor_flag_is_public_and_round_trips(scene_ply) -> None:
    # The compositor reads ``own_floor`` to decide whether to hide the MuJoCo
    # grid ground, so it is part of the public surface.
    assert GsplatBackground(scene_ply, own_floor=True).own_floor is True
    assert GsplatBackground(scene_ply, own_floor=False).own_floor is False


def test_explicit_transform_disables_skybox_and_backdrop_auto_alignment(scene_ply) -> None:
    # Documented contract: skybox / auto_backdrop only fit a world_from_gs when
    # no explicit transform is supplied. Passing one pins the pose verbatim.
    world_from_gs = np.diag([2.0, 2.0, 2.0, 1.0])
    bg = GsplatBackground(
        scene_ply,
        transform=world_from_gs,
        skybox=True,
        auto_backdrop=True,
    )

    assert bg._explicit_transform is True
    assert bg._skybox is False
    assert bg._auto_backdrop is False
    np.testing.assert_allclose(bg._transform, world_from_gs)


def test_skybox_and_backdrop_modes_activate_without_an_explicit_transform(scene_ply) -> None:
    skybox = GsplatBackground(scene_ply, skybox=True)
    assert skybox._skybox is True

    backdrop = GsplatBackground(scene_ply, auto_backdrop=True)
    assert backdrop._auto_backdrop is True


def test_bg_fill_defaults_to_neutral_grey_in_skybox_and_black_otherwise(scene_ply) -> None:
    # Skybox voids read as a light-grey ceiling/sky; the plain backdrop leaves
    # uncovered pixels black. An explicit bg_fill overrides either default.
    np.testing.assert_array_equal(
        GsplatBackground(scene_ply, skybox=True)._bg_fill,
        np.array([188.0, 188.0, 192.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        GsplatBackground(scene_ply)._bg_fill,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        GsplatBackground(scene_ply, bg_fill=(10, 20, 30))._bg_fill,
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )


def test_explicit_up_and_major_axis_are_stored_as_tuples(scene_ply) -> None:
    bg = GsplatBackground(scene_ply, up_axis=(0.0, 0.0, 1.0), major_axis=(1.0, 0.0, 0.0))
    assert bg._up_axis == (0.0, 0.0, 1.0)
    assert bg._major_axis == (1.0, 0.0, 0.0)

    pca = GsplatBackground(scene_ply)
    assert pca._up_axis is None
    assert pca._major_axis is None


def test_render_without_the_sim_gs_extra_fails_loud_with_an_install_hint(scene_ply) -> None:
    # First render triggers _load(), which require_optional()s gsplat. When the
    # extra is absent the caller must get an actionable ImportError, not a
    # cryptic failure deep inside rasterization.
    if _gsplat_importable():
        pytest.skip("gsplat is installed; cannot exercise the missing-extra path")

    bg = GsplatBackground(scene_ply, device="cpu")
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
