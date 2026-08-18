# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment-map bake + key-light derivation tests (issue #2323, stage 1).

The bake's only GPU-bound step is ``background.render``; everything else --
the six cube-face camera poses, the equirect reprojection, the dominant-light
estimate -- is pure numpy and is pinned here against a fake background that
paints each face with a color identifying the world direction it was rendered
toward.
"""

import numpy as np
import pytest

from strands_robots.rendering import (
    bake_environment_map,
    derive_key_light,
    environment_map_cache_path,
    render_environment_map,
)


def _face_color(fwd) -> np.ndarray:
    """Map an outward face direction to a distinct RGB so an equirect pixel's
    color identifies which world direction it sampled."""
    return ((np.asarray(fwd, dtype=float) + 1.0) * 0.5 * 255.0).astype(np.uint8)


class DirectionPaintingBackground:
    """Fake background: paints each requested face with its direction color
    and records every camera it was asked to render."""

    name = "direction-paint"

    def __init__(self):
        self.cams = []

    def render(self, cam):
        self.cams.append(cam)
        # The bake builds T_world_cam columns [right, up, -fwd], so the
        # outward face direction is -Z of the camera rotation.
        fwd = -np.asarray(cam.T_world_cam, dtype=float)[:3, 2]
        rgb = np.empty((cam.height, cam.width, 3), np.uint8)
        rgb[:] = _face_color(fwd)
        return rgb, np.zeros((cam.height, cam.width), np.float32)


class TestRenderEnvironmentMap:
    def test_reprojects_cube_faces_into_equirect_directions(self) -> None:
        bg = DirectionPaintingBackground()
        env = render_environment_map(bg, origin_world=(0.0, 0.0, 0.0), face_size=32, equi_w=64, equi_h=32)

        assert env.shape == (32, 64, 3)
        assert env.dtype == np.uint8
        # Equirect convention (matches PanoramaBackground): column W/2 ->
        # theta 0 and row H/2 -> phi 0 -> +X; column 0 -> theta -pi -> -X;
        # row 0 -> phi +pi/2 -> +Z.
        np.testing.assert_array_equal(env[16, 32], _face_color([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(env[16, 0], _face_color([-1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(env[0, 32], _face_color([0.0, 0.0, 1.0]))

    def test_bakes_from_the_requested_world_origin(self) -> None:
        # The whole point over bake_gsplat_panorama: the cameras sit at the
        # robot's position in the world frame, so an aligned background
        # renders the environment the robot actually stands in.
        bg = DirectionPaintingBackground()
        origin = (0.25, -0.5, 0.4)
        render_environment_map(bg, origin_world=origin, face_size=16, equi_w=32, equi_h=16)

        assert len(bg.cams) == 6
        for cam in bg.cams:
            np.testing.assert_allclose(cam.T_world_cam[:3, 3], origin)

    def test_faces_are_90_degree_pinhole_cameras(self) -> None:
        bg = DirectionPaintingBackground()
        render_environment_map(bg, origin_world=(0.0, 0.0, 0.0), face_size=32, equi_w=32, equi_h=16)

        forwards = []
        for cam in bg.cams:
            assert (cam.width, cam.height) == (32, 32)
            # 90 deg FOV -> focal = size / 2; principal point centered.
            assert cam.K[0, 0] == pytest.approx(16.0)
            assert cam.K[1, 1] == pytest.approx(16.0)
            assert cam.K[0, 2] == pytest.approx(16.0)
            R = np.asarray(cam.T_world_cam)[:3, :3]
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            forwards.append(tuple(np.round(-R[:, 2]).astype(int)))
        # The six outward directions cover the axes exactly once each.
        assert sorted(forwards) == sorted([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])


class TestBakeEnvironmentMap:
    def test_writes_the_map_and_returns_the_path(self, tmp_path) -> None:
        pytest.importorskip("PIL")
        from PIL import Image

        out = tmp_path / "env.png"
        result = bake_environment_map(
            DirectionPaintingBackground(), out, origin_world=(0.0, 0.0, 0.0), face_size=16, equi_w=32, equi_h=16
        )

        assert result == out
        img = np.array(Image.open(out))
        assert img.shape == (16, 32, 3)
        np.testing.assert_array_equal(img[8, 16], _face_color([1.0, 0.0, 0.0]))

    def test_non_empty_output_short_circuits_the_bake(self, tmp_path) -> None:
        out = tmp_path / "env.png"
        out.write_bytes(b"cached-environment-map")

        class Boom:
            name = "boom"

            def render(self, cam):
                raise AssertionError("bake must not render when a cached map exists")

        result = bake_environment_map(Boom(), out, origin_world=(0.0, 0.0, 0.0))

        assert result == out
        assert out.read_bytes() == b"cached-environment-map"


class TestEnvironmentMapCachePath:
    def test_encodes_every_input_that_changes_the_pixels(self, tmp_path) -> None:
        # A warm file short-circuits the bake, so the default name must
        # identify the map it holds: origin and every resolution knob.
        scene = tmp_path / "scene.ply"
        base = environment_map_cache_path(scene, origin_world=(0.0, 0.0, 0.4))
        assert base.parent == tmp_path
        assert base.suffix == ".png"
        variants = [
            environment_map_cache_path(scene, origin_world=(0.0, 0.0, 0.5)),
            environment_map_cache_path(scene, origin_world=(0.0, 0.0, 0.4), face_size=256),
            environment_map_cache_path(scene, origin_world=(0.0, 0.0, 0.4), equi_w=1024),
            environment_map_cache_path(scene, origin_world=(0.0, 0.0, 0.4), equi_h=512),
        ]
        names = {base.name, *[v.name for v in variants]}
        assert len(names) == 5, f"cache names must differ per input, got {names}"
        # Same inputs -> same name (it IS a cache key).
        assert environment_map_cache_path(scene, origin_world=(0.0, 0.0, 0.4)) == base


class TestDeriveKeyLight:
    W, H = 128, 64

    def _env_with_blob(self, row: int, col: int, color=(255, 255, 255)) -> np.ndarray:
        env = np.zeros((self.H, self.W, 3), np.uint8)
        env[row - 2 : row + 3, col - 2 : col + 3] = color
        return env

    def test_direction_points_at_the_bright_blob(self) -> None:
        # Blob at theta=0, phi=+45 deg -> direction (cos45, 0, sin45).
        env = self._env_with_blob(row=16, col=64)
        est = derive_key_light(env)
        np.testing.assert_allclose(est.direction, [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4)], atol=0.05)
        assert est.azimuth_deg == pytest.approx(0.0, abs=3.0)
        assert est.elevation_deg == pytest.approx(45.0, abs=3.0)

    def test_direction_follows_the_blob_around_the_sphere(self) -> None:
        # Blob at theta=+90 deg (three-quarter column), horizon row -> +Y.
        # The horizon straddles both hemispheres, so search the full sphere.
        env = self._env_with_blob(row=32, col=96)
        est = derive_key_light(env, upper_hemisphere=False)
        np.testing.assert_allclose(est.direction, [0.0, 1.0, 0.0], atol=0.06)

    def test_bright_floor_does_not_hijack_the_key_light(self) -> None:
        # A bright surface *under* the bake point (the tabletop preset's
        # counter) must not aim the key light from underneath: below-horizon
        # radiance is bounce, which the dome texture already provides.
        env = self._env_with_blob(row=16, col=64, color=(200, 200, 200))
        env[48:, :] = 255  # blazing floor, brighter than the actual light
        est = derive_key_light(env)
        assert est.elevation_deg > 0, "key light must come from above the horizon"
        assert est.azimuth_deg == pytest.approx(0.0, abs=3.0)
        # Opting into the full sphere finds the floor instead -- the caller
        # asked for it.
        full = derive_key_light(env, upper_hemisphere=False)
        assert full.elevation_deg < 0

    def test_color_is_the_blob_chromaticity_normalized(self) -> None:
        env = self._env_with_blob(row=16, col=64, color=(255, 128, 0))
        est = derive_key_light(env)
        assert est.color[0] == pytest.approx(1.0, abs=1e-6)
        # linear(128/255) / linear(1.0) ~= 0.2158 -- chromaticity is linear
        # light, not a copy of the sRGB bytes.
        assert est.color[1] == pytest.approx(0.2158, abs=0.01)
        assert est.color[2] == pytest.approx(0.0, abs=1e-6)

    def test_uniform_map_has_no_dominant_direction_and_says_so(self) -> None:
        # On the full sphere a uniform map's directions cancel exactly; on
        # the (default) upper hemisphere the only honest answer is the
        # zenith.
        env = np.full((32, 64, 3), 128, np.uint8)
        with pytest.raises(ValueError, match="no dominant direction"):
            derive_key_light(env, upper_hemisphere=False)
        assert derive_key_light(env).elevation_deg == pytest.approx(90.0, abs=1.0)

    def test_black_map_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no light"):
            derive_key_light(np.zeros((16, 32, 3), np.uint8))

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5, float("nan"), float("inf"), True, "many"])
    def test_unusable_brightest_fraction_is_refused(self, fraction) -> None:
        env = self._env_with_blob(row=16, col=64)
        with pytest.raises(ValueError, match="brightest_fraction"):
            derive_key_light(env, brightest_fraction=fraction)

    @pytest.mark.parametrize("shape", [(16, 32), (16, 32, 4), (3,)])
    def test_non_image_env_map_is_refused(self, shape) -> None:
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            derive_key_light(np.zeros(shape, np.uint8))

    def test_an_env_map_whose_scale_is_ambiguous_is_refused_not_read_as_another_light(self) -> None:
        """The map's *shape* is checked; its *scale* reaches the decode untouched.

        A map whose dtype carries no scale (a 16-bit capture, or an array a
        caller widened) decoded as unit light saturates every non-black texel,
        so the estimate stops being the brightest region's direction and
        becomes the solid-angle centroid of everything that is not pure black --
        a different light, reported as success.
        """
        env = self._env_with_blob(row=16, col=64, color=(220, 200, 40))
        env[8:12, 100:112] = 60  # a dimmer source elsewhere, which must not win
        reference = derive_key_light(env)
        assert reference.color[2] < 0.5, "premise: the blob is not white, so its chromaticity is visible"

        try:
            got = derive_key_light(env.astype(np.uint16))
        except ValueError as exc:
            assert "uint8" in str(exc)
            return

        raise AssertionError(
            f"the same map as uint16 derived azimuth={got.azimuth_deg:.1f} deg "
            f"elevation={got.elevation_deg:.1f} deg color={tuple(round(c, 3) for c in got.color)}, "
            f"not azimuth={reference.azimuth_deg:.1f} deg "
            f"elevation={reference.elevation_deg:.1f} deg "
            f"color={tuple(round(c, 3) for c in reference.color)}."
        )
