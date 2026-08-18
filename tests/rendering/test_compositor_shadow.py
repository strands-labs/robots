# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shadow-catcher + linear-blend compositor tests (issue #2323, stages 2-3).

Pure numpy against fake frame sources, same pattern as ``test_compositor.py``:
the analytic plane depth (:func:`plane_depth`) is checked as geometry, and the
shadow pass is checked as behavior -- the catcher plane is never painted, its
shading darkens the background multiplicatively in linear light, and the
``CompositeFrame`` contract is unchanged.
"""

from typing import Any

import numpy as np
import pytest

from strands_robots.rendering import (
    CameraParams,
    HybridCompositor,
    linear_to_srgb,
    plane_depth,
    relative_luminance,
    srgb_to_linear,
)

W, H = 16, 12
ZFAR = 100.0
CAM_HEIGHT = 2.0

BG_COLOR = (200, 150, 100)
PLANE_GRAY = 200
SHADOW_GRAY = 130


def _down_cam(width: int = W, height: int = H) -> CameraParams:
    """Camera at (0, 0, CAM_HEIGHT) looking straight down at the z=0 plane.

    Identity rotation in the OpenGL convention means forward = -Z, i.e.
    world-down -- so the z=0 plane sits at a constant z-depth CAM_HEIGHT.
    """
    T = np.eye(4)
    T[2, 3] = CAM_HEIGHT
    fy = 0.5 * height / np.tan(np.deg2rad(45.0) / 2.0)
    K = np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
    return CameraParams(K=K, T_world_cam=T, width=width, height=height, znear=0.01, zfar=ZFAR)


class PlaneSceneSim:
    """Frame source: a robot square over a full-frame catcher plane.

    The plane fills the frame at the analytic z=0 depth (CAM_HEIGHT); the
    robot square sits 1 m above it; the plane carries a darker patch -- the
    shadow the simulator rendered onto it.
    """

    def __init__(self, cam: CameraParams | None = None):
        self.cam = cam or _down_cam()

    def get_camera_params(self, camera_name="default", width=None, height=None):
        return self.cam

    def get_frame(self, camera_name="default", width=None, height=None):
        rgb = np.full((H, W, 3), PLANE_GRAY, dtype=np.uint8)
        depth = np.full((H, W), CAM_HEIGHT, dtype=np.float32)
        rgb[8:11, 2:6] = SHADOW_GRAY  # the cast shadow on the plane
        rgb[4:8, 6:10] = 255  # the robot
        depth[4:8, 6:10] = CAM_HEIGHT - 1.0
        return rgb, depth


class FlatBackground:
    """Solid-color background whose geometry is the z=0 surface."""

    name = "flat-bg"

    def __init__(self, color=BG_COLOR, depth=CAM_HEIGHT):
        self.color = color
        self.depth = depth

    def render(self, cam: CameraParams):
        rgb = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        rgb[..., :] = self.color
        return rgb, np.full((cam.height, cam.width), self.depth, dtype=np.float32)


# --------------------------------------------------------------------------- #
# plane_depth: the analytic geometry the catcher detection rests on
# --------------------------------------------------------------------------- #


class TestPlaneDepth:
    def test_nadir_camera_sees_a_constant_z_depth(self) -> None:
        # z-depth of a horizontal plane under a straight-down camera is the
        # camera height for every pixel (z-depth, not ray length).
        d = plane_depth(_down_cam(), 0.0)
        assert d.shape == (H, W)
        np.testing.assert_allclose(d, CAM_HEIGHT, atol=1e-6)

    def test_plane_height_shifts_the_depth(self) -> None:
        d = plane_depth(_down_cam(), 0.5)
        np.testing.assert_allclose(d, CAM_HEIGHT - 0.5, atol=1e-6)

    def test_plane_behind_the_camera_is_inf(self) -> None:
        d = plane_depth(_down_cam(), CAM_HEIGHT + 1.0)
        assert np.isinf(d).all()

    def test_horizontal_camera_hits_below_the_horizon_only(self) -> None:
        # Camera 1 m up, looking along +X: rays above the optical center
        # never meet the floor; a ray y' below center meets it at z-depth
        # 1/y' (similar triangles: drop 1 m over y' per unit depth).
        fwd, up = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        u = np.cross(right, fwd)
        T = np.eye(4)
        T[:3, :3] = np.stack([right, u, -fwd], axis=1)
        T[:3, 3] = [0.0, 0.0, 1.0]
        fy = 10.0
        K = np.array([[fy, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
        cam = CameraParams(K=K, T_world_cam=T, width=W, height=H, znear=0.01, zfar=ZFAR)

        d = plane_depth(cam, 0.0)
        assert np.isinf(d[0, :]).all()  # above the horizon
        # Pixel rows below the principal point: y' = (v - cy) / fy.
        v = H - 1
        expected = 1.0 / ((v - H / 2.0) / fy)
        np.testing.assert_allclose(d[v, :], expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# the shadow-catcher pass
# --------------------------------------------------------------------------- #


def _shadow_compositor(**kwargs: "Any") -> HybridCompositor:
    defaults: dict[str, Any] = dict(background=FlatBackground(), feather_pixels=0, shadow_plane_z=0.0)
    defaults.update(kwargs)
    return HybridCompositor(PlaneSceneSim(), **defaults)


class TestShadowCatcher:
    def test_catcher_plane_is_never_painted(self) -> None:
        frame = _shadow_compositor().render("cam")
        # Unshadowed plane pixels show the background color -- not the
        # plane's own gray -- and are not foreground.
        assert not frame.foreground_mask[0, 0]
        assert tuple(frame.rgb[0, 0]) == BG_COLOR
        # The robot still wins exactly as before.
        assert frame.foreground_mask[5, 7]
        assert tuple(frame.rgb[5, 7]) == (255, 255, 255)

    def test_shadow_darkens_the_background_in_linear_light(self) -> None:
        frame = _shadow_compositor().render("cam")
        shadowed = frame.rgb[9, 3]
        unshadowed = frame.rgb[0, 0]
        assert (shadowed < unshadowed).all(), "the caught shadow must darken the backdrop"
        # The darkening is the plane's own shading ratio, applied to linear
        # light: factor = lum(shadow gray) / lum(plane gray).
        factor = relative_luminance(srgb_to_linear(np.full(3, SHADOW_GRAY, np.uint8))) / relative_luminance(
            srgb_to_linear(np.full(3, PLANE_GRAY, np.uint8))
        )
        expected_lin = srgb_to_linear(np.array(BG_COLOR, np.uint8)) * factor
        expected = np.clip(linear_to_srgb(expected_lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(shadowed, expected)

    def test_shadow_never_goes_below_the_floor_factor(self) -> None:
        frame = _shadow_compositor(shadow_min_factor=0.9).render("cam")
        expected_lin = srgb_to_linear(np.array(BG_COLOR, np.uint8)) * 0.9
        expected = np.clip(linear_to_srgb(expected_lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(frame.rgb[9, 3], expected)

    def test_background_rgb_debug_layer_carries_the_shadow(self) -> None:
        frame = _shadow_compositor().render("cam")
        assert (frame.background_rgb[9, 3] < frame.background_rgb[0, 0]).all()

    def test_depth_off_the_plane_is_not_read_as_catcher(self) -> None:
        # Geometry 0.5 m above the plane is ordinary foreground: nearer than
        # the background surface, so it wins the composite instead of
        # darkening it.
        class OffPlaneSim(PlaneSceneSim):
            def get_frame(self, camera_name="default", width=None, height=None):
                rgb, depth = super().get_frame(camera_name, width, height)
                depth = depth.copy()
                depth[0, 0] = CAM_HEIGHT - 0.5
                return rgb, depth

        comp = HybridCompositor(OffPlaneSim(), background=FlatBackground(), feather_pixels=0, shadow_plane_z=0.0)
        frame = comp.render("cam")
        assert frame.foreground_mask[0, 0]
        assert tuple(frame.rgb[0, 0]) == (PLANE_GRAY,) * 3

    def test_disabled_by_default_plane_zfights_as_before(self) -> None:
        # Without shadow_plane_z the catcher pixels are ordinary foreground
        # at the same depth as the background: they lose the (biased) depth
        # test, so the shadow is simply lost. That is the pre-#2323 behavior
        # this feature exists to replace -- pinned so the default stays put.
        comp = HybridCompositor(PlaneSceneSim(), background=FlatBackground(), feather_pixels=0)
        frame = comp.render("cam")
        assert tuple(frame.rgb[9, 3]) == BG_COLOR  # shadow gone
        assert not frame.foreground_mask[9, 3]

    def test_all_robot_frame_with_no_catcher_pixels_is_untouched(self) -> None:
        class RobotOnlySim(PlaneSceneSim):
            def get_frame(self, camera_name="default", width=None, height=None):
                rgb = np.full((H, W, 3), 255, dtype=np.uint8)
                return rgb, np.full((H, W), 1.0, dtype=np.float32)

        comp = HybridCompositor(RobotOnlySim(), background=FlatBackground(), feather_pixels=0, shadow_plane_z=0.0)
        frame = comp.render("cam")
        assert frame.foreground_mask.all()
        assert (frame.rgb == 255).all()


# --------------------------------------------------------------------------- #
# blend_in_linear (stage 3)
# --------------------------------------------------------------------------- #


class TestBlendInLinear:
    def test_interior_pixels_are_byte_identical_either_way(self) -> None:
        byte_frame = _shadow_compositor(feather_pixels=1, blend_in_linear=False).render("cam")
        lin_frame = _shadow_compositor(feather_pixels=1, blend_in_linear=True).render("cam")
        # Fully-foreground and fully-background pixels round-trip exactly.
        np.testing.assert_array_equal(lin_frame.rgb[5, 7], byte_frame.rgb[5, 7])
        np.testing.assert_array_equal(lin_frame.rgb[0, 0], byte_frame.rgb[0, 0])

    def test_seam_pixels_blend_brighter_in_linear_light(self) -> None:
        # Averaging gamma-encoded bytes under-weights the brighter layer;
        # linear blending does not, so a white-over-color seam pixel comes
        # out strictly brighter. Scene: a white square over sky, so the
        # feathered ring outside the mask blends white against the backdrop.
        class SquareOverSkySim(PlaneSceneSim):
            def get_frame(self, camera_name="default", width=None, height=None):
                rgb = np.full((H, W, 3), 255, dtype=np.uint8)
                depth = np.full((H, W), ZFAR, dtype=np.float32)
                depth[4:8, 6:10] = 1.0
                return rgb, depth

        def render(blend_in_linear: bool):
            comp = HybridCompositor(
                SquareOverSkySim(),
                background=FlatBackground(),
                feather_pixels=1,
                blend_in_linear=blend_in_linear,
            )
            return comp.render("cam")

        byte_frame, lin_frame = render(False), render(True)
        seam = (~byte_frame.foreground_mask) & (byte_frame.rgb.max(axis=-1) > np.array(BG_COLOR).max())
        assert seam.any(), "expected feathered seam pixels around the robot square"
        assert (lin_frame.rgb[seam].astype(int) >= byte_frame.rgb[seam].astype(int)).all()
        assert (lin_frame.rgb[seam].astype(int) > byte_frame.rgb[seam].astype(int)).any()


# --------------------------------------------------------------------------- #
# option guards
# --------------------------------------------------------------------------- #


class TestOptionGuards:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "floor"])
    def test_unusable_shadow_plane_z_is_refused(self, value) -> None:
        with pytest.raises(ValueError, match="shadow_plane_z"):
            _shadow_compositor(shadow_plane_z=value)

    @pytest.mark.parametrize("value", [0, -0.01, float("nan"), float("inf"), True, "close"])
    def test_unusable_shadow_plane_tolerance_is_refused(self, value) -> None:
        with pytest.raises(ValueError, match="shadow_plane_tolerance"):
            _shadow_compositor(shadow_plane_tolerance=value)

    @pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), True, "dark"])
    def test_unusable_shadow_min_factor_is_refused(self, value) -> None:
        with pytest.raises(ValueError, match="shadow_min_factor"):
            _shadow_compositor(shadow_min_factor=value)

    @pytest.mark.parametrize("value", [1, 0, "yes", None])
    def test_non_bool_blend_in_linear_is_refused(self, value) -> None:
        with pytest.raises(ValueError, match="blend_in_linear"):
            _shadow_compositor(blend_in_linear=value)

    def test_shadow_options_are_inert_while_the_feature_is_off(self) -> None:
        # shadow_plane_z=None disables the pass outright; the other shadow
        # knobs are still validated (a bad value must fail at construction,
        # not on the day the feature is switched on).
        comp = HybridCompositor(PlaneSceneSim(), background=FlatBackground(), feather_pixels=0)
        assert comp.shadow_plane_z is None
