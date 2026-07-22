# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HybridCompositor contract tests against a fake (pure-numpy) frame source.

The compositor is pure numpy over ``get_frame`` / ``get_camera_params``
(issue #1537), so its per-pixel z-compare, background caching, and
no-depth-backend refusal are all pinned here without any GL / sim deps.
"""

import numpy as np
import pytest

from strands_robots.rendering import CameraParams, HybridCompositor, feather_mask

W, H = 16, 12
ZFAR = 100.0


class FakeBackground:
    """Solid-color background at infinity, counting render calls."""

    name = "fake-bg"

    def __init__(self, color=(0, 0, 255)):
        self.color = color
        self.render_calls = 0

    def render(self, cam: CameraParams):
        self.render_calls += 1
        rgb = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        rgb[..., :] = self.color
        depth = np.full((cam.height, cam.width), cam.zfar, dtype=np.float32)
        return rgb, depth


class FakeSim:
    """Frame source with a centered square of foreground geometry."""

    def __init__(self, depth: "np.ndarray | None"):
        self._depth = depth

    def get_camera_params(self, camera_name="default", width=None, height=None):
        w, h = int(width or W), int(height or H)
        fy = 0.5 * h / np.tan(np.deg2rad(45.0) / 2.0)
        K = np.array([[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])
        return CameraParams(K=K, T_world_cam=np.eye(4), width=w, height=h, znear=0.01, zfar=ZFAR)

    def get_frame(self, camera_name="default", width=None, height=None):
        rgb = np.full((H, W, 3), 255, dtype=np.uint8)  # white robot pixels
        return rgb, self._depth


def _square_depth() -> np.ndarray:
    """Foreground geometry (1 m) in a centered square; sky (zfar) elsewhere."""
    depth = np.full((H, W), ZFAR, dtype=np.float32)
    depth[4:8, 6:10] = 1.0
    return depth


def test_foreground_wins_where_geometry_background_elsewhere() -> None:
    sim = FakeSim(_square_depth())
    comp = HybridCompositor(sim, background=FakeBackground(), feather_pixels=0)
    frame = comp.render("cam")
    # Geometry pixels show the (white) foreground.
    assert frame.foreground_mask[5, 7]
    assert tuple(frame.rgb[5, 7]) == (255, 255, 255)
    # Sky pixels (depth pinned to zfar) show the (blue) background.
    assert not frame.foreground_mask[0, 0]
    assert tuple(frame.rgb[0, 0]) == (0, 0, 255)


def test_isaac_style_zero_and_inf_depth_reads_as_background() -> None:
    depth = _square_depth()
    depth[0, 0] = 0.0  # Isaac no-hit convention
    depth[0, 1] = np.inf
    depth[0, 2] = np.nan
    comp = HybridCompositor(FakeSim(depth), background=FakeBackground(), feather_pixels=0)
    frame = comp.render("cam")
    for x in (0, 1, 2):
        assert not frame.foreground_mask[0, x]
        assert tuple(frame.rgb[0, x]) == (0, 0, 255)


def test_missing_depth_raises_instead_of_silent_wrong_output() -> None:
    comp = HybridCompositor(FakeSim(depth=None), background=FakeBackground())
    with pytest.raises(RuntimeError, match="depth"):
        comp.render("cam")


def test_background_cached_per_camera_pose() -> None:
    bg = FakeBackground()
    comp = HybridCompositor(FakeSim(_square_depth()), background=bg, feather_pixels=0)
    comp.render("cam")
    comp.render("cam")
    comp.render("cam")
    assert bg.render_calls == 1  # static camera: one background pass


def test_set_background_clears_cache_and_takes_effect() -> None:
    bg1 = FakeBackground(color=(0, 0, 255))
    bg2 = FakeBackground(color=(0, 255, 0))
    comp = HybridCompositor(FakeSim(_square_depth()), background=bg1, feather_pixels=0)
    comp.render("cam")
    comp.set_background(bg2)
    frame = comp.render("cam")
    assert tuple(frame.rgb[0, 0]) == (0, 255, 0)
    assert bg2.render_calls == 1


def test_composite_frame_carries_debug_layers() -> None:
    comp = HybridCompositor(FakeSim(_square_depth()), background=FakeBackground(), feather_pixels=0)
    frame = comp.render("cam")
    assert frame.rgb.shape == (H, W, 3)
    assert frame.foreground_rgb.shape == (H, W, 3)
    assert frame.background_rgb.shape == (H, W, 3)
    assert frame.depth.shape == (H, W)
    assert frame.depth.dtype == np.float32
    assert frame.camera.width == W and frame.camera.height == H


def test_feather_mask_softens_boundary_only() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    alpha = feather_mask(mask, radius=1)
    assert alpha.dtype == np.float32
    assert alpha[5, 5] == pytest.approx(1.0)  # interior untouched
    assert alpha[0, 0] == pytest.approx(0.0)  # far exterior untouched
    assert 0.0 < alpha[3, 3] < 1.0  # boundary softened
    # radius=0 degrades to the raw mask.
    assert np.array_equal(feather_mask(mask, radius=0), mask.astype(np.float32))
