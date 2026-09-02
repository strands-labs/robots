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
        """Return buffers at exactly the size asked for, as a real engine does.

        The MuJoCo backend renders at the requested dimensions and the Isaac
        one raises rather than return another size, so a double that ignored
        the request could only be consumed by a compositor that truncated its
        layers -- which
        ``tests/rendering/test_composite_layer_size_matches_the_camera.py``
        now refuses. At the default size this is the frame it always returned.
        """
        w, h = int(width or W), int(height or H)
        rgb = np.full((h, w, 3), 255, dtype=np.uint8)  # white robot pixels
        depth = self._depth
        if depth is None or depth.shape == (h, w):
            return rgb, depth
        # Re-window the caller's depth pattern into the requested frame; sky
        # (zfar) fills whatever the pattern does not cover.
        sized = np.full((h, w), ZFAR, dtype=np.float32)
        rows, cols = min(h, depth.shape[0]), min(w, depth.shape[1])
        sized[:rows, :cols] = depth[:rows, :cols]
        return rgb, sized


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


class FakeBackgroundRGBA(FakeBackground):
    """Background that returns a 4-channel (RGBA) render, as some renderers do.

    The compositor must drop the alpha channel down to RGB before compositing.
    """

    def render(self, cam: CameraParams):
        rgb, depth = super().render(cam)
        alpha = np.full((cam.height, cam.width, 1), 255, dtype=np.uint8)
        return np.concatenate([rgb, alpha], axis=2), depth


class FakeSimRGBA(FakeSim):
    """Frame source whose foreground render carries an alpha channel."""

    def get_frame(self, camera_name="default", width=None, height=None):
        rgb, depth = super().get_frame(camera_name, width, height)
        alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([rgb, alpha], axis=2), depth


def test_rgba_foreground_alpha_channel_is_dropped() -> None:
    comp = HybridCompositor(FakeSimRGBA(_square_depth()), background=FakeBackground(), feather_pixels=0)
    frame = comp.render("cam")
    # Alpha is stripped: debug + composite layers are 3-channel and the
    # geometry pixel still shows the (white) foreground colour.
    assert frame.foreground_rgb.shape == (H, W, 3)
    assert frame.rgb.shape == (H, W, 3)
    assert tuple(frame.rgb[5, 7]) == (255, 255, 255)


def test_rgba_background_alpha_channel_is_dropped() -> None:
    comp = HybridCompositor(FakeSim(_square_depth()), background=FakeBackgroundRGBA(), feather_pixels=0)
    frame = comp.render("cam")
    assert frame.background_rgb.shape == (H, W, 3)
    # Sky pixel shows the (blue) background with its alpha dropped.
    assert tuple(frame.rgb[0, 0]) == (0, 0, 255)


def test_render_with_feathering_blends_the_geometry_boundary() -> None:
    # feather_pixels > 0 routes the winner mask through feather_mask, softening
    # the foreground/background seam so edge pixels are a blend of both.
    comp = HybridCompositor(FakeSim(_square_depth()), background=FakeBackground(), feather_pixels=1)
    frame = comp.render("cam")
    fg, bg = (255, 255, 255), (0, 0, 255)
    blended = [tuple(frame.rgb[y, x]) for y in range(H) for x in range(W) if tuple(frame.rgb[y, x]) not in (fg, bg)]
    assert blended, "feathering should produce at least one blended edge pixel"


def test_background_cache_is_bounded_under_many_camera_poses() -> None:
    # Distinct camera names key distinct cache entries; past the cap the cache
    # is dropped so it cannot grow without bound during a long fly-through.
    bg = FakeBackground()
    comp = HybridCompositor(FakeSim(_square_depth()), background=bg, feather_pixels=0)
    for i in range(18):
        comp.render(f"cam{i}")
    assert bg.render_calls == 18  # each distinct pose rendered exactly once
    # The overflow cleared the cache, so an early camera is now a miss and
    # re-renders rather than being served stale.
    comp.render("cam0")
    assert bg.render_calls == 19


def test_clear_caches_forces_background_recompute() -> None:
    bg = FakeBackground()
    comp = HybridCompositor(FakeSim(_square_depth()), background=bg, feather_pixels=0)
    comp.render("cam")
    comp.render("cam")
    assert bg.render_calls == 1  # second render served from cache
    comp.clear_caches()
    comp.render("cam")
    assert bg.render_calls == 2  # cache dropped -> background recomputed


class FakeFiniteBackground(FakeBackground):
    """Background at a uniform *finite* depth, to exercise the z-compare bias.

    Every other fixture parks the background at ``zfar`` (100 m), so the
    foreground/background depths are always ~99 m apart and the 1 mm tie-break
    in :meth:`HybridCompositor.render` is never straddled. This background sits
    at a close, finite depth so a near-tie foreground can be placed either side
    of the bias.
    """

    def __init__(self, depth: float, color=(0, 0, 255)):
        super().__init__(color=color)
        self.depth = float(depth)

    def render(self, cam: CameraParams):
        rgb, _ = super().render(cam)
        depth = np.full((cam.height, cam.width), self.depth, dtype=np.float32)
        return rgb, depth


def _uniform_depth(value: float) -> np.ndarray:
    """A full-frame foreground at a single finite depth (valid geometry)."""
    return np.full((H, W), float(value), dtype=np.float32)


def test_foreground_within_depth_bias_loses_to_background() -> None:
    # The winner rule is ``fg_depth + 1e-3 < bg_depth``: the foreground must be
    # *more than* 1 mm in front to win. A foreground only 0.5 mm closer than the
    # background is inside that bias, so the background must still show -- this
    # is the anti-z-fighting guard that keeps a near-coincident gsplat backdrop
    # from flickering against sim geometry.
    bg_depth = 5.0
    comp = HybridCompositor(
        FakeSim(_uniform_depth(bg_depth - 0.0005)),  # 0.5 mm closer: within the 1 mm bias
        background=FakeFiniteBackground(bg_depth),
        feather_pixels=0,
    )
    frame = comp.render("cam")
    assert not frame.foreground_mask.any()
    assert tuple(frame.rgb[H // 2, W // 2]) == (0, 0, 255)  # background wins


def test_foreground_beyond_depth_bias_wins() -> None:
    # A foreground 2 mm closer than the background clears the 1 mm bias and
    # wins the z-compare everywhere.
    bg_depth = 5.0
    comp = HybridCompositor(
        FakeSim(_uniform_depth(bg_depth - 0.002)),  # 2 mm closer: beyond the 1 mm bias
        background=FakeFiniteBackground(bg_depth),
        feather_pixels=0,
    )
    frame = comp.render("cam")
    assert frame.foreground_mask.all()
    assert tuple(frame.rgb[H // 2, W // 2]) == (255, 255, 255)  # foreground wins
