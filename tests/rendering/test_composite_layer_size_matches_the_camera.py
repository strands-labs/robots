# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A composited frame is the size its own camera parameters describe.

``HybridCompositor.render`` returns a ``CompositeFrame`` carrying the
``CameraParams`` it resolved, and every consumer reads the image through them:
``camera.K`` places the principal point at ``(width / 2, height / 2)``, and
``camera.width`` / ``camera.height`` are what a video writer or a 3D->2D
projection is sized from.

The compositor used to align its layers by truncating to the shortest one
(``h = min(fg.shape[0], bg.shape[0])``), so a layer shorter than the camera
returned a composite at that shorter size while still reporting the camera the
caller asked for -- a 64x64 frame whose reported principal point (160, 120) lies
outside it, at a size nobody requested and with nothing saying so.

A short layer is reachable from either side, and both are extension points:

* ``BackgroundRenderer`` is a public :class:`typing.Protocol` that
  ``HybridCompositor(background=...)`` and :meth:`set_background` accept, so any
  third-party rasterizer that caps its output (a device/texture limit) supplies
  one.
* ``get_frame`` belongs to the engine. Its Isaac implementation already decided
  this exact question the other way for the foreground -- it raises on a size it
  cannot render "rather than silently dropping the requested size" -- naming the
  compositor as the consumer that decision protects.

So a layer at any other size is refused, naming the side, the buffer, the size
it returned and the size the camera declares.
"""

import numpy as np
import pytest

from strands_robots.rendering import CameraParams, HybridCompositor

W, H = 16, 12
ZFAR = 100.0


def _extent(cap: "int | None", capped: bool, height: int, width: int) -> tuple[int, int]:
    """The ``(rows, cols)`` a capped rasterizer can actually draw."""
    if cap is None or not capped:
        return height, width
    return min(cap, height), min(cap, width)


class SizedBackground:
    """Background whose rasterizer caps the frame it can draw.

    Honours ``cam.K`` -- its pixel ``(i, j)`` is the same world ray as the
    foreground's -- but cannot draw past ``cap`` pixels on an axis, the shape a
    real device or texture limit produces. ``cap=None`` conforms.
    """

    name = "sized-bg"

    def __init__(self, cap: "int | None" = None, buffers: tuple[str, ...] = ("rgb", "depth")) -> None:
        self.cap = cap
        self.buffers = buffers

    def render(self, cam: CameraParams) -> tuple[np.ndarray, np.ndarray]:
        rgb_h, rgb_w = _extent(self.cap, "rgb" in self.buffers, cam.height, cam.width)
        d_h, d_w = _extent(self.cap, "depth" in self.buffers, cam.height, cam.width)
        return (
            np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8),
            np.full((d_h, d_w), cam.zfar, dtype=np.float32),
        )


class SizedSim:
    """Frame source whose ``get_frame`` can under-deliver, like a capped RTX sensor."""

    def __init__(self, cap: "int | None" = None, buffers: tuple[str, ...] = ("rgb", "depth")) -> None:
        self.cap = cap
        self.buffers = buffers

    def get_camera_params(self, camera_name="default", width=None, height=None) -> CameraParams:
        w, h = int(width or W), int(height or H)
        fy = 0.5 * h / np.tan(np.deg2rad(45.0) / 2.0)
        K = np.array([[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])
        return CameraParams(K=K, T_world_cam=np.eye(4), width=w, height=h, znear=0.01, zfar=ZFAR)

    def get_frame(self, camera_name="default", width=None, height=None) -> tuple[np.ndarray, np.ndarray]:
        w, h = int(width or W), int(height or H)
        rgb_h, rgb_w = _extent(self.cap, "rgb" in self.buffers, h, w)
        d_h, d_w = _extent(self.cap, "depth" in self.buffers, h, w)
        rgb = np.full((rgb_h, rgb_w, 3), 255, dtype=np.uint8)
        depth = np.full((d_h, d_w), ZFAR, dtype=np.float32)
        depth[d_h // 3 : 2 * d_h // 3, d_w // 3 : 2 * d_w // 3] = 1.0  # foreground geometry
        return rgb, depth


class TestALayerAtAnotherSizeIsRefused:
    """The composite cannot be smaller than the camera it reports."""

    @pytest.mark.parametrize("buffers", [("rgb", "depth"), ("rgb",), ("depth",)])
    def test_a_short_background_is_refused(self, buffers: tuple[str, ...]) -> None:
        comp = HybridCompositor(SizedSim(), background=SizedBackground(cap=4, buffers=buffers))
        with pytest.raises(RuntimeError, match="background") as excinfo:
            comp.render("cam")
        text = str(excinfo.value)
        assert "4x4" in text, text
        assert f"{W}x{H}" in text, text
        assert "sized-bg" in text, text

    @pytest.mark.parametrize("buffers", [("rgb", "depth"), ("rgb",), ("depth",)])
    def test_a_short_foreground_is_refused(self, buffers: tuple[str, ...]) -> None:
        comp = HybridCompositor(SizedSim(cap=4, buffers=buffers), background=SizedBackground())
        with pytest.raises(RuntimeError, match="foreground") as excinfo:
            comp.render("cam")
        text = str(excinfo.value)
        assert "4x4" in text, text
        assert f"{W}x{H}" in text, text
        assert "get_frame" in text, text

    def test_the_refusal_names_the_principal_point_the_layer_cannot_serve(self) -> None:
        """The reason is geometric, so the message carries the point that moves."""
        comp = HybridCompositor(SizedSim(), background=SizedBackground(cap=4))
        with pytest.raises(RuntimeError, match=r"principal point is \(8\.0, 6\.0\)"):
            comp.render("cam")

    def test_a_taller_layer_is_still_a_disagreement(self) -> None:
        """The refusal is equality, not a floor: an over-sized layer is a bug too."""

        class OversizedBackground(SizedBackground):
            def render(self, cam: CameraParams) -> tuple[np.ndarray, np.ndarray]:
                return (
                    np.zeros((cam.height + 2, cam.width + 2, 3), dtype=np.uint8),
                    np.full((cam.height + 2, cam.width + 2), cam.zfar, dtype=np.float32),
                )

        comp = HybridCompositor(SizedSim(), background=OversizedBackground())
        with pytest.raises(RuntimeError, match="background rgb"):
            comp.render("cam")

    def test_the_shadow_catcher_pass_refuses_it_too(self) -> None:
        """The catcher path reads the layers as well, so it must not slip through."""
        comp = HybridCompositor(SizedSim(), background=SizedBackground(cap=4), shadow_plane_z=0.0, feather_pixels=0)
        with pytest.raises(RuntimeError, match="background"):
            comp.render("cam")


class TestAConformingCompositeIsUnchanged:
    """Controls: nothing that already agreed with the camera is refused."""

    def test_a_conforming_render_reports_the_size_it_wrote(self) -> None:
        comp = HybridCompositor(SizedSim(), background=SizedBackground(), feather_pixels=0)
        frame = comp.render("cam")
        assert frame.rgb.shape == (H, W, 3)
        assert frame.depth.shape == (H, W)
        assert frame.foreground_mask.shape == (H, W)
        assert (frame.camera.width, frame.camera.height) == (W, H)

    def test_an_explicit_size_is_honoured_end_to_end(self) -> None:
        """A per-call size resolves the camera, and every layer follows it."""
        comp = HybridCompositor(SizedSim(), background=SizedBackground(), feather_pixels=0)
        frame = comp.render("cam", width=24, height=20)
        assert frame.rgb.shape == (20, 24, 3)
        assert (frame.camera.width, frame.camera.height) == (24, 20)
        assert frame.camera.K[0, 2] == pytest.approx(12.0)

    def test_a_missing_depth_buffer_still_reports_its_own_reason(self) -> None:
        """The size check must not shadow the pre-existing no-depth refusal."""

        class DepthlessSim(SizedSim):
            def get_frame(self, camera_name="default", width=None, height=None):
                rgb, _ = super().get_frame(camera_name, width, height)
                return rgb, None

        comp = HybridCompositor(DepthlessSim(), background=SizedBackground())
        with pytest.raises(RuntimeError, match="no depth buffer"):
            comp.render("cam")
