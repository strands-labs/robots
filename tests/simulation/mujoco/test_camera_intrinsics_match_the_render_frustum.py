# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``get_camera_params`` intrinsics agree with the frustum MuJoCo rasterizes.

A camera declaring a physical sensor (MJCF ``sensorsize`` / ``focal`` /
``principal`` / ``resolution``) is drawn through MuJoCo's intrinsic model, and
``K`` has to describe that same projection or every unprojected pixel
(``get_world_point``, the hybrid compositor) lands somewhere the renderer never
drew.

The vertical half of that projection is not stable across the supported MuJoCo
range: 3.6.0 fixed swapped vertical frustum bounds for a camera with a
principal-point offset, so a positive MJCF ``principal`` y-offset moves the
principal point to opposite sides of the image center on 3.5 and on 3.6+.
These tests pin ``K`` against the projection itself rather than against either
convention: the rasterizer decides in the GL test, and the frustum -> ``K``
mapping is pinned for both conventions with a stub.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.rendering import RenderingMixin  # noqa: E402
from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402

# The review's repro camera: 6.4 x 4.8 mm sensor, 6 mm focal, principal point
# offset by (1.5, 0.8) mm, 320x240 pixels, znear 0.01 m. A sphere sits exactly
# on its optical axis, so the rasterizer draws it at the principal point.
_SCENE = """
<mujoco model="offset_principal_point">
  <statistic extent="2" center="0 0 0"/>
  <visual><map znear="0.005" zfar="25"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <body name="ball" pos="0 0 0.1">
      <geom type="sphere" size="0.01" rgba="1 0 0 1"/>
    </body>
    <camera name="poff" pos="0 -1 0.1" xyaxes="1 0 0 0 0 1"
            sensorsize="0.0064 0.0048" focal="0.006 0.006"
            resolution="320 240" principal="0.0015 0.0008"/>
  </worldbody>
</mujoco>
"""

# MuJoCo's own frustum for that camera at znear = 0.01 (engine_vis_visualize.c
# getFrustum): zhor = znear/focal_x * (sensor_w/2 -+ principal_x), zver =
# znear/focal_y * (sensor_h/2 -+ principal_y).
_Z_HOR = (0.01 / 0.006 * (0.0032 - 0.0015), 0.01 / 0.006 * (0.0032 + 0.0015))
_Z_VER = (0.01 / 0.006 * (0.0024 - 0.0008), 0.01 / 0.006 * (0.0024 + 0.0008))
_FRUSTUM_CENTER = (_Z_HOR[1] - _Z_HOR[0]) / 2.0
_FRUSTUM_HALFWIDTH = (_Z_HOR[1] + _Z_HOR[0]) / 2.0


class _GlCamera:
    """The subset of ``mjvGLCamera`` the intrinsics read."""

    def __init__(self, top: float, bottom: float) -> None:
        self.frustum_top = top
        self.frustum_bottom = bottom
        self.frustum_center = _FRUSTUM_CENTER
        self.frustum_width = _FRUSTUM_HALFWIDTH
        self.frustum_near = 0.01


class _StubScene:
    def __init__(self, gl_camera: _GlCamera) -> None:
        self.camera = (gl_camera, gl_camera)


class _StubMj:
    """A ``mujoco`` stand-in whose ``mjv_updateCamera`` yields a fixed frustum.

    Emulating the frustum rather than reading it off the installed build is
    what lets one run pin both of MuJoCo's vertical conventions.
    """

    class mjtCamera:  # noqa: N801 - mirrors the mujoco enum name
        mjCAMERA_FIXED = 2

    def __init__(self, gl_camera: _GlCamera) -> None:
        self._gl_camera = gl_camera
        self.requested_cam_ids: list[int] = []

    def MjvCamera(self) -> Any:  # noqa: N802 - mirrors the mujoco factory name
        return type("Cam", (), {"type": None, "fixedcamid": -1})()

    def MjvScene(self) -> _StubScene:  # noqa: N802 - mirrors the mujoco factory name
        return _StubScene(self._gl_camera)

    def mjv_updateCamera(self, model: Any, data: Any, cam: Any, scene: Any) -> None:  # noqa: N802
        assert cam.type == self.mjtCamera.mjCAMERA_FIXED
        self.requested_cam_ids.append(cam.fixedcamid)


def _K_from_frustum(top: float, bottom: float, w: int = 320, h: int = 240) -> Any:
    mj = _StubMj(_GlCamera(top=top, bottom=bottom))
    K = RenderingMixin._explicit_intrinsics_K(mj, np, object(), object(), 7, w, h)
    assert mj.requested_cam_ids == [7], "the requested camera id must reach mjv_updateCamera"
    return K


class TestPrincipalPointFollowsTheFrustumConvention:
    """Both vertical conventions MuJoCo has shipped map to the right ``cy``.

    The sensor geometry fixes the offset's magnitude (0.8 mm on a 4.8 mm
    sensor over 240 rows = 40 px); the frustum fixes its side of the image
    center. ``fx``, ``fy`` and ``cx`` are the same either way - the swap was
    vertical only.
    """

    def test_pre_3_6_frustum_puts_the_principal_point_below_the_center(self) -> None:
        """MuJoCo <= 3.5: ``frustum_top = zver[1]`` (the larger extent)."""
        K = _K_from_frustum(top=_Z_VER[1], bottom=-_Z_VER[0])
        assert K is not None
        assert K[0, 0] == pytest.approx(300.0)
        assert K[1, 1] == pytest.approx(300.0)
        assert K[0, 2] == pytest.approx(85.0)
        assert K[1, 2] == pytest.approx(160.0)

    def test_3_6_and_later_frustum_puts_the_principal_point_above_the_center(self) -> None:
        """MuJoCo >= 3.6: ``frustum_top = zver[0]`` (the smaller extent)."""
        K = _K_from_frustum(top=_Z_VER[0], bottom=-_Z_VER[1])
        assert K is not None
        assert K[0, 0] == pytest.approx(300.0)
        assert K[1, 1] == pytest.approx(300.0)
        assert K[0, 2] == pytest.approx(85.0)
        assert K[1, 2] == pytest.approx(80.0)

    def test_the_two_conventions_are_mirrored_about_the_image_center(self) -> None:
        """Reading the wrong one is not an approximation - it is 2x the offset."""
        pre = _K_from_frustum(top=_Z_VER[1], bottom=-_Z_VER[0])
        post = _K_from_frustum(top=_Z_VER[0], bottom=-_Z_VER[1])
        assert pre is not None and post is not None
        assert (pre[1, 2] - 120.0) == pytest.approx(-(post[1, 2] - 120.0))
        assert abs(pre[1, 2] - post[1, 2]) == pytest.approx(80.0)

    def test_a_larger_viewport_scales_the_frustum_derived_K_linearly(self) -> None:
        """The sensor fixes the frustum; the renderer maps it onto the viewport."""
        small = _K_from_frustum(top=_Z_VER[1], bottom=-_Z_VER[0], w=320, h=240)
        large = _K_from_frustum(top=_Z_VER[1], bottom=-_Z_VER[0], w=640, h=480)
        assert small is not None and large is not None
        assert np.allclose(large[:2, :], 2.0 * small[:2, :])


class TestFrustumReadEdgeCases:
    def test_a_sensorless_frustum_leaves_the_fovy_path_reachable(self) -> None:
        """No physical sensor -> MuJoCo zeroes the horizontal extent."""
        gl = _GlCamera(top=0.004142, bottom=-0.004142)
        gl.frustum_center = 0.0
        gl.frustum_width = 0.0
        K = RenderingMixin._explicit_intrinsics_K(_StubMj(gl), np, object(), object(), 0, 320, 240)
        assert K is None

    def test_a_collapsed_vertical_frustum_is_reported_not_divided_by(self) -> None:
        with pytest.raises(ValueError, match="non-positive vertical frustum extent"):
            _K_from_frustum(top=-0.001, bottom=-0.001)


@requires_gl
def test_the_rasterizer_draws_an_on_axis_target_at_the_reported_principal_point(tmp_path) -> None:
    """The oracle: a sphere on the optical axis lands at ``(cx, cy)``.

    The camera looks down +y with +z up and the sphere sits exactly on its
    optical axis, so its depth-blob centroid IS the principal point MuJoCo
    rasterized with - no ``K`` involved in producing it. A ``K`` derived from a
    vertical convention this MuJoCo build does not use misses it by twice the
    principal-point offset (40 px at the camera's own resolution, 80 px at the
    render size below), which is exactly the case that reported success while
    grounding a pixel a quarter of a metre away.
    """
    from strands_robots.simulation import Simulation

    scene = tmp_path / "offset_principal_point.xml"
    scene.write_text(_SCENE)
    sim = Simulation(mesh=False)
    try:
        assert sim.create_world(ground_plane=False)["status"] == "success"
        assert sim.load_scene(str(scene))["status"] == "success"
        _rgb, depth = sim.get_frame("poff")
        cam = sim.get_camera_params("poff")
        assert depth is not None and depth.shape == (cam.height, cam.width)

        rows, cols = np.where(depth < cam.zfar * 0.5)
        assert len(rows) > 0, "the on-axis sphere must be visible in depth"
        # Pixel indices are the continuous image coordinates minus half a pixel.
        assert cols.mean() == pytest.approx(cam.K[0, 2] - 0.5, abs=1.0)
        assert rows.mean() == pytest.approx(cam.K[1, 2] - 0.5, abs=1.0)
        # ... and the offset is real, so this is not a centered-K tautology.
        assert abs(cam.K[0, 2] - cam.width / 2) == pytest.approx(cam.width * 0.0015 / 0.0064, abs=1.0)
        assert abs(cam.K[1, 2] - cam.height / 2) == pytest.approx(cam.height * 0.0008 / 0.0048, abs=1.0)
    finally:
        sim.destroy()
