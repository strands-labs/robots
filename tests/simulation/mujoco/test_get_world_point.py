# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``get_world_point`` on the MuJoCo backend (issue #1647).

Acceptance criteria: place a box at a known pose, pick pixels on its visible
face via a fixed camera, and assert the median world point lands on the
ground-truth surface; sky/zfar pixels are excluded; the action is advertised
in tool_spec + describe() and dispatches with validated params. GL-needing
tests are gated behind the shared runtime probe; the pure-math contract lives
in ``tests/simulation/test_get_world_point_math.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl

# Ground truth: a static box whose top face is centered at [0.4, 0, 0.1].
_BOX_POS = [0.4, 0.0, 0.05]
_BOX_SIZE = [0.2, 0.2, 0.1]  # full extents -> top face z = 0.1, x in [0.3, 0.5], y in [-0.1, 0.1]
_TOP_CENTER = np.array([0.4, 0.0, 0.1])


def _make_box_sim(width: int = 96, height: int = 72):
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    # No ground plane: everything that is not the box is background (pinned to
    # zfar by get_frame), so invalid-pixel behavior is testable.
    sim.create_world(ground_plane=False)
    sim.add_object(name="target_box", shape="box", position=list(_BOX_POS), size=list(_BOX_SIZE), is_static=True)
    # The camera-to-target ray hits the box's TOP face exactly at _TOP_CENTER
    # (it clears the front face: at y = -0.1 the ray is at z = 0.18 > 0.1).
    sim.add_camera("front", position=[0.4, -0.5, 0.5], target=[0.4, 0.0, 0.1], width=width, height=height)
    sim.step(n_steps=2)
    return sim


def _project(sim, world_point: np.ndarray, camera: str = "front") -> tuple[int, int]:
    """Project a world point to integer pixel indices via get_camera_params."""
    cam = sim.get_camera_params(camera)
    p_cam = np.linalg.inv(cam.T_world_cam) @ np.array([*world_point, 1.0])
    d = -p_cam[2]  # OpenGL optical frame: -Z forward
    assert d > 0, "point must be in front of the camera"
    u = p_cam[0] / d * cam.K[0, 0] + cam.K[0, 2] - 0.5
    v = -p_cam[1] / d * cam.K[1, 1] + cam.K[1, 2] - 0.5
    return int(round(u)), int(round(v))


def _json_block(result: dict) -> dict:
    assert result["status"] == "success", result
    return result["content"][1]["json"]


# ----- Regression: known box pose round-trip (GL required) ----- #


@requires_gl
def test_median_world_point_matches_box_ground_truth() -> None:
    """5 pixels on the box's top face -> median within tolerance of sim truth.

    The center pixel is the projection of the top-face center [0.4, 0, 0.1];
    the 4 neighbors are symmetric +-3 px offsets on the same face, so the
    per-component median must recover the center point itself. This is the
    issue's acceptance regression: a sign/convention error in the
    unprojection (y-flip, z-forward, pixel center) lands centimeters-to-
    meters away and fails loudly.
    """
    sim = _make_box_sim()
    try:
        u, v = _project(sim, _TOP_CENTER)
        pixels = [[u, v], [u - 3, v], [u + 3, v], [u, v - 3], [u, v + 3]]
        data = _json_block(sim.get_world_point("front", pixels=pixels))
        assert data["n_valid"] == 5, data
        point = np.asarray(data["point"])
        # Depth axis (top face plane): tight. Lateral: dominated by the +-3 px
        # sampling offsets (~3 cm on the face), which the median cancels.
        assert point[2] == pytest.approx(0.1, abs=0.01)
        assert point[0] == pytest.approx(0.4, abs=0.02)
        assert point[1] == pytest.approx(0.0, abs=0.03)
        # Every individual sample stayed on the top face.
        for p in data["points"]:
            assert p is not None
            assert 0.3 - 0.02 <= p[0] <= 0.5 + 0.02
            assert -0.1 - 0.03 <= p[1] <= 0.1 + 0.03
            assert p[2] == pytest.approx(0.1, abs=0.01)
    finally:
        sim.destroy()


@requires_gl
def test_sky_pixels_are_excluded_and_reported_in_n_valid() -> None:
    """A background (zfar) pixel is dropped from the median, not unprojected."""
    sim = _make_box_sim()
    try:
        _rgb, depth = sim.get_frame("front")
        cam = sim.get_camera_params("front")
        sky = np.argwhere(depth >= cam.zfar * 0.999)
        assert len(sky) > 0, "scene with no ground plane must have background pixels"
        sky_v, sky_u = int(sky[0][0]), int(sky[0][1])
        u, v = _project(sim, _TOP_CENTER)
        data = _json_block(sim.get_world_point("front", pixels=[[sky_u, sky_v], [u, v]]))
        assert data["n_valid"] == 1 and data["n_requested"] == 2
        assert data["points"][0] is None
        assert np.asarray(data["point"])[2] == pytest.approx(0.1, abs=0.01)
    finally:
        sim.destroy()


@requires_gl
def test_all_sky_pixels_return_a_structured_error() -> None:
    sim = _make_box_sim()
    try:
        _rgb, depth = sim.get_frame("front")
        cam = sim.get_camera_params("front")
        sky = np.argwhere(depth >= cam.zfar * 0.999)
        assert len(sky) >= 3
        pixels = [[int(p[1]), int(p[0])] for p in sky[:3]]
        result = sim.get_world_point("front", pixels=pixels)
        assert result["status"] == "error"
        assert "no valid depth" in result["content"][0]["text"].lower()
    finally:
        sim.destroy()


@requires_gl
def test_out_of_bounds_pixel_is_rejected_with_the_frame_size() -> None:
    sim = _make_box_sim(width=96, height=72)
    try:
        result = sim.get_world_point("front", pixels=[[96, 0]])
        assert result["status"] == "error"
        assert "96x72" in result["content"][0]["text"]
    finally:
        sim.destroy()


@requires_gl
def test_dispatch_route_matches_programmatic_call() -> None:
    """The agent-facing dispatch path returns the same grounding as the
    programmatic API -- no silent kwarg drops through the router."""
    sim = _make_box_sim()
    try:
        u, v = _project(sim, _TOP_CENTER)
        via_dispatch = sim._dispatch_action(
            "get_world_point", {"action": "get_world_point", "camera_name": "front", "pixels": [[u, v]]}
        )
        direct = sim.get_world_point("front", pixels=[[u, v]])
        assert via_dispatch["status"] == "success", via_dispatch
        assert _json_block(via_dispatch) == _json_block(direct)
    finally:
        sim.destroy()


# ----- Explicit MJCF intrinsics (review on #1649, item 2) ----- #

# A camera declared with a physical sensor: MuJoCo rasterizes with the
# intrinsic model (offset principal point, fovy ignored). The fovy-derived K
# put the principal point at the image center and produced ~25 cm of silent
# world-point error on this exact camera.
_INTRINSICS_SCENE = """
<mujoco model="intrinsics_scene">
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


class TestExplicitIntrinsicsK:
    """Sensor-derived ``K`` pins on the installed MuJoCo (no GL: no render).

    ``fx``/``fy`` are the sensor's own pixel pitch and ``cx`` follows MuJoCo's
    horizontal frustum, so all three are pinned exactly here. The vertical
    offset's magnitude is pinned too, but not which side of the image center it
    falls on: MuJoCo 3.6.0 changed the vertical frustum convention, so that
    side is a property of the build. Which frustum maps to which ``cy`` is
    pinned for both conventions by the frustum-mapping tests.
    """

    @staticmethod
    def _K(camera_attrs: str, w: int, h: int):
        import mujoco

        from strands_robots.simulation.mujoco.rendering import RenderingMixin

        xml = f"""
        <mujoco>
          <worldbody>
            <camera name="c" pos="0 -1 0.1" xyaxes="1 0 0 0 0 1" {camera_attrs}/>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "c")
        return RenderingMixin._explicit_intrinsics_K(mujoco, np, model, data, cam_id, w, h)

    def test_offset_principal_point(self) -> None:
        K = self._K(
            'sensorsize="0.0064 0.0048" focal="0.006 0.006" resolution="320 240" principal="0.0015 0.0008"',
            320,
            240,
        )
        assert K is not None
        assert K[0, 0] == pytest.approx(300.0)
        assert K[1, 1] == pytest.approx(300.0)
        assert K[0, 2] == pytest.approx(85.0)
        # 0.8 mm on a 4.8 mm sensor over 240 rows = 40 px off the center.
        assert abs(K[1, 2] - 120.0) == pytest.approx(40.0)

    def test_non_square_sensor(self) -> None:
        K = self._K('sensorsize="0.008 0.0048" focal="0.006 0.006" resolution="320 240"', 320, 240)
        assert K is not None
        assert K[0, 0] == pytest.approx(240.0)  # fx != fy: non-square pixels honored
        assert K[1, 1] == pytest.approx(300.0)
        assert K[0, 2] == pytest.approx(160.0)
        assert K[1, 2] == pytest.approx(120.0)

    def test_asymmetric_focal_negative_principal(self) -> None:
        K = self._K(
            'sensorsize="0.0064 0.0048" focal="0.004 0.006" resolution="320 240" principal="-0.001 0.0012"',
            320,
            240,
        )
        assert K is not None
        assert K[0, 0] == pytest.approx(200.0)
        assert K[1, 1] == pytest.approx(300.0)
        assert K[0, 2] == pytest.approx(210.0)
        # 1.2 mm on a 4.8 mm sensor over 240 rows = 60 px off the center.
        assert abs(K[1, 2] - 120.0) == pytest.approx(60.0)

    def test_scales_linearly_with_render_size(self) -> None:
        """The sensor fixes the frustum; a 2x viewport doubles K per axis
        (verified against the rasterizer: the blob centroid at 640x480 lands
        at exactly 2x its 320x240 coordinates)."""
        attrs = 'sensorsize="0.0064 0.0048" focal="0.006 0.006" resolution="320 240" principal="0.0015 0.0008"'
        K1 = self._K(attrs, 320, 240)
        K2 = self._K(attrs, 640, 480)
        assert K1 is not None and K2 is not None
        assert np.allclose(K2[:2, :], 2.0 * K1[:2, :])

    def test_fovy_camera_returns_none(self) -> None:
        """No physical sensor declared -> the fovy path applies (fallback)."""
        assert self._K('fovy="45"', 320, 240) is None


@requires_gl
def test_world_point_correct_on_explicit_intrinsics_camera(tmp_path) -> None:
    """Round-trip on the review's repro camera, non-circularly: the target
    pixel comes from the DEPTH-blob centroid (rasterizer truth, no K
    involved), and the unprojected point must land on the sphere's visible
    surface. The fovy-derived K was ~25 cm off here and reported success."""
    from strands_robots.simulation import Simulation

    scene = tmp_path / "intrinsics_scene.xml"
    scene.write_text(_INTRINSICS_SCENE)
    sim = Simulation(mesh=False)
    try:
        assert sim.create_world(ground_plane=False)["status"] == "success"
        assert sim.load_scene(str(scene))["status"] == "success", "load_scene failed"
        _rgb, depth = sim.get_frame("poff")
        assert depth is not None
        cam = sim.get_camera_params("poff")
        mask = depth < cam.zfar * 0.5
        assert mask.sum() > 0, "sphere not visible in depth"
        vs, us = np.where(mask)
        u, v = int(round(us.mean())), int(round(vs.mean()))
        data = _json_block(sim.get_world_point("poff", pixels=[[u, v]]))
        point = np.asarray(data["point"])
        # Camera at (0, -1, 0.1) looking +y; the sphere (r=0.01) at
        # (0, 0, 0.1) presents its near surface at ~(0, -0.01, 0.1).
        expected = np.array([0.0, -0.01, 0.1])
        assert np.linalg.norm(point - expected) < 0.02, (
            f"world point {point.tolist()} is {np.linalg.norm(point - expected):.4f} m from "
            f"the sphere surface {expected.tolist()} - explicit intrinsics not honored"
        )
    finally:
        sim.destroy()


# A scene whose FREE camera is orthographic. MuJoCo rasterizes it normally --
# ``get_frame`` returns a full RGB + depth pair -- but an orthographic
# projection has no pinhole ``K``, so ``get_camera_params`` refuses. That makes
# it the reachable real-world instance of a failure that arrives AFTER a
# successful render, on pixels ``get_world_point`` has already accepted.
_ORTHOGRAPHIC_SCENE = """
<mujoco model="orthographic_free_camera">
  <visual><global orthographic="true" fovy="0.35"/></visual>
  <worldbody>
    <light pos="0.4 -0.6 1.2" dir="-0.2 0.4 -1"/>
    <body name="cube" pos="0.4 0 0.05">
      <geom name="cube_g" type="box" size="0.1 0.1 0.05" rgba="0.9 0.4 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _load_scene_sim(tmp_path, scene_xml: str, name: str):
    """Fresh sim with *scene_xml* loaded, mirroring the intrinsics fixture."""
    from strands_robots.simulation import Simulation

    scene = tmp_path / name
    scene.write_text(scene_xml)
    sim = Simulation(mesh=False)
    assert sim.create_world(ground_plane=False)["status"] == "success"
    assert sim.load_scene(str(scene))["status"] == "success", "load_scene failed"
    return sim


@requires_gl
def test_orthographic_free_camera_renders_but_reports_unreadable_intrinsics(tmp_path) -> None:
    """The camera-params read fails on a frame that rendered fine.

    ``get_world_point`` makes two backend reads. This pins the second one
    failing while the first succeeded, through a documented MJCF option rather
    than a mock: the pixels are valid, the depth buffer is real, and the call
    still has to report -- never raise, and never return a point derived from
    intrinsics it could not read. The reported text has to carry MuJoCo's own
    actionable reason, because nothing else in the result names the cause.
    """
    sim = _load_scene_sim(tmp_path, _ORTHOGRAPHIC_SCENE, "orthographic.xml")
    try:
        # The frame read succeeds: this is not a rendering failure.
        rgb, depth = sim.get_frame("default")
        assert depth is not None and depth.shape[:2] == rgb.shape[:2]

        result = sim.get_world_point("default", pixels=[[320, 240]])
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "camera parameters" in text, text
        assert "orthographic" in text, text
        # The remedy has to survive into the envelope, not just the exception.
        assert "add_camera" in text, text
        # ... and it must be the params read being reported, not the render.
        assert "render camera frame" not in text, text
    finally:
        sim.destroy()


@requires_gl
def test_a_perspective_free_camera_on_the_same_scene_still_grounds(tmp_path) -> None:
    """Control for the case above: the identical scene, camera and pixel
    ground successfully once the projection is perspective, so the refusal is
    the projection and not the fixture."""
    perspective = _ORTHOGRAPHIC_SCENE.replace('orthographic="true" fovy="0.35"', 'fovy="45"')
    sim = _load_scene_sim(tmp_path, perspective, "perspective.xml")
    try:
        data = _json_block(sim.get_world_point("default", pixels=[[320, 240]]))
        assert data["n_valid"] == 1
        assert len(data["point"]) == 3
    finally:
        sim.destroy()


# ----- Tool surface (no GL required) ----- #


def _spec() -> dict:
    spec_path = Path(__file__).resolve().parents[3] / "strands_robots" / "simulation" / "mujoco" / "tool_spec.json"
    return json.loads(spec_path.read_text())


def test_tool_spec_advertises_get_world_point_and_pixels() -> None:
    spec = _spec()
    assert "get_world_point" in spec["properties"]["action"]["enum"]
    pixels = spec["properties"]["pixels"]
    assert pixels["type"] == "array"
    assert pixels["items"]["type"] == "array"
    # The description carries the paper's localization guidance for agents.
    for keyword in ("median", "surface", "n_valid"):
        assert keyword in pixels["description"], keyword


def test_describe_advertises_get_world_point() -> None:
    from strands_robots.simulation import Simulation

    sim = Simulation(mesh=False)
    try:
        sim.create_world(ground_plane=False)
        methods = sim.describe()["methods"]
        assert "get_world_point" in methods
        for keyword in ("pixels", "median", "surface"):
            assert keyword.lower() in methods["get_world_point"].lower(), keyword
    finally:
        sim.destroy()


def test_dispatch_without_world_is_a_structured_error() -> None:
    from strands_robots.simulation import Simulation

    sim = Simulation(mesh=False)
    try:
        result = sim._dispatch_action("get_world_point", {"pixels": [[1, 1]]})
        assert result["status"] == "error"
        assert "No world" in result["content"][0]["text"]
    finally:
        sim.cleanup()


def test_dispatch_rejects_unknown_params() -> None:
    from strands_robots.simulation import Simulation

    sim = Simulation(mesh=False)
    try:
        result = sim._dispatch_action("get_world_point", {"pixels": [[1, 1]], "bogus": 1})
        assert result["status"] == "error"
        assert "Unknown parameter 'bogus'" in result["content"][0]["text"]
    finally:
        sim.cleanup()


def test_dispatch_non_string_camera_name_is_a_structured_error() -> None:
    """A camera INDEX (int) must degrade like every sibling camera action -
    error dict, never a TypeError out of mj_name2id through the envelope
    (review on #1649, item 1)."""
    from strands_robots.simulation import Simulation

    sim = Simulation(mesh=False)
    try:
        sim.create_world(ground_plane=False)
        result = sim._dispatch_action("get_world_point", {"camera_name": 5, "pixels": [[1, 1]]})
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "camera_name" in text
        assert "int" in text
    finally:
        sim.destroy()


def test_dispatch_missing_pixels_names_the_parameter() -> None:
    from strands_robots.simulation import Simulation

    sim = Simulation(mesh=False)
    try:
        result = sim._dispatch_action("get_world_point", {"camera_name": "front"})
        assert result["status"] == "error"
        assert "pixels" in result["content"][0]["text"]
    finally:
        sim.cleanup()
