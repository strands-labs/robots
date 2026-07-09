"""Regression: sim robot camera frames published on mesh with JPEG encoding.

Bug (issue #24, part 2): _publish_cameras_once() only worked for hardware
robots (checks inner.is_connected). Sim robot child peers on the mesh never
published camera frames because the SimRobot dataclass has no inner lerobot
Robot wrapper.

Fix: _publish_cameras_once() now delegates to _publish_sim_cameras() when
self.robot is a SimRobot with a _world back-reference. The sim path renders
each camera from the MuJoCo world and publishes JPEG-encoded frames on
strands/<peer_id>/camera/<cam_name>.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="sim camera tests require mujoco")


@pytest.fixture(autouse=True)
def _egl_headless(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")


class TestSimCameraPublish:
    """Unit tests for _publish_sim_cameras path."""

    def test_publish_sim_cameras_noop_without_world(self):
        """No-op when SimRobot has no _world reference."""
        from strands_robots.mesh.core import Mesh
        from strands_robots.simulation.models import SimRobot

        robot = SimRobot(name="arm", urdf_path="/tmp/x.urdf")
        robot._world = None

        mesh = Mesh.__new__(Mesh)
        mesh.robot = robot
        mesh.peer_id = "test__arm"
        mesh._running = True

        with patch.object(mesh, "publish") as pub:
            mesh._publish_sim_cameras()
            pub.assert_not_called()

    def test_publish_sim_cameras_renders_and_encodes(self):
        """With a live world + camera, renders and publishes JPEG."""
        from strands_robots.mesh.core import Mesh
        from strands_robots.simulation.models import SimRobot, SimWorld

        # Build a minimal MuJoCo world with one camera
        xml = """
        <mujoco>
          <worldbody>
            <camera name="arm/top_cam" pos="0 0 1" xyaxes="1 0 0 0 1 0"/>
            <body name="arm/base">
              <joint name="arm/j1" type="hinge"/>
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        world = SimWorld()
        world._model = model
        world._data = data

        robot = SimRobot(
            name="arm",
            urdf_path="/tmp/x.urdf",
            namespace="arm/",
            joint_names=["j1"],
        )
        robot._world = world

        mesh = Mesh.__new__(Mesh)
        mesh.robot = robot
        mesh.peer_id = "sim__arm"
        mesh._running = True

        published: list[tuple[str, dict]] = []

        def fake_publish(key, payload):
            published.append((key, payload))

        mesh.publish = fake_publish

        mesh._publish_sim_cameras()

        # Should have published one frame for "top_cam" (namespace stripped)
        assert len(published) == 1, f"Expected 1 camera publish, got {len(published)}"
        key, payload = published[0]
        assert key == "strands/sim__arm/camera/top_cam"
        assert payload["cam"] == "top_cam"
        assert payload["encoding"] == "jpeg"
        assert payload["dtype"] == "uint8"
        assert "data" in payload
        assert len(payload["data"]) > 0  # non-empty JPEG base64

    def test_publish_sim_cameras_skips_unscoped_cameras(self):
        """Cameras not under this robot's namespace are not published."""
        from strands_robots.mesh.core import Mesh
        from strands_robots.simulation.models import SimRobot, SimWorld

        xml = """
        <mujoco>
          <worldbody>
            <camera name="other/cam" pos="0 0 1" xyaxes="1 0 0 0 1 0"/>
            <body name="arm/base">
              <joint name="arm/j1" type="hinge"/>
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        world = SimWorld()
        world._model = model
        world._data = data

        robot = SimRobot(
            name="arm",
            urdf_path="/tmp/x.urdf",
            namespace="arm/",
            joint_names=["j1"],
        )
        robot._world = world

        mesh = Mesh.__new__(Mesh)
        mesh.robot = robot
        mesh.peer_id = "sim__arm"
        mesh._running = True

        published: list = []
        mesh.publish = lambda k, p: published.append((k, p))

        mesh._publish_sim_cameras()

        # "other/cam" doesn't start with "arm/" so it's skipped
        assert len(published) == 0

    def test_publish_cameras_once_dispatches_to_sim_path(self):
        """_publish_cameras_once routes to _publish_sim_cameras for SimRobots."""
        from strands_robots.mesh.core import Mesh
        from strands_robots.simulation.models import SimRobot

        robot = SimRobot(name="arm", urdf_path="/tmp/x.urdf")
        robot._world = MagicMock()  # non-None triggers sim path

        mesh = Mesh.__new__(Mesh)
        mesh.robot = robot
        mesh.peer_id = "sim__arm"
        mesh._running = True

        with (
            patch.object(mesh, "_publish_sim_cameras") as sim_cam,
            patch.object(mesh, "_publish_hardware_cameras") as hw_cam,
            patch("strands_robots.mesh._zenoh_config._bool_env", return_value=False),
        ):
            mesh._publish_cameras_once()
            sim_cam.assert_called_once()
            hw_cam.assert_not_called()

    def test_publish_cameras_once_dispatches_to_hardware_path(self):
        """_publish_cameras_once routes to _publish_hardware_cameras for hw robots."""
        from strands_robots.mesh.core import Mesh

        class FakeHwRobot:
            class robot:
                is_connected = True
                config = MagicMock()
                config.cameras = {"cam0": {}}

        mesh = Mesh.__new__(Mesh)
        mesh.robot = FakeHwRobot()
        mesh.peer_id = "hw-bot"
        mesh._running = True

        with (
            patch.object(mesh, "_publish_hardware_cameras") as hw_cam,
            patch.object(mesh, "_publish_sim_cameras") as sim_cam,
            patch("strands_robots.mesh._zenoh_config._bool_env", return_value=False),
        ):
            mesh._publish_cameras_once()
            hw_cam.assert_called_once()
            sim_cam.assert_not_called()


class TestSimCameraPublishGuards:
    """Guard + resilience contracts for the _publish_sim_cameras path.

    A sim child peer must publish camera frames without ever crashing the
    background camera loop: partially-attached worlds, nameless robots, a
    missing mujoco install, unnamed cameras, and per-camera render failures
    are all tolerated and result in a silent no-op rather than an exception.
    """

    @staticmethod
    def _mesh_for(robot):
        from strands_robots.mesh.core import Mesh

        mesh = Mesh.__new__(Mesh)
        mesh.robot = robot
        mesh.peer_id = "sim__arm"
        mesh._running = True
        published: list = []
        mesh.publish = lambda k, p: published.append((k, p))
        return mesh, published

    def test_noop_when_world_not_built(self):
        """World attached but MuJoCo model/data not built yet -> no-op.

        A SimRobot gets its _world back-reference on attach, but the world's
        _model/_data are only populated once the sim is constructed. Publishing
        before that must short-circuit instead of dereferencing None.
        """
        from strands_robots.simulation.models import SimRobot, SimWorld

        world = SimWorld()
        world._model = None
        world._data = None

        robot = SimRobot(name="arm", urdf_path="/tmp/x.urdf", namespace="arm/")
        robot._world = world

        mesh, published = self._mesh_for(robot)
        mesh._publish_sim_cameras()

        assert published == []

    def test_noop_without_robot_name(self):
        """A SimRobot with no name cannot scope its cameras -> no-op."""
        from strands_robots.simulation.models import SimRobot, SimWorld

        xml = """
        <mujoco>
          <worldbody>
            <camera name="arm/top_cam" pos="0 0 1" xyaxes="1 0 0 0 1 0"/>
            <body name="arm/base">
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        world = SimWorld()
        world._model = model
        world._data = data

        robot = SimRobot(name="", urdf_path="/tmp/x.urdf", namespace="arm/")
        robot._world = world

        mesh, published = self._mesh_for(robot)
        mesh._publish_sim_cameras()

        assert published == []

    def test_noop_without_mujoco_installed(self, monkeypatch):
        """When mujoco is not importable the sim camera path is a silent no-op.

        [mesh] does not depend on [sim-mujoco]; a mesh peer running without the
        sim extra must degrade gracefully rather than raise ImportError.
        """
        import sys

        from strands_robots.simulation.models import SimRobot, SimWorld

        world = SimWorld()
        world._model = MagicMock()
        world._data = MagicMock()

        robot = SimRobot(name="arm", urdf_path="/tmp/x.urdf", namespace="arm/")
        robot._world = world

        mesh, published = self._mesh_for(robot)
        # Setting the cached module entry to None makes `import mujoco` raise
        # ImportError, simulating a peer without the [sim-mujoco] extra.
        monkeypatch.setitem(sys.modules, "mujoco", None)
        mesh._publish_sim_cameras()

        assert published == []

    def test_skips_unnamed_camera(self):
        """A camera with no name attribute is skipped (mj_id2name -> empty)."""
        from strands_robots.simulation.models import SimRobot, SimWorld

        # Unnamed camera in an un-namespaced robot: cam_name resolves to empty,
        # so it is skipped before any render is attempted.
        xml = """
        <mujoco>
          <worldbody>
            <camera pos="0 0 1" xyaxes="1 0 0 0 1 0"/>
            <body name="base">
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        world = SimWorld()
        world._model = model
        world._data = data

        robot = SimRobot(name="arm", urdf_path="/tmp/x.urdf", namespace="")
        robot._world = world

        mesh, published = self._mesh_for(robot)
        mesh._publish_sim_cameras()

        assert published == []

    def test_survives_per_camera_render_failure(self):
        """A render failure on one camera is swallowed; nothing is published.

        One flaky camera must not crash the background publish loop nor leak a
        partial frame. The render error is logged at debug and the method
        returns normally with an empty frame set.
        """
        from strands_robots.simulation.models import SimRobot, SimWorld

        xml = """
        <mujoco>
          <worldbody>
            <camera name="arm/top_cam" pos="0 0 1" xyaxes="1 0 0 0 1 0"/>
            <body name="arm/base">
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        world = SimWorld()
        world._model = model
        world._data = data

        robot = SimRobot(name="arm", urdf_path="/tmp/x.urdf", namespace="arm/")
        robot._world = world

        mesh, published = self._mesh_for(robot)
        with patch("mujoco.Renderer", side_effect=RuntimeError("EGL context lost")):
            mesh._publish_sim_cameras()  # must not raise

        assert published == []


class TestEncodeAndPublishFrames:
    """Unit tests for the shared _encode_and_publish_frames helper."""

    def test_jpeg_encoding_produces_valid_base64(self):
        """JPEG encoding path produces decodable base64 payload."""
        import base64

        from strands_robots.mesh.core import Mesh

        mesh = Mesh.__new__(Mesh)
        mesh.peer_id = "test-peer"
        mesh._running = True

        # Synthetic 4x4 RGB frame
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[1, 1] = [255, 0, 0]

        published: list[tuple[str, dict]] = []
        mesh.publish = lambda k, p: published.append((k, p))

        mesh._encode_and_publish_frames({"cam0": frame}, ["cam0"])

        assert len(published) == 1
        _, payload = published[0]
        assert payload["encoding"] in ("jpeg", "raw")
        # Verify base64 decodes without error
        raw_bytes = base64.b64decode(payload["data"])
        assert len(raw_bytes) > 0

    def test_skips_none_frames(self):
        """Frames that are None are skipped silently."""
        from strands_robots.mesh.core import Mesh

        mesh = Mesh.__new__(Mesh)
        mesh.peer_id = "test-peer"
        mesh._running = True

        published: list = []
        mesh.publish = lambda k, p: published.append((k, p))

        mesh._encode_and_publish_frames({"cam0": None}, ["cam0"])
        assert len(published) == 0

    def test_converts_float32_to_uint8(self):
        """Float32 frames are cast to uint8 before encoding."""
        from strands_robots.mesh.core import Mesh

        mesh = Mesh.__new__(Mesh)
        mesh.peer_id = "test-peer"
        mesh._running = True

        # MuJoCo renders as uint8, but some pipelines produce float32
        frame = np.ones((4, 4, 3), dtype=np.float32) * 128.0

        published: list[tuple[str, dict]] = []
        mesh.publish = lambda k, p: published.append((k, p))

        mesh._encode_and_publish_frames({"cam0": frame}, ["cam0"])
        assert len(published) == 1
        assert published[0][1]["dtype"] == "uint8"


class TestHardwareCameraPublish:
    """Unit tests for the _publish_hardware_cameras path.

    A hardware mesh peer wraps a lerobot ``Robot`` (``self.robot.robot``).
    Camera frames are sourced from ``inner.get_observation()`` when available,
    and otherwise read directly from the per-camera objects in
    ``inner.cameras`` (``async_read``/``read``). Either way the configured
    cameras (``inner.config.cameras``) determine which keys are published.
    """

    @staticmethod
    def _mesh_with_capture():
        from strands_robots.mesh.core import Mesh

        mesh = Mesh.__new__(Mesh)
        mesh.peer_id = "hw-peer"
        mesh._running = True
        published: list[tuple[str, dict]] = []
        mesh.publish = lambda k, p: published.append((k, p))
        return mesh, published

    def test_publishes_frames_from_get_observation(self):
        """Configured cameras present in get_observation() are published."""
        mesh, published = self._mesh_with_capture()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        inner = MagicMock()
        inner.config.cameras = {"wrist": object(), "top": object()}
        inner.get_observation.return_value = {"wrist": frame, "top": frame}

        mesh._publish_hardware_cameras(inner)

        topics = {k for k, _ in published}
        assert topics == {"strands/hw-peer/camera/wrist", "strands/hw-peer/camera/top"}

    def test_noop_when_no_cameras_configured(self):
        """An inner robot with no camera config publishes nothing."""
        mesh, published = self._mesh_with_capture()
        inner = MagicMock()
        inner.config.cameras = {}

        mesh._publish_hardware_cameras(inner)

        assert published == []
        # The camera-config gate short-circuits before reading observations.
        inner.get_observation.assert_not_called()

    def test_falls_back_to_camera_objects_when_observation_unavailable(self):
        """When get_observation() raises, frames are read from inner.cameras.

        The per-camera objects expose ``async_read`` (preferred) or ``read``;
        the fallback path must still publish a frame per configured camera.
        """
        mesh, published = self._mesh_with_capture()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        async_cam = MagicMock(spec=["async_read"])
        async_cam.async_read.return_value = frame
        read_cam = MagicMock(spec=["read"])
        read_cam.read.return_value = frame

        inner = MagicMock()
        inner.config.cameras = {"wrist": object(), "top": object()}
        inner.get_observation.side_effect = RuntimeError("device busy")
        inner.cameras = {"wrist": async_cam, "top": read_cam}

        mesh._publish_hardware_cameras(inner)

        async_cam.async_read.assert_called_once()
        read_cam.read.assert_called_once()
        topics = {k for k, _ in published}
        assert topics == {"strands/hw-peer/camera/wrist", "strands/hw-peer/camera/top"}

    def test_noop_when_fallback_has_no_camera_objects(self):
        """get_observation() unavailable AND no inner.cameras -> nothing published."""
        mesh, published = self._mesh_with_capture()
        inner = MagicMock()
        inner.config.cameras = {"wrist": object()}
        inner.get_observation.side_effect = RuntimeError("device busy")
        inner.cameras = {}

        mesh._publish_hardware_cameras(inner)

        assert published == []

    def test_noop_when_every_fallback_camera_read_fails(self):
        """Fallback cameras present but each read raises -> obs stays empty.

        Per-camera read errors are swallowed so one flaky device cannot crash
        the publish loop; when no frame is recovered nothing is published.
        """
        mesh, published = self._mesh_with_capture()
        flaky_cam = MagicMock(spec=["async_read"])
        flaky_cam.async_read.side_effect = OSError("frame grab failed")

        inner = MagicMock()
        inner.config.cameras = {"wrist": object()}
        inner.get_observation.side_effect = RuntimeError("device busy")
        inner.cameras = {"wrist": flaky_cam}

        mesh._publish_hardware_cameras(inner)

        flaky_cam.async_read.assert_called_once()
        assert published == []
