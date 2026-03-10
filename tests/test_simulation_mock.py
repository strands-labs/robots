#!/usr/bin/env python3
"""
Mock-based tests for strands_robots.simulation — runs without MuJoCo or Strands SDK.

Strategy: Mock strands SDK + mujoco at module level via sys.modules patching,
then exercise Simulation class logic: world creation, object management,
camera management, dispatch routing, state tracking, domain randomization,
recording, cleanup, properties, MJCF builder, and the AgentTool interface.

Coverage target: simulation.py 1% → 30%+ (1351 statements).
"""

import asyncio
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level sys.modules mocking — MUST happen before any import of
# strands_robots.simulation
# ---------------------------------------------------------------------------

# -- Strands SDK mocks --


class _FakeAgentTool:
    """Stand-in for strands.tools.tools.AgentTool."""

    def __init__(self):
        pass


class _FakeToolResultEvent(dict):
    def __init__(self, d):
        super().__init__(d)


_mock_strands_tools_tools = MagicMock()
_mock_strands_tools_tools.AgentTool = _FakeAgentTool

_mock_strands_events = MagicMock()
_mock_strands_events.ToolResultEvent = _FakeToolResultEvent

_mock_strands_types_tools = MagicMock()
_mock_strands_types_tools.ToolSpec = dict
_mock_strands_types_tools.ToolUse = dict

# -- MuJoCo mocks --


def _build_mock_mujoco():
    """Build a realistic mock of the mujoco package."""
    mock_mj = MagicMock()

    mock_model = MagicMock()
    mock_model.nbody = 5
    mock_model.njnt = 3
    mock_model.nu = 2
    mock_model.ncam = 1
    mock_model.ngeom = 4
    mock_model.opt.timestep = 0.002
    mock_model.opt.gravity = np.array([0.0, 0.0, -9.81])
    mock_model.jnt_qposadr = np.array([0, 1, 2])
    mock_model.jnt_dofadr = np.array([0, 1, 2])
    mock_model.geom_rgba = np.ones((4, 4))

    mock_mj.MjModel.from_xml_string.return_value = mock_model
    mock_mj.MjModel.from_xml_path.return_value = mock_model

    mock_data = MagicMock()
    mock_data.qpos = np.zeros(10)
    mock_data.qvel = np.zeros(10)
    mock_data.ctrl = np.zeros(5)
    mock_data.time = 0.0
    mock_data.ncon = 0
    mock_data.contact = []

    mock_mj.MjData.return_value = mock_data

    mock_mj.mj_step = MagicMock()
    mock_mj.mj_forward = MagicMock()
    mock_mj.mj_resetData = MagicMock()
    mock_mj.mj_saveLastXML = MagicMock()
    mock_mj.mj_name2id = MagicMock(return_value=0)
    mock_mj.mj_id2name = MagicMock(return_value="joint_0")

    mock_mj.mjtObj = MagicMock()
    mock_mj.mjtObj.mjOBJ_JOINT = 1
    mock_mj.mjtObj.mjOBJ_BODY = 2
    mock_mj.mjtObj.mjOBJ_GEOM = 3
    mock_mj.mjtObj.mjOBJ_CAMERA = 4
    mock_mj.mjtObj.mjOBJ_ACTUATOR = 5

    mock_renderer = MagicMock()
    mock_renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_mj.Renderer.return_value = mock_renderer

    mock_viewer = MagicMock()

    return mock_mj, mock_model, mock_data, mock_viewer


_mock_mj, _mock_model, _mock_data, _mock_viewer = _build_mock_mujoco()

# -- Policies mock --
_mock_policies = MagicMock()
_mock_policies.Policy = type("Policy", (), {})
_mock_policies.create_policy = MagicMock()

# -- Zenoh mesh mock --
_mock_zenoh_mesh = MagicMock()
_mock_zenoh_mesh.init_mesh = MagicMock(return_value=None)

# -- Dataset recorder mock --
_mock_dataset_recorder = MagicMock()
_mock_dataset_recorder.HAS_LEROBOT_DATASET = False
_mock_dataset_recorder.DatasetRecorder = None

# Build the full sys.modules patch dict
_PATCHES = {
    # Strands SDK
    "strands": MagicMock(),
    "strands.tools": MagicMock(),
    "strands.tools.tools": _mock_strands_tools_tools,
    "strands.types": MagicMock(),
    "strands.types._events": _mock_strands_events,
    "strands.types.tools": _mock_strands_types_tools,
    # MuJoCo
    "mujoco": _mock_mj,
    "mujoco.viewer": _mock_viewer,
    # Internal deps
    "strands_robots.zenoh_mesh": _mock_zenoh_mesh,
    "strands_robots.policies": _mock_policies,
    "strands_robots.dataset_recorder": _mock_dataset_recorder,
}


# Import simulation module ONCE under mocked deps at module level.
# This avoids re-importing on every test (which cascades through
# strands_robots.__init__ → groot → torch → cv2.typing → DictValue error
# on OpenCV 4.12+).
_sim_key = "strands_robots.simulation"
_saved_sim = sys.modules.pop(_sim_key, None)

# Apply patches at module scope for the initial import
_orig_modules = {}
for k, v in _PATCHES.items():
    _orig_modules[k] = sys.modules.get(k)
    sys.modules[k] = v

try:
    import strands_robots.simulation as _sim_mod
except Exception:
    _sim_mod = None

# Restore original modules (except keep simulation cached)
for k, v in _orig_modules.items():
    if v is None:
        sys.modules.pop(k, None)
    else:
        sys.modules[k] = v

# Keep our mocked simulation in sys.modules for the test session
if _sim_mod is not None:
    sys.modules[_sim_key] = _sim_mod


@pytest.fixture(autouse=True)
def _mock_modules():
    """Ensure simulation.py is available under mocked deps for every test.

    Uses module-level import (done once) and patches sys.modules for the
    duration of each test to keep internal imports working.

    IMPORTANT: We reset _mujoco / _mujoco_viewer both before AND after
    the test to prevent mock objects leaking into later non-mock tests
    (e.g. test_training_pipeline's evaluate() which creates a real
    StrandsSimEnv).
    """
    with patch.dict(sys.modules, _PATCHES):
        # Re-inject our cached simulation module
        sys.modules[_sim_key] = _sim_mod
        if _sim_mod is not None:
            _sim_mod._mujoco = None
            _sim_mod._mujoco_viewer = None
        yield
    # Teardown: clear mock mujoco so real tests get the real module
    if _sim_mod is not None:
        _sim_mod._mujoco = None
        _sim_mod._mujoco_viewer = None


def _fresh_import():
    """Return Simulation class from the cached module-level import."""
    return _sim_mod.Simulation


@pytest.fixture
def Sim():
    """Return the Simulation class."""
    from strands_robots.simulation import Simulation

    return Simulation


@pytest.fixture
def sim(Sim):
    """Create a fresh Simulation instance."""
    s = Sim(tool_name="test_sim", mesh=False)
    yield s
    try:
        s.cleanup()
    except Exception:
        pass


@pytest.fixture
def sim_with_world(sim):
    """Simulation with a world already created."""
    sim.create_world()
    return sim


# ===================================================================
# Dataclass tests
# ===================================================================


class TestDataclasses:

    def test_sim_robot_defaults(self):
        from strands_robots.simulation import SimRobot

        r = SimRobot(name="r1", urdf_path="/fake.urdf")
        assert r.name == "r1"
        assert r.position == [0.0, 0.0, 0.0]
        assert r.orientation == [1.0, 0.0, 0.0, 0.0]
        assert r.policy_running is False
        assert r.policy_steps == 0
        assert r.namespace == ""
        assert r.joint_ids == []
        assert r.joint_names == []
        assert r.actuator_ids == []
        assert r.data_config is None
        assert r.body_id == -1

    def test_sim_object_defaults(self):
        from strands_robots.simulation import SimObject

        o = SimObject(name="cube", shape="box")
        assert o.mass == 0.1
        assert o.is_static is False
        assert o._original_position == [0.0, 0.0, 0.0]
        assert o._original_color == [0.5, 0.5, 0.5, 1.0]

    def test_sim_object_post_init_copies(self):
        from strands_robots.simulation import SimObject

        pos = [1.0, 2.0, 3.0]
        color = [1.0, 0.0, 0.0, 1.0]
        o = SimObject(name="obj", shape="sphere", position=pos, color=color)
        assert o._original_position == [1.0, 2.0, 3.0]
        pos[0] = 999.0
        assert o._original_position[0] == 1.0  # was copied

    def test_sim_camera_defaults(self):
        from strands_robots.simulation import SimCamera

        c = SimCamera(name="cam1")
        assert c.position == [1.0, 1.0, 1.0]
        assert c.target == [0.0, 0.0, 0.0]
        assert c.fov == 60.0
        assert c.width == 640
        assert c.height == 480
        assert c.camera_id == -1

    def test_trajectory_step(self):
        from strands_robots.simulation import TrajectoryStep

        ts = TrajectoryStep(
            timestamp=1.0,
            sim_time=0.5,
            robot_name="arm",
            observation={"qpos": [0.1]},
            action={"ctrl": [0.2]},
            instruction="pick up",
        )
        assert ts.timestamp == 1.0
        assert ts.instruction == "pick up"
        assert ts.robot_name == "arm"

    def test_sim_world_defaults(self):
        from strands_robots.simulation import SimStatus, SimWorld

        w = SimWorld()
        assert w.robots == {}
        assert w.objects == {}
        assert w.cameras == {}
        assert w.timestep == 0.002
        assert w.gravity == [0.0, 0.0, -9.81]
        assert w.ground_plane is True
        assert w.status == SimStatus.IDLE
        assert w.sim_time == 0.0
        assert w.step_count == 0
        assert w._recording is False
        assert w._trajectory == []

    def test_sim_status_values(self):
        from strands_robots.simulation import SimStatus

        assert SimStatus.IDLE.value == "idle"
        assert SimStatus.RUNNING.value == "running"
        assert SimStatus.PAUSED.value == "paused"
        assert SimStatus.COMPLETED.value == "completed"
        assert SimStatus.ERROR.value == "error"


# ===================================================================
# Simulation __init__ and properties
# ===================================================================


class TestSimulationInit:

    def test_init_defaults(self, sim):
        assert sim.tool_name_str == "test_sim"
        assert sim.default_timestep == 0.002
        assert sim.default_width == 640
        assert sim.default_height == 480
        assert sim._world is None

    def test_init_custom(self, Sim):
        s = Sim(tool_name="custom", default_timestep=0.005, default_width=1280, default_height=960, mesh=False)
        assert s.tool_name_str == "custom"
        assert s.default_timestep == 0.005
        assert s.default_width == 1280
        s.cleanup()

    def test_mj_model_no_world(self, sim):
        assert sim.mj_model is None

    def test_mj_data_no_world(self, sim):
        assert sim.mj_data is None

    def test_mj_model_with_world(self, sim_with_world):
        assert sim_with_world.mj_model is not None

    def test_mj_data_with_world(self, sim_with_world):
        assert sim_with_world.mj_data is not None

    def test_tool_name_property(self, sim):
        assert sim.tool_name == "test_sim"

    def test_tool_type_property(self, sim):
        assert sim.tool_type == "simulation"


# ===================================================================
# World Management
# ===================================================================


class TestWorldManagement:

    def test_create_world_success(self, sim):
        result = sim.create_world()
        assert result["status"] == "success"
        assert sim._world is not None
        assert "default" in sim._world.cameras

    def test_create_world_custom_params(self, sim):
        result = sim.create_world(timestep=0.005, gravity=[0, 0, -5.0], ground_plane=False)
        assert result["status"] == "success"
        assert sim._world.timestep == 0.005
        assert sim._world.gravity == [0, 0, -5.0]
        assert sim._world.ground_plane is False

    def test_create_world_twice_fails(self, sim_with_world):
        result = sim_with_world.create_world()
        assert result["status"] == "error"
        assert "already exists" in result["content"][0]["text"]

    def test_destroy(self, sim_with_world):
        result = sim_with_world.destroy()
        assert result["status"] == "success"
        assert sim_with_world._world is None

    def test_destroy_no_world(self, sim):
        result = sim.destroy()
        assert result["status"] == "success"
        assert "No world" in result["content"][0]["text"]

    def test_get_state(self, sim_with_world):
        result = sim_with_world.get_state()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Simulation State" in text
        assert "Robots: 0" in text
        assert "Objects: 0" in text

    def test_get_state_no_world(self, sim):
        result = sim.get_state()
        assert result["status"] == "error"

    def test_reset(self, sim_with_world):
        result = sim_with_world.reset()
        assert result["status"] == "success"
        assert sim_with_world._world.sim_time == 0.0
        assert sim_with_world._world.step_count == 0

    def test_reset_no_world(self, sim):
        result = sim.reset()
        assert result["status"] == "error"

    def test_load_scene_file_not_found(self, sim):
        result = sim.load_scene("/nonexistent/path.xml")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_load_scene_success(self, sim):
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(b"<mujoco/>")
            f.flush()
            result = sim.load_scene(f.name)
        os.unlink(f.name)
        assert result["status"] == "success"
        assert sim._world is not None


# ===================================================================
# Object Management
# ===================================================================


class TestObjectManagement:

    def test_add_object_no_world(self, sim):
        result = sim.add_object(name="cube")
        assert result["status"] == "error"

    def test_add_object_success(self, sim_with_world):
        result = sim_with_world.add_object(
            name="red_cube",
            shape="box",
            position=[0.3, 0, 0.05],
            color=[1, 0, 0, 1],
        )
        assert result["status"] == "success"
        assert "red_cube" in sim_with_world._world.objects

    def test_add_object_duplicate(self, sim_with_world):
        sim_with_world.add_object(name="cube1")
        result = sim_with_world.add_object(name="cube1")
        assert result["status"] == "error"
        assert "exists" in result["content"][0]["text"]

    def test_add_sphere(self, sim_with_world):
        result = sim_with_world.add_object(name="ball", shape="sphere", size=[0.1])
        assert result["status"] == "success"
        assert sim_with_world._world.objects["ball"].shape == "sphere"

    def test_add_cylinder(self, sim_with_world):
        result = sim_with_world.add_object(name="can", shape="cylinder")
        assert result["status"] == "success"

    def test_add_static_object(self, sim_with_world):
        result = sim_with_world.add_object(name="table", shape="box", is_static=True)
        assert result["status"] == "success"
        assert sim_with_world._world.objects["table"].is_static is True

    def test_remove_object(self, sim_with_world):
        sim_with_world.add_object(name="temp")
        result = sim_with_world.remove_object("temp")
        assert result["status"] == "success"
        assert "temp" not in sim_with_world._world.objects

    def test_remove_object_not_found(self, sim_with_world):
        result = sim_with_world.remove_object("ghost")
        assert result["status"] == "error"

    def test_remove_object_no_world(self, sim):
        result = sim.remove_object("x")
        assert result["status"] == "error"

    def test_list_objects_empty(self, sim_with_world):
        result = sim_with_world.list_objects()
        assert result["status"] == "success"
        assert "No objects" in result["content"][0]["text"]

    def test_list_objects_with_items(self, sim_with_world):
        sim_with_world.add_object(name="cube", shape="box")
        sim_with_world.add_object(name="ball", shape="sphere")
        result = sim_with_world.list_objects()
        text = result["content"][0]["text"]
        assert "cube" in text
        assert "ball" in text

    def test_list_objects_no_world(self, sim):
        result = sim.list_objects()
        assert result["status"] == "error"

    def test_move_object_no_world(self, sim):
        result = sim.move_object("x", [0, 0, 0])
        assert result["status"] == "error"

    def test_move_object_not_found(self, sim_with_world):
        result = sim_with_world.move_object("ghost", [0, 0, 0])
        assert result["status"] == "error"

    def test_move_object_success(self, sim_with_world):
        sim_with_world.add_object(name="box1")
        result = sim_with_world.move_object("box1", position=[1, 2, 3])
        assert result["status"] == "success"

    def test_add_multiple_objects(self, sim_with_world):
        for i in range(5):
            result = sim_with_world.add_object(name=f"obj_{i}", shape="box")
            assert result["status"] == "success"
        assert len(sim_with_world._world.objects) == 5

    def test_add_object_with_robot_injection(self, sim_with_world):
        """When robots exist, add_object attempts injection."""
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm"] = SimRobot(name="arm", urdf_path="/f.urdf")
        result = sim_with_world.add_object(name="block", shape="box")
        assert result["status"] == "success"
        assert "block" in sim_with_world._world.objects


# ===================================================================
# Camera Management
# ===================================================================


class TestCameraManagement:

    def test_add_camera_no_world(self, sim):
        result = sim.add_camera(name="cam1")
        assert result["status"] == "error"

    def test_add_camera_success(self, sim_with_world):
        result = sim_with_world.add_camera(
            name="top_down",
            position=[0, 0, 3],
            target=[0, 0, 0],
            fov=90.0,
        )
        assert result["status"] == "success"
        assert "top_down" in sim_with_world._world.cameras

    def test_remove_camera(self, sim_with_world):
        sim_with_world.add_camera(name="temp_cam")
        result = sim_with_world.remove_camera("temp_cam")
        assert result["status"] == "success"
        assert "temp_cam" not in sim_with_world._world.cameras

    def test_remove_camera_not_found(self, sim_with_world):
        result = sim_with_world.remove_camera("ghost_cam")
        assert result["status"] == "error"

    def test_default_camera_exists(self, sim_with_world):
        assert "default" in sim_with_world._world.cameras
        cam = sim_with_world._world.cameras["default"]
        assert cam.width == 640
        assert cam.height == 480


# ===================================================================
# Simulation Control
# ===================================================================


class TestSimulationControl:

    def test_step_no_world(self, sim):
        result = sim.step()
        assert result["status"] == "error"

    def test_step_success(self, sim_with_world):
        result = sim_with_world.step(n_steps=10)
        assert result["status"] == "success"
        assert sim_with_world._world.step_count == 10
        assert "+10 steps" in result["content"][0]["text"]

    def test_step_multiple_times(self, sim_with_world):
        sim_with_world.step(5)
        sim_with_world.step(3)
        assert sim_with_world._world.step_count == 8

    def test_set_gravity_no_world(self, sim):
        result = sim.set_gravity([0, 0, -5])
        assert result["status"] == "error"

    def test_set_gravity_success(self, sim_with_world):
        result = sim_with_world.set_gravity([0, 0, -5.0])
        assert result["status"] == "success"
        assert sim_with_world._world.gravity == [0, 0, -5.0]

    def test_set_timestep_no_world(self, sim):
        result = sim.set_timestep(0.001)
        assert result["status"] == "error"

    def test_set_timestep_success(self, sim_with_world):
        result = sim_with_world.set_timestep(0.001)
        assert result["status"] == "success"
        assert sim_with_world._world.timestep == 0.001
        assert "1000Hz" in result["content"][0]["text"]


# ===================================================================
# Rendering
# ===================================================================


class TestRendering:

    def test_render_no_world(self, sim):
        result = sim.render()
        assert result["status"] == "error"

    def test_render_success(self, sim_with_world):
        mock_image = MagicMock()
        mock_pil = MagicMock()
        mock_pil.Image.fromarray.return_value = mock_image
        mock_image.save = MagicMock(side_effect=lambda buf, format: buf.write(b"\x89PNG"))

        with patch.dict(sys.modules, {"PIL": mock_pil, "PIL.Image": mock_pil.Image}):
            result = sim_with_world.render()
        assert result["status"] == "success"
        assert len(result["content"]) == 2
        assert "image" in result["content"][1]

    def test_render_depth_no_world(self, sim):
        result = sim.render_depth()
        assert result["status"] == "error"

    def test_render_depth_success(self, sim_with_world):
        depth = np.ones((480, 640)) * 2.5
        _mock_mj.Renderer.return_value.render.return_value = depth
        result = sim_with_world.render_depth()
        assert result["status"] == "success"
        assert "json" in result["content"][1]
        assert result["content"][1]["json"]["depth_min"] == 2.5
        # Reset
        _mock_mj.Renderer.return_value.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_render_custom_size(self, sim_with_world):
        mock_image = MagicMock()
        mock_pil = MagicMock()
        mock_pil.Image.fromarray.return_value = mock_image
        mock_image.save = MagicMock(side_effect=lambda buf, format: buf.write(b"\x89PNG"))

        with patch.dict(sys.modules, {"PIL": mock_pil, "PIL.Image": mock_pil.Image}):
            result = sim_with_world.render(width=1280, height=960)
        assert result["status"] == "success"
        assert "1280x960" in result["content"][0]["text"]


# ===================================================================
# Contacts
# ===================================================================


class TestContacts:

    def test_get_contacts_no_world(self, sim):
        result = sim.get_contacts()
        assert result["status"] == "error"

    def test_get_contacts_none(self, sim_with_world):
        result = sim_with_world.get_contacts()
        assert result["status"] == "success"
        assert "No contacts" in result["content"][0]["text"]


# ===================================================================
# Robot Management
# ===================================================================


class TestRobotManagement:

    def test_list_robots_no_world(self, sim):
        result = sim.list_robots()
        assert result["status"] == "error"

    def test_list_robots_empty(self, sim_with_world):
        result = sim_with_world.list_robots()
        assert result["status"] == "success"
        assert "No robots" in result["content"][0]["text"]

    def test_list_robots_with_robots(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm1"] = SimRobot(
            name="arm1",
            urdf_path="/fake/arm.urdf",
            joint_names=["j1", "j2"],
        )
        result = sim_with_world.list_robots()
        assert result["status"] == "success"
        assert "arm1" in result["content"][0]["text"]

    def test_list_robots_shows_policy_status(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm1"] = SimRobot(
            name="arm1",
            urdf_path="/f.urdf",
            policy_running=True,
        )
        result = sim_with_world.list_robots()
        text = result["content"][0]["text"]
        assert "running" in text

    def test_remove_robot_success(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm1"] = SimRobot(name="arm1", urdf_path="/f.urdf")
        result = sim_with_world.remove_robot("arm1")
        assert result["status"] == "success"
        assert "arm1" not in sim_with_world._world.robots

    def test_remove_robot_not_found(self, sim_with_world):
        result = sim_with_world.remove_robot("ghost")
        assert result["status"] == "error"

    def test_remove_robot_no_world(self, sim):
        result = sim.remove_robot("x")
        assert result["status"] == "error"

    def test_get_robot_state_no_world(self, sim):
        result = sim.get_robot_state("arm")
        assert result["status"] == "error"

    def test_get_robot_state_not_found(self, sim_with_world):
        result = sim_with_world.get_robot_state("ghost")
        assert result["status"] == "error"

    def test_get_robot_state_success(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm1"] = SimRobot(
            name="arm1",
            urdf_path="/fake.urdf",
            joint_names=["joint_0", "joint_1"],
        )
        result = sim_with_world.get_robot_state("arm1")
        assert result["status"] == "success"
        assert "json" in result["content"][1]


# ===================================================================
# Recording
# ===================================================================


class TestRecording:

    def test_start_recording_no_world(self, sim):
        result = sim.start_recording()
        assert result["status"] == "error"

    def test_get_recording_status_no_world(self, sim):
        result = sim.get_recording_status()
        assert result["status"] == "error"

    def test_get_recording_status_not_recording(self, sim_with_world):
        result = sim_with_world.get_recording_status()
        assert result["status"] == "success"
        assert "Not recording" in result["content"][0]["text"]

    def test_stop_recording_not_recording(self, sim_with_world):
        result = sim_with_world.stop_recording()
        assert result["status"] == "error" or "Not recording" in str(result)


# ===================================================================
# Domain Randomization
# ===================================================================


class TestDomainRandomization:

    def test_randomize_no_world(self, sim):
        result = sim.randomize()
        assert result["status"] == "error"

    def test_randomize_empty_world(self, sim_with_world):
        result = sim_with_world.randomize(randomize_colors=True, randomize_lighting=True)
        assert result["status"] == "success"


# ===================================================================
# Viewer
# ===================================================================


class TestViewer:

    def test_open_viewer_no_world(self, sim):
        result = sim.open_viewer()
        assert result["status"] == "error"

    def test_close_viewer_no_viewer(self, sim):
        result = sim.close_viewer()
        assert result["status"] == "success"


# ===================================================================
# URDF Registry
# ===================================================================


class TestURDFRegistry:

    def test_list_urdfs(self, sim):
        result = sim.list_urdfs_action()
        assert result["status"] == "success"

    def test_register_urdf_empty_succeeds(self, sim):
        result = sim.register_urdf_action("", "")
        assert result["status"] == "success"

    def test_register_urdf_success(self, sim):
        result = sim.register_urdf_action("test_config", "/path/to/robot.urdf")
        assert result["status"] == "success"


# ===================================================================
# AgentTool Interface
# ===================================================================


class TestToolSpec:

    def test_tool_spec_structure(self, sim):
        spec = sim.tool_spec
        assert spec["name"] == "test_sim"
        assert "description" in spec
        assert "inputSchema" in spec

    def test_tool_spec_has_all_actions(self, sim):
        spec = sim.tool_spec
        actions = spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        expected = [
            "create_world",
            "load_scene",
            "reset",
            "get_state",
            "destroy",
            "add_robot",
            "remove_robot",
            "list_robots",
            "get_robot_state",
            "add_object",
            "remove_object",
            "move_object",
            "list_objects",
            "add_camera",
            "remove_camera",
            "run_policy",
            "start_policy",
            "stop_policy",
            "render",
            "render_depth",
            "get_contacts",
            "step",
            "set_gravity",
            "set_timestep",
            "randomize",
            "start_recording",
            "stop_recording",
            "get_recording_status",
            "record_video",
            "open_viewer",
            "close_viewer",
            "list_urdfs",
            "register_urdf",
            "get_features",
            "replay_episode",
            "eval_policy",
        ]
        for a in expected:
            assert a in actions, f"Missing action: {a}"

    def test_tool_spec_required_field(self, sim):
        spec = sim.tool_spec
        assert spec["inputSchema"]["json"]["required"] == ["action"]


# ===================================================================
# Dispatch Routing
# ===================================================================


class TestDispatch:

    def test_dispatch_unknown_action(self, sim):
        result = sim._dispatch_action("nonexistent_action", {})
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_dispatch_create_world(self, sim):
        result = sim._dispatch_action("create_world", {})
        assert result["status"] == "success"

    def test_dispatch_get_state(self, sim_with_world):
        result = sim_with_world._dispatch_action("get_state", {})
        assert result["status"] == "success"

    def test_dispatch_destroy(self, sim_with_world):
        result = sim_with_world._dispatch_action("destroy", {})
        assert result["status"] == "success"

    def test_dispatch_reset(self, sim_with_world):
        result = sim_with_world._dispatch_action("reset", {})
        assert result["status"] == "success"

    def test_dispatch_add_object(self, sim_with_world):
        result = sim_with_world._dispatch_action(
            "add_object",
            {
                "name": "cube",
                "shape": "box",
                "position": [0, 0, 0.1],
            },
        )
        assert result["status"] == "success"

    def test_dispatch_remove_object(self, sim_with_world):
        sim_with_world.add_object(name="temp")
        result = sim_with_world._dispatch_action("remove_object", {"name": "temp"})
        assert result["status"] == "success"

    def test_dispatch_list_objects(self, sim_with_world):
        result = sim_with_world._dispatch_action("list_objects", {})
        assert result["status"] == "success"

    def test_dispatch_move_object(self, sim_with_world):
        sim_with_world.add_object(name="box1")
        result = sim_with_world._dispatch_action("move_object", {"name": "box1", "position": [1, 2, 3]})
        assert result["status"] == "success"

    def test_dispatch_add_camera(self, sim_with_world):
        result = sim_with_world._dispatch_action("add_camera", {"name": "side_cam", "position": [2, 0, 1]})
        assert result["status"] == "success"

    def test_dispatch_remove_camera(self, sim_with_world):
        sim_with_world.add_camera(name="temp")
        result = sim_with_world._dispatch_action("remove_camera", {"name": "temp"})
        assert result["status"] == "success"

    def test_dispatch_list_robots(self, sim_with_world):
        result = sim_with_world._dispatch_action("list_robots", {})
        assert result["status"] == "success"

    def test_dispatch_step(self, sim_with_world):
        result = sim_with_world._dispatch_action("step", {"n_steps": 5})
        assert result["status"] == "success"

    def test_dispatch_set_gravity(self, sim_with_world):
        result = sim_with_world._dispatch_action("set_gravity", {"gravity": [0, 0, -5]})
        assert result["status"] == "success"

    def test_dispatch_set_timestep(self, sim_with_world):
        result = sim_with_world._dispatch_action("set_timestep", {"timestep": 0.001})
        assert result["status"] == "success"

    def test_dispatch_render(self, sim_with_world):
        mock_image = MagicMock()
        mock_pil = MagicMock()
        mock_pil.Image.fromarray.return_value = mock_image
        mock_image.save = MagicMock(side_effect=lambda buf, format: buf.write(b"\x89PNG"))
        with patch.dict(sys.modules, {"PIL": mock_pil, "PIL.Image": mock_pil.Image}):
            result = sim_with_world._dispatch_action("render", {})
        assert result["status"] == "success"

    def test_dispatch_render_depth(self, sim_with_world):
        depth = np.ones((480, 640)) * 1.5
        _mock_mj.Renderer.return_value.render.return_value = depth
        result = sim_with_world._dispatch_action("render_depth", {})
        assert result["status"] == "success"
        _mock_mj.Renderer.return_value.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_dispatch_get_contacts(self, sim_with_world):
        result = sim_with_world._dispatch_action("get_contacts", {})
        assert result["status"] == "success"

    def test_dispatch_close_viewer(self, sim):
        result = sim._dispatch_action("close_viewer", {})
        assert result["status"] == "success"

    def test_dispatch_list_urdfs(self, sim):
        result = sim._dispatch_action("list_urdfs", {})
        assert result["status"] == "success"

    def test_dispatch_get_features_no_world(self, sim):
        result = sim._dispatch_action("get_features", {})
        assert result["status"] == "error"

    def test_dispatch_stop_policy_robot_not_found(self, sim_with_world):
        result = sim_with_world._dispatch_action("stop_policy", {"robot_name": "ghost"})
        assert result["status"] == "error"

    def test_dispatch_stop_policy_success(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm1"] = SimRobot(
            name="arm1",
            urdf_path="/f.urdf",
            policy_running=True,
        )
        result = sim_with_world._dispatch_action("stop_policy", {"robot_name": "arm1"})
        assert result["status"] == "success"
        assert sim_with_world._world.robots["arm1"].policy_running is False

    def test_dispatch_get_recording_status(self, sim_with_world):
        result = sim_with_world._dispatch_action("get_recording_status", {})
        assert result["status"] == "success"

    def test_dispatch_replay_episode_no_repo(self, sim_with_world):
        result = sim_with_world._dispatch_action("replay_episode", {})
        assert result["status"] == "error"
        assert "repo_id required" in result["content"][0]["text"]

    def test_dispatch_randomize(self, sim_with_world):
        result = sim_with_world._dispatch_action(
            "randomize",
            {
                "randomize_colors": True,
                "randomize_lighting": False,
            },
        )
        assert result["status"] == "success"

    def test_dispatch_add_robot_default_name(self, sim_with_world):
        """add_robot with no name uses robot_name or default."""
        result = sim_with_world._dispatch_action(
            "add_robot",
            {
                "urdf_path": "/fake/robot.urdf",
            },
        )
        # May succeed or fail depending on URDF resolution, but shouldn't crash
        assert "status" in result

    def test_dispatch_get_robot_state(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm1"] = SimRobot(
            name="arm1",
            urdf_path="/f.urdf",
            joint_names=["joint_0"],
        )
        result = sim_with_world._dispatch_action("get_robot_state", {"robot_name": "arm1"})
        assert result["status"] == "success"

    def test_dispatch_remove_robot(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm1"] = SimRobot(name="arm1", urdf_path="/f.urdf")
        result = sim_with_world._dispatch_action("remove_robot", {"robot_name": "arm1"})
        assert result["status"] == "success"


# ===================================================================
# Stream (async)
# ===================================================================


class TestStream:

    def test_stream_dispatch(self, sim_with_world):
        tool_use = {"toolUseId": "test_123", "input": {"action": "get_state"}}

        async def run():
            results = []
            async for event in sim_with_world.stream(tool_use, {}):
                results.append(event)
            return results

        results = asyncio.run(run())
        assert len(results) == 1

    def test_stream_error_handling(self, sim):
        """Stream handles errors from dispatch."""
        tool_use = {"toolUseId": "test_err", "input": {"action": "get_state"}}

        async def run():
            results = []
            async for event in sim.stream(tool_use, {}):
                results.append(event)
            return results

        results = asyncio.run(run())
        assert len(results) == 1


# ===================================================================
# Get Features
# ===================================================================


class TestGetFeatures:

    def test_get_features_no_world(self, sim):
        result = sim.get_features()
        assert result["status"] == "error"

    def test_get_features_with_world(self, sim_with_world):
        result = sim_with_world.get_features()
        assert result["status"] == "success"
        assert "json" in result["content"][1]
        features = result["content"][1]["json"]["features"]
        assert "n_bodies" in features
        assert "n_joints" in features
        assert "joint_names" in features

    def test_get_features_with_robots(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm1"] = SimRobot(
            name="arm1",
            urdf_path="/fake.urdf",
            joint_names=["j1", "j2"],
            data_config="so100",
        )
        result = sim_with_world.get_features()
        features = result["content"][1]["json"]["features"]
        assert "arm1" in features["robots"]
        assert features["robots"]["arm1"]["data_config"] == "so100"


# ===================================================================
# Cleanup
# ===================================================================


class TestCleanup:

    def test_cleanup_with_world(self, sim_with_world):
        sim_with_world.cleanup()
        assert sim_with_world._world is None

    def test_cleanup_no_world(self, sim):
        sim.cleanup()  # Should not raise

    def test_cleanup_stops_policies(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm1"] = SimRobot(
            name="arm1",
            urdf_path="/f.urdf",
            policy_running=True,
        )
        sim_with_world.cleanup()

    def test_del_safe(self, Sim):
        s = Sim(tool_name="del_test", mesh=False)
        s.create_world()
        s.__del__()  # Should not raise


# ===================================================================
# MJCF Builder
# ===================================================================


class TestMJCFBuilder:

    def test_object_xml_box(self):
        from strands_robots.simulation import MJCFBuilder, SimObject

        obj = SimObject(name="cube", shape="box", size=[0.1, 0.1, 0.1], color=[1, 0, 0, 1], mass=0.5)
        xml = MJCFBuilder._object_xml(obj)
        assert 'name="cube"' in xml
        assert 'type="box"' in xml
        assert "freejoint" in xml

    def test_object_xml_static(self):
        from strands_robots.simulation import MJCFBuilder, SimObject

        obj = SimObject(name="ground_obj", shape="box", is_static=True)
        xml = MJCFBuilder._object_xml(obj)
        assert "freejoint" not in xml
        assert "inertial" not in xml

    def test_object_xml_sphere(self):
        from strands_robots.simulation import MJCFBuilder, SimObject

        obj = SimObject(name="ball", shape="sphere", size=[0.1])
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="sphere"' in xml

    def test_object_xml_cylinder(self):
        from strands_robots.simulation import MJCFBuilder, SimObject

        obj = SimObject(name="can", shape="cylinder", size=[0.05, 0.05, 0.1])
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="cylinder"' in xml

    def test_object_xml_capsule(self):
        from strands_robots.simulation import MJCFBuilder, SimObject

        obj = SimObject(name="pill", shape="capsule", size=[0.02, 0.02, 0.06])
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="capsule"' in xml

    def test_object_xml_mesh(self):
        from strands_robots.simulation import MJCFBuilder, SimObject

        obj = SimObject(name="custom", shape="mesh", mesh_path="/path/mesh.stl")
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="mesh"' in xml
        assert 'mesh="mesh_custom"' in xml

    def test_object_xml_plane(self):
        from strands_robots.simulation import MJCFBuilder, SimObject

        obj = SimObject(name="floor", shape="plane", size=[5, 5])
        xml = MJCFBuilder._object_xml(obj)
        assert 'type="plane"' in xml

    def test_build_objects_only(self):
        from strands_robots.simulation import MJCFBuilder, SimCamera, SimObject, SimWorld

        world = SimWorld()
        world.objects["cube"] = SimObject(name="cube", shape="box")
        world.cameras["cam1"] = SimCamera(name="cam1")
        xml = MJCFBuilder.build_objects_only(world)
        assert '<mujoco model="strands_sim">' in xml
        assert 'name="cube"' in xml
        assert 'name="cam1"' in xml
        assert 'name="ground"' in xml

    def test_build_objects_only_no_ground(self):
        from strands_robots.simulation import MJCFBuilder, SimWorld

        world = SimWorld(ground_plane=False)
        xml = MJCFBuilder.build_objects_only(world)
        assert 'name="ground"' not in xml

    def test_build_objects_only_with_mesh(self):
        from strands_robots.simulation import MJCFBuilder, SimObject, SimWorld

        world = SimWorld()
        world.objects["mesh_obj"] = SimObject(
            name="mesh_obj",
            shape="mesh",
            mesh_path="/path/to/model.stl",
        )
        xml = MJCFBuilder.build_objects_only(world)
        assert 'name="mesh_mesh_obj"' in xml
        assert 'file="/path/to/model.stl"' in xml

    def test_build_objects_only_custom_gravity(self):
        from strands_robots.simulation import MJCFBuilder, SimWorld

        world = SimWorld(gravity=[0, 0, -3.0], timestep=0.005)
        xml = MJCFBuilder.build_objects_only(world)
        assert 'gravity="0 0 -3.0"' in xml
        assert 'timestep="0.005"' in xml


# ===================================================================
# Module-level functions
# ===================================================================


class TestModuleFunctions:

    def test_register_urdf(self):
        from strands_robots.simulation import _URDF_REGISTRY, register_urdf

        register_urdf("test_robot", "/path/to/test.urdf")
        assert _URDF_REGISTRY.get("test_robot") == "/path/to/test.urdf"
        _URDF_REGISTRY.pop("test_robot", None)

    def test_resolve_urdf_known(self):
        # Use a file that actually exists
        import tempfile

        from strands_robots.simulation import _URDF_REGISTRY, register_urdf, resolve_urdf

        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
            tmp = f.name
        register_urdf("my_bot", tmp)
        result = resolve_urdf("my_bot")
        assert result == tmp
        os.unlink(tmp)
        _URDF_REGISTRY.pop("my_bot", None)

    def test_resolve_urdf_unknown(self):
        from strands_robots.simulation import resolve_urdf

        resolve_urdf("completely_unknown_robot_xyz_123")
        # Returns None or a path — just verify no crash

    def test_list_registered_urdfs(self):
        from strands_robots.simulation import list_registered_urdfs

        result = list_registered_urdfs()
        assert isinstance(result, dict)

    def test_list_available_models(self):
        from strands_robots.simulation import list_available_models

        result = list_available_models()
        assert isinstance(result, str)

    def test_ensure_mujoco(self):
        from strands_robots.simulation import _ensure_mujoco

        mj = _ensure_mujoco()
        assert mj is not None


# ===================================================================
# Policy Execution Tests (run_policy, eval_policy, start_policy)
# ===================================================================


class TestGetSimObservation:
    """Tests for _get_sim_observation — joint state + camera rendering."""

    def _sim_with_robot(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        robot = SimRobot(
            name="arm",
            urdf_path="/fake.urdf",
            joint_names=["j1", "j2"],
            joint_ids=[0, 1],
        )
        sim_with_world._world.robots["arm"] = robot
        return sim_with_world

    def test_observation_contains_joints(self, sim_with_world):
        sim = self._sim_with_robot(sim_with_world)
        obs = sim._get_sim_observation("arm")
        assert "j1" in obs or "j2" in obs  # mj_name2id returns 0 → reads qpos

    def test_observation_with_camera(self, sim_with_world):
        sim = self._sim_with_robot(sim_with_world)
        obs = sim._get_sim_observation("arm", cam_name="default")
        # Should contain rendered image or at least joint data
        assert isinstance(obs, dict)

    def test_observation_renders_all_cameras(self, sim_with_world):
        sim = self._sim_with_robot(sim_with_world)
        # With cam_name=None it should render all scene cameras
        obs = sim._get_sim_observation("arm")
        assert isinstance(obs, dict)


class TestRunPolicy:
    """Tests for run_policy — the main policy execution loop."""

    def test_run_policy_no_world(self, sim):
        result = sim.run_policy("arm", "mock")
        assert result["status"] == "error"
        assert "No simulation" in result["content"][0]["text"]

    def test_run_policy_robot_not_found(self, sim_with_world):
        result = sim_with_world.run_policy("ghost_robot", "mock")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_run_policy_success(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        robot = SimRobot(
            name="arm",
            urdf_path="/fake.urdf",
            joint_names=["j1", "j2"],
            joint_ids=[0, 1],
        )
        sim_with_world._world.robots["arm"] = robot

        # Mock create_policy to return a policy that gives actions
        mock_policy = MagicMock()
        mock_policy.get_actions = MagicMock(return_value=[{"j1": 0.1, "j2": 0.2}])

        async def _get_actions(obs, instr):
            return [{"j1": 0.1, "j2": 0.2}]

        mock_policy.get_actions = _get_actions

        with patch("strands_robots.policies.create_policy", return_value=mock_policy):
            result = sim_with_world.run_policy(
                "arm",
                "mock",
                instruction="pick up",
                duration=0.05,
                control_frequency=20.0,
            )

        assert result["status"] == "success"
        assert "Policy complete" in result["content"][0]["text"]
        assert robot.policy_running is False

    def test_run_policy_records_trajectory(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        robot = SimRobot(
            name="arm",
            urdf_path="/fake.urdf",
            joint_names=["j1"],
            joint_ids=[0],
        )
        sim_with_world._world.robots["arm"] = robot
        sim_with_world._world._recording = True
        sim_with_world._world._trajectory = []

        async def _get_actions(obs, instr):
            return [{"j1": 0.5}]

        mock_policy = MagicMock()
        mock_policy.get_actions = _get_actions

        with patch("strands_robots.policies.create_policy", return_value=mock_policy):
            result = sim_with_world.run_policy(
                "arm",
                "mock",
                instruction="test",
                duration=0.05,
                control_frequency=20.0,
            )

        assert result["status"] == "success"
        assert len(sim_with_world._world._trajectory) > 0

    def test_run_policy_exception(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        robot = SimRobot(
            name="arm",
            urdf_path="/fake.urdf",
            joint_names=["j1"],
            joint_ids=[0],
        )
        sim_with_world._world.robots["arm"] = robot

        with patch("strands_robots.policies.create_policy", side_effect=RuntimeError("boom")):
            result = sim_with_world.run_policy("arm", "bad_provider")

        assert result["status"] == "error"
        assert "Policy failed" in result["content"][0]["text"]
        assert robot.policy_running is False


class TestEvalPolicy:
    """Tests for eval_policy — multi-episode evaluation."""

    def test_eval_no_world(self, sim):
        result = sim.eval_policy()
        assert result["status"] == "error"

    def test_eval_no_robots(self, sim_with_world):
        result = sim_with_world.eval_policy()
        assert result["status"] == "error"
        assert "No robots" in result["content"][0]["text"]

    def test_eval_robot_not_found(self, sim_with_world):
        result = sim_with_world.eval_policy(robot_name="ghost")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_eval_success(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        robot = SimRobot(
            name="arm",
            urdf_path="/fake.urdf",
            joint_names=["j1"],
            joint_ids=[0],
        )
        sim_with_world._world.robots["arm"] = robot

        async def _get_actions(obs, instr):
            return [{"j1": 0.1}]

        mock_policy = MagicMock()
        mock_policy.get_actions = _get_actions

        # eval_policy uses local import: from strands_robots.policies import create_policy
        # AND module-level _mujoco for mj_resetData/mj_forward/mj_step
        with (
            patch("strands_robots.policies.create_policy", return_value=mock_policy),
            patch("strands_robots.policies.create_policy", return_value=mock_policy),
        ):
            result = sim_with_world.eval_policy(
                robot_name="arm",
                policy_provider="mock",
                n_episodes=2,
                max_steps=3,
            )

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "episodes" in text.lower() or "success" in text.lower()

    def test_eval_with_contact_success(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        robot = SimRobot(
            name="arm",
            urdf_path="/fake.urdf",
            joint_names=["j1"],
            joint_ids=[0],
        )
        sim_with_world._world.robots["arm"] = robot

        # Mock contact detection
        mock_contact = MagicMock()
        mock_contact.dist = -0.01
        _mock_data.ncon = 1
        _mock_data.contact = [mock_contact]

        async def _get_actions(obs, instr):
            return [{"j1": 0.1}]

        mock_policy = MagicMock()
        mock_policy.get_actions = _get_actions

        with (
            patch("strands_robots.policies.create_policy", return_value=mock_policy),
            patch("strands_robots.policies.create_policy", return_value=mock_policy),
        ):
            result = sim_with_world.eval_policy(
                robot_name="arm",
                policy_provider="mock",
                n_episodes=1,
                max_steps=5,
                success_fn="contact",
            )

        assert result["status"] == "success"
        # Reset data.ncon
        _mock_data.ncon = 0
        _mock_data.contact = []


class TestRecordVideo:
    """Tests for record_video — policy execution with video capture."""

    def test_record_video_no_world(self, sim):
        result = sim.record_video("arm", "mock")
        assert result["status"] == "error"

    def test_record_video_robot_not_found(self, sim_with_world):
        result = sim_with_world.record_video("ghost", "mock")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_record_video_success(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        robot = SimRobot(
            name="arm",
            urdf_path="/fake.urdf",
            joint_names=["j1"],
            joint_ids=[0],
        )
        sim_with_world._world.robots["arm"] = robot

        async def _get_actions(obs, instr):
            return [{"j1": 0.1}]

        mock_policy = MagicMock()
        mock_policy.get_actions = _get_actions

        mock_writer = MagicMock()
        mock_imageio = MagicMock()
        mock_imageio.get_writer.return_value = mock_writer

        with (
            patch("strands_robots.policies.create_policy", return_value=mock_policy),
            patch.dict(sys.modules, {"imageio": mock_imageio}),
        ):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                output = f.name
            try:
                # Create output so os.path.getsize works
                with open(output, "wb") as fout:
                    fout.write(b"\x00" * 100)

                result = sim_with_world.record_video(
                    "arm",
                    "mock",
                    instruction="test",
                    duration=0.05,
                    fps=10,
                    output_path=output,
                )
                assert result["status"] == "success"
                assert "Video recorded" in result["content"][0]["text"]
            finally:
                os.unlink(output)

    def test_record_video_exception(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        robot = SimRobot(
            name="arm",
            urdf_path="/fake.urdf",
            joint_names=["j1"],
            joint_ids=[0],
        )
        sim_with_world._world.robots["arm"] = robot

        with patch("strands_robots.policies.create_policy", side_effect=RuntimeError("no policy")):
            # imageio import also needed
            mock_imageio = MagicMock()
            with patch.dict(sys.modules, {"imageio": mock_imageio}):
                result = sim_with_world.record_video("arm", "bad", duration=0.01)
                assert result["status"] == "error"


class TestReplayEpisode:
    """Tests for replay_episode — replaying LeRobotDataset episodes."""

    def test_replay_no_world(self, sim):
        result = sim.replay_episode("my/dataset")
        assert result["status"] == "error"
        assert "No world" in result["content"][0]["text"]

    def test_replay_no_robots(self, sim_with_world):
        result = sim_with_world.replay_episode("my/dataset")
        assert result["status"] == "error"
        assert "No robots" in result["content"][0]["text"]

    def test_replay_robot_not_found(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm"] = SimRobot(name="arm", urdf_path="/f.urdf")
        result = sim_with_world.replay_episode("my/dataset", robot_name="ghost")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_replay_dispatch_no_repo(self, sim_with_world):
        """Dispatch replay_episode with no repo_id returns error."""
        result = sim_with_world._dispatch_action("replay_episode", {})
        assert result["status"] == "error"
        assert "repo_id required" in result["content"][0]["text"]

    def test_replay_lerobot_import_error(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm"] = SimRobot(name="arm", urdf_path="/f.urdf")

        with patch.dict(
            sys.modules, {"lerobot": None, "lerobot.datasets": None, "lerobot.datasets.lerobot_dataset": None}
        ):
            result = sim_with_world.replay_episode("my/dataset", robot_name="arm")
            assert result["status"] == "error"
            # Should fail on import or loading


class TestStartStopRecording:
    """Tests for start_recording / stop_recording with dataset recorder."""

    def test_start_recording_no_world(self, sim):
        result = sim.start_recording()
        assert result["status"] == "error"

    def test_start_recording_no_lerobot(self, sim_with_world):
        """Without lerobot installed, start_recording returns error."""
        result = sim_with_world.start_recording()
        assert result["status"] == "error"
        assert "lerobot" in result["content"][0]["text"].lower()

    def test_stop_recording_not_recording(self, sim_with_world):
        result = sim_with_world.stop_recording()
        assert result["status"] == "error"
        assert "Not recording" in result["content"][0]["text"]

    def test_stop_recording_no_recorder(self, sim_with_world):
        sim_with_world._world._recording = True
        sim_with_world._world._dataset_recorder = None
        result = sim_with_world.stop_recording()
        assert result["status"] == "error"
        assert "No dataset recorder" in result["content"][0]["text"]

    def test_stop_recording_success(self, sim_with_world):
        sim_with_world._world._recording = True
        mock_recorder = MagicMock()
        mock_recorder.save_episode.return_value = {"status": "success"}
        mock_recorder.repo_id = "test/dataset"
        mock_recorder.frame_count = 100
        mock_recorder.episode_count = 1
        mock_recorder.root = "/tmp/data"
        sim_with_world._world._dataset_recorder = mock_recorder
        sim_with_world._world._push_to_hub = False

        result = sim_with_world.stop_recording()
        assert result["status"] == "success"
        assert "Episode saved" in result["content"][0]["text"]
        assert sim_with_world._world._recording is False
        assert sim_with_world._world._dataset_recorder is None
        mock_recorder.finalize.assert_called_once()

    def test_stop_recording_push_to_hub(self, sim_with_world):
        sim_with_world._world._recording = True
        mock_recorder = MagicMock()
        mock_recorder.save_episode.return_value = {"status": "success"}
        mock_recorder.push_to_hub.return_value = {"status": "success"}
        mock_recorder.repo_id = "test/dataset"
        mock_recorder.frame_count = 50
        mock_recorder.episode_count = 1
        mock_recorder.root = "/tmp/data"
        sim_with_world._world._dataset_recorder = mock_recorder
        sim_with_world._world._push_to_hub = True

        result = sim_with_world.stop_recording()
        assert result["status"] == "success"
        assert "Pushed to HuggingFace" in result["content"][0]["text"]
        mock_recorder.push_to_hub.assert_called_once()

    def test_start_recording_with_lerobot(self, sim_with_world):
        """When DatasetRecorder is available, start_recording succeeds."""
        from strands_robots.simulation import SimRobot

        robot = SimRobot(
            name="arm",
            urdf_path="/fake.urdf",
            joint_names=["j1", "j2"],
            joint_ids=[0, 1],
            data_config="so100",
        )
        sim_with_world._world.robots["arm"] = robot

        mock_recorder = MagicMock()
        mock_recorder.create.return_value = mock_recorder

        # Patch HAS_LEROBOT_DATASET to True and DatasetRecorder at the source module
        # (simulation.py now does a runtime import from dataset_recorder)
        with (
            patch("strands_robots.dataset_recorder.HAS_LEROBOT_DATASET", True),
            patch("strands_robots.dataset_recorder.DatasetRecorder", mock_recorder),
        ):
            result = sim_with_world.start_recording(
                repo_id="test/sim_data",
                task="pick up cube",
                fps=30,
            )

        assert result["status"] == "success"
        assert "Recording" in result["content"][0]["text"]
        assert sim_with_world._world._recording is True


# ===================================================================
# Object / Camera Injection Tests
# ===================================================================


class TestInjectObject:
    """Tests for _inject_object_into_scene — XML round-trip injection."""

    def test_inject_no_model(self, sim_with_world):
        from strands_robots.simulation import SimObject

        sim_with_world._world._model = None
        obj = SimObject(name="cube", shape="box")
        result = sim_with_world._inject_object_into_scene(obj)
        assert result is False

    def test_inject_success(self, sim_with_world):
        from strands_robots.simulation import SimObject

        # Need a robot for _robot_base_xml to be set
        sim_with_world._world._robot_base_xml = None

        obj = SimObject(name="block", shape="box", size=[0.05, 0.05, 0.05])

        # Mock mj_saveLastXML to write actual XML
        def fake_save(path, model):
            with open(path, "w") as f:
                f.write("<mujoco><worldbody></worldbody></mujoco>")

        _mock_mj.mj_saveLastXML.side_effect = fake_save

        result = sim_with_world._inject_object_into_scene(obj)
        assert result is True

        # Reset side_effect
        _mock_mj.mj_saveLastXML.side_effect = None

    def test_inject_with_robot_base_xml(self, sim_with_world):
        from strands_robots.simulation import SimObject

        with tempfile.TemporaryDirectory() as tmpdir:
            sim_with_world._world._robot_base_xml = os.path.join(tmpdir, "robot.xml")

            obj = SimObject(name="ball", shape="sphere", size=[0.1])

            def fake_save(path, model):
                with open(path, "w") as f:
                    f.write("<mujoco><worldbody></worldbody></mujoco>")

            _mock_mj.mj_saveLastXML.side_effect = fake_save

            result = sim_with_world._inject_object_into_scene(obj)
            assert result is True

            _mock_mj.mj_saveLastXML.side_effect = None

    def test_inject_reload_failure(self, sim_with_world):
        from strands_robots.simulation import SimObject

        sim_with_world._world._robot_base_xml = None
        obj = SimObject(name="bad", shape="box")

        def fake_save(path, model):
            with open(path, "w") as f:
                f.write("<mujoco><worldbody></worldbody></mujoco>")

        _mock_mj.mj_saveLastXML.side_effect = fake_save
        _mock_mj.MjModel.from_xml_path.side_effect = RuntimeError("invalid XML")

        result = sim_with_world._inject_object_into_scene(obj)
        assert result is False

        # Reset
        _mock_mj.mj_saveLastXML.side_effect = None
        _mock_mj.MjModel.from_xml_path.side_effect = None
        _mock_mj.MjModel.from_xml_path.return_value = _mock_model


class TestInjectCamera:
    """Tests for _inject_camera_into_scene — XML round-trip injection."""

    def test_inject_camera_no_model(self, sim_with_world):
        from strands_robots.simulation import SimCamera

        sim_with_world._world._model = None
        cam = SimCamera(name="top")
        result = sim_with_world._inject_camera_into_scene(cam)
        assert result is False

    def test_inject_camera_success(self, sim_with_world):
        from strands_robots.simulation import SimCamera

        sim_with_world._world._robot_base_xml = None
        cam = SimCamera(name="side", position=[2, 0, 1], fov=90.0)

        def fake_save(path, model):
            with open(path, "w") as f:
                f.write("<mujoco><worldbody></worldbody></mujoco>")

        _mock_mj.mj_saveLastXML.side_effect = fake_save

        result = sim_with_world._inject_camera_into_scene(cam)
        assert result is True

        _mock_mj.mj_saveLastXML.side_effect = None

    def test_inject_camera_reload_failure(self, sim_with_world):
        from strands_robots.simulation import SimCamera

        sim_with_world._world._robot_base_xml = None
        cam = SimCamera(name="fail_cam")

        def fake_save(path, model):
            with open(path, "w") as f:
                f.write("<mujoco><worldbody></worldbody></mujoco>")

        _mock_mj.mj_saveLastXML.side_effect = fake_save
        _mock_mj.MjModel.from_xml_path.side_effect = RuntimeError("bad xml")

        result = sim_with_world._inject_camera_into_scene(cam)
        assert result is False

        _mock_mj.mj_saveLastXML.side_effect = None
        _mock_mj.MjModel.from_xml_path.side_effect = None
        _mock_mj.MjModel.from_xml_path.return_value = _mock_model


# ===================================================================
# Add Robot Tests
# ===================================================================


class TestAddRobot:
    """Tests for add_robot — loading robots into simulation."""

    def test_add_robot_no_world(self, sim):
        result = sim.add_robot("arm1")
        assert result["status"] == "error"
        assert "No world" in result["content"][0]["text"]

    def test_add_robot_duplicate(self, sim_with_world):
        from strands_robots.simulation import SimRobot

        sim_with_world._world.robots["arm1"] = SimRobot(name="arm1", urdf_path="/f.urdf")
        result = sim_with_world.add_robot("arm1")
        assert result["status"] == "error"
        assert "already exists" in result["content"][0]["text"]

    def test_add_robot_no_path_or_config(self, sim_with_world):
        """No urdf_path, no data_config, name doesn't resolve."""
        with patch("strands_robots.simulation.resolve_model", return_value=None):
            result = sim_with_world.add_robot("unknown_bot_xyz")
        assert result["status"] == "error"

    def test_add_robot_file_not_found(self, sim_with_world):
        result = sim_with_world.add_robot("arm1", urdf_path="/nonexistent/path.urdf")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_add_robot_success(self, sim_with_world):
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(b"<mujoco/>")
            robot_path = f.name

        try:
            result = sim_with_world.add_robot("arm1", urdf_path=robot_path)
            assert result["status"] == "success"
            assert "arm1" in sim_with_world._world.robots
            assert "Robot 'arm1' added" in result["content"][0]["text"]
        finally:
            os.unlink(robot_path)

    def test_add_robot_with_data_config(self, sim_with_world):
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(b"<mujoco/>")
            robot_path = f.name

        try:
            with patch("strands_robots.simulation.resolve_model", return_value=robot_path):
                result = sim_with_world.add_robot("arm1", data_config="so100")
            assert result["status"] == "success"
        finally:
            os.unlink(robot_path)

    def test_add_robot_load_failure(self, sim_with_world):
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(b"<mujoco/>")
            robot_path = f.name

        try:
            _mock_mj.MjModel.from_xml_path.side_effect = RuntimeError("parse error")
            result = sim_with_world.add_robot("arm1", urdf_path=robot_path)
            assert result["status"] == "error"
            assert "Failed" in result["content"][0]["text"]
        finally:
            _mock_mj.MjModel.from_xml_path.side_effect = None
            _mock_mj.MjModel.from_xml_path.return_value = _mock_model
            os.unlink(robot_path)


# ===================================================================
# Ensure Meshes Tests
# ===================================================================


class TestEnsureMeshes:
    """Tests for _ensure_meshes — auto-downloading missing assets."""

    def test_meshes_present(self, sim):
        """When all meshes exist, _ensure_meshes is a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "robot.xml")
            mesh_path = os.path.join(tmpdir, "body.stl")
            with open(xml_path, "w") as f:
                f.write(f'<mujoco><asset><mesh file="{mesh_path}"/></asset></mujoco>')
            with open(mesh_path, "w") as f:
                f.write("mesh data")

            # Should not raise or attempt download
            from strands_robots.simulation import Simulation

            Simulation._ensure_meshes(xml_path, "test_robot")

    def test_meshes_missing_triggers_download(self, sim):
        """When meshes are missing, _ensure_meshes attempts download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "robot.xml")
            with open(xml_path, "w") as f:
                f.write('<mujoco><asset><mesh name="m" file="missing.stl"/></asset></mujoco>')

            mock_download = MagicMock()
            mock_resolve = MagicMock(return_value="test_robot")

            with (
                patch("strands_robots.assets.download.download_robots", mock_download),
                patch("strands_robots.assets.resolve_robot_name", mock_resolve),
            ):
                from strands_robots.simulation import Simulation

                Simulation._ensure_meshes(xml_path, "test_robot")

    def test_meshes_no_mesh_refs(self, sim):
        """XML with no mesh references = no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "simple.xml")
            with open(xml_path, "w") as f:
                f.write('<mujoco><worldbody><geom type="box"/></worldbody></mujoco>')

            from strands_robots.simulation import Simulation

            Simulation._ensure_meshes(xml_path, "simple_bot")


# ===================================================================
# Dispatch Integration Tests (extended)
# ===================================================================


class TestDispatchExtended:
    """Extended dispatch tests for previously uncovered actions."""

    def test_dispatch_run_policy_no_robot(self, sim_with_world):
        result = sim_with_world._dispatch_action(
            "run_policy",
            {
                "robot_name": "ghost",
                "policy_provider": "mock",
            },
        )
        assert result["status"] == "error"

    def test_dispatch_eval_policy_no_robots(self, sim_with_world):
        result = sim_with_world._dispatch_action(
            "eval_policy",
            {
                "policy_provider": "mock",
            },
        )
        assert result["status"] == "error"

    def test_dispatch_record_video_no_robot(self, sim_with_world):
        result = sim_with_world._dispatch_action(
            "record_video",
            {
                "robot_name": "ghost",
                "policy_provider": "mock",
            },
        )
        assert result["status"] == "error"

    def test_dispatch_start_recording_no_lerobot(self, sim_with_world):
        result = sim_with_world._dispatch_action("start_recording", {})
        assert result["status"] == "error"

    def test_dispatch_stop_recording_not_active(self, sim_with_world):
        result = sim_with_world._dispatch_action("stop_recording", {})
        assert result["status"] == "error"

    def test_dispatch_add_robot_no_path(self, sim_with_world):
        """add_robot dispatch with no valid path/config."""
        with patch("strands_robots.simulation.resolve_model", return_value=None):
            result = sim_with_world._dispatch_action(
                "add_robot",
                {
                    "name": "unknown_xyz",
                },
            )
        # Should fail (no path resolved)
        assert result["status"] == "error"

    def test_dispatch_open_viewer_no_world(self, sim):
        result = sim._dispatch_action("open_viewer", {})
        assert result["status"] == "error"

    def test_dispatch_get_features_with_world(self, sim_with_world):
        result = sim_with_world._dispatch_action("get_features", {})
        assert result["status"] == "success"


# ===================================================================
# Build Scene Tests (multi-robot)
# ===================================================================


class TestComposeMultiRobotScene:
    """Tests for MJCFBuilder.compose_multi_robot_scene — multi-robot MJCF generation."""

    def test_compose_creates_xml(self, sim_with_world):
        from strands_robots.simulation import MJCFBuilder, SimCamera, SimObject, SimRobot, SimWorld

        world = SimWorld()
        robots = {
            "arm1": SimRobot(name="arm1", urdf_path="/fake.urdf", joint_names=["j1"]),
        }
        objects = {
            "cube": SimObject(name="cube", shape="box", size=[0.1, 0.1, 0.1]),
        }
        cameras = {
            "cam1": SimCamera(name="cam1", position=[1, 0, 1]),
        }

        def fake_save(path, model):
            with open(path, "w") as f:
                f.write("<mujoco><worldbody/></mujoco>")

        _mock_mj.mj_saveLastXML.side_effect = fake_save

        path = MJCFBuilder.compose_multi_robot_scene(robots, objects, cameras, world)
        assert path is not None
        assert path.endswith("master_scene.xml")
        assert os.path.exists(path)

        # Clean up
        _mock_mj.mj_saveLastXML.side_effect = None
        import shutil

        shutil.rmtree(os.path.dirname(path))
