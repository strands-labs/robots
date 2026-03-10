#!/usr/bin/env python3
"""Comprehensive tests for strands_robots.tools — serial_tool, robot_mesh, pose_tool, teleoperator.

All hardware-dependent operations are mocked. Target: tools/ 6-13% → 60%+.

Key pattern: serial_tool, robot_mesh, pose_tool, and teleoperator all do
top-level `from strands import tool` and/or `import serial`. We mock these
in sys.modules via autouse fixtures + importlib.import_module to force
fresh imports each time.
"""

import importlib
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if tools PR (#11) not merged yet
try:
    import strands_robots.tools.robot_mesh  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("Requires PR #11 (agent-tools)", allow_module_level=True)

# --- Module-level mocks for heavy deps ---
_mock_strands = MagicMock()
_mock_strands.tool = lambda f: f  # passthrough decorator

_mock_serial = MagicMock()
# Wire up hierarchy so serial.tools.list_ports.comports() works
_mock_serial.tools.list_ports.comports.return_value = []
_mock_serial.SerialException = type("SerialException", (Exception,), {})


@pytest.fixture(autouse=True)
def _mock_heavy_deps(monkeypatch):
    """Mock strands and serial in sys.modules so tools can be imported."""
    monkeypatch.setitem(sys.modules, "strands", _mock_strands)
    monkeypatch.setitem(sys.modules, "serial", _mock_serial)
    monkeypatch.setitem(sys.modules, "serial.tools", _mock_serial.tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", _mock_serial.tools.list_ports)

    # Reset side effects between tests
    _mock_serial.Serial.side_effect = None
    _mock_serial.Serial.reset_mock()
    _mock_serial.tools.list_ports.comports.return_value = []

    # Clear cached modules to force reimport with mocks
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("strands_robots.tools."):
            del sys.modules[mod_name]


@pytest.fixture
def serial_mod(_mock_heavy_deps):
    """Import serial_tool fresh with mocked deps."""
    return importlib.import_module("strands_robots.tools.serial_tool")


@pytest.fixture
def mesh_mod(_mock_heavy_deps):
    """Import robot_mesh fresh with mocked deps."""
    return importlib.import_module("strands_robots.tools.robot_mesh")


@pytest.fixture
def pose_mod(_mock_heavy_deps):
    """Import pose_tool fresh with mocked deps."""
    return importlib.import_module("strands_robots.tools.pose_tool")


@pytest.fixture
def teleop_mod(_mock_heavy_deps):
    """Import teleoperator fresh with mocked deps."""
    return importlib.import_module("strands_robots.tools.teleoperator")


# ═════════════════════════════════════════════════════════════════════════════
# serial_tool Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestSerialToolListPorts:
    """Test serial_tool list_ports action."""

    def test_list_ports_empty(self, serial_mod):
        _mock_serial.tools.list_ports.comports.return_value = []
        result = serial_mod.serial_tool(action="list_ports")
        assert result["status"] == "success"
        assert "0 serial ports" in result["content"][0]["text"]
        assert result["ports"] == []

    def test_list_ports_with_devices(self, serial_mod):
        port = MagicMock()
        port.device = "/dev/ttyACM0"
        port.name = "ttyACM0"
        port.description = "Feetech Motor Controller"
        port.manufacturer = "FEETECH"
        port.vid = 0x1234
        port.pid = 0x5678
        port.serial_number = "SN12345"
        _mock_serial.tools.list_ports.comports.return_value = [port]

        result = serial_mod.serial_tool(action="list_ports")
        assert result["status"] == "success"
        assert "1 serial ports" in result["content"][0]["text"]
        assert len(result["ports"]) == 1
        assert result["ports"][0]["device"] == "/dev/ttyACM0"


class TestSerialToolSend:
    """Test serial_tool send action."""

    def test_send_no_port(self, serial_mod):
        result = serial_mod.serial_tool(action="send")
        assert result["status"] == "error"
        assert "Port" in result["content"][0]["text"]

    def test_send_string_data(self, serial_mod):
        mock_ser = MagicMock()
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="send", port="/dev/ttyACM0", data="hello")
        assert result["status"] == "success"
        mock_ser.write.assert_called_once_with(b"hello")
        mock_ser.close.assert_called_once()

    def test_send_hex_data(self, serial_mod):
        mock_ser = MagicMock()
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="send", port="/dev/ttyACM0", hex_data="FF FF 01 04")
        assert result["status"] == "success"
        mock_ser.write.assert_called_once_with(bytes.fromhex("FFFF0104"))

    def test_send_no_data(self, serial_mod):
        mock_ser = MagicMock()
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="send", port="/dev/ttyACM0")
        assert result["status"] == "error"
        assert "No data" in result["content"][0]["text"]


class TestSerialToolRead:
    def test_read_data(self, serial_mod):
        mock_ser = MagicMock()
        mock_ser.read.return_value = b"\x48\x65\x6c\x6c\x6f"
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="read", port="/dev/ttyACM0")
        assert result["status"] == "success"
        assert "5 bytes" in result["content"][0]["text"]
        assert "Hello" in result["content"][0]["text"]  # ASCII rendering


class TestSerialToolSendRead:
    def test_send_read_string(self, serial_mod):
        mock_ser = MagicMock()
        mock_ser.read.return_value = b"\x4f\x4b"
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="send_read", port="/dev/ttyACM0", data="status")
        assert result["status"] == "success"
        assert "OK" in result["content"][0]["text"]

    def test_send_read_hex(self, serial_mod):
        mock_ser = MagicMock()
        mock_ser.read.return_value = b"\xff\x01"
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="send_read", port="/dev/ttyACM0", hex_data="FF FF 01")
        assert result["status"] == "success"

    def test_send_read_no_data(self, serial_mod):
        mock_ser = MagicMock()
        _mock_serial.Serial.return_value = mock_ser
        result = serial_mod.serial_tool(action="send_read", port="/dev/ttyACM0")
        assert result["status"] == "error"


class TestSerialToolFeetech:
    def test_feetech_position(self, serial_mod):
        mock_ser = MagicMock()
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="feetech_position", port="/dev/ttyACM0", motor_id=1, position=2048)
        assert result["status"] == "success"
        assert "Motor 1" in result["content"][0]["text"]
        assert "Position 2048" in result["content"][0]["text"]
        mock_ser.write.assert_called_once()

    def test_feetech_position_missing_params(self, serial_mod):
        mock_ser = MagicMock()
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="feetech_position", port="/dev/ttyACM0", motor_id=1)
        assert result["status"] == "error"

    def test_feetech_velocity(self, serial_mod):
        mock_ser = MagicMock()
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="feetech_velocity", port="/dev/ttyACM0", motor_id=2, velocity=500)
        assert result["status"] == "success"
        assert "Motor 2" in result["content"][0]["text"]

    def test_feetech_velocity_missing_params(self, serial_mod):
        mock_ser = MagicMock()
        _mock_serial.Serial.return_value = mock_ser
        result = serial_mod.serial_tool(action="feetech_velocity", port="/dev/ttyACM0", motor_id=2)
        assert result["status"] == "error"

    def test_feetech_ping_success(self, serial_mod):
        mock_ser = MagicMock()
        mock_ser.read.return_value = b"\xff\xff\x01\x02\x00\xfc\x00\x00\x00\x00"
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="feetech_ping", port="/dev/ttyACM0", motor_id=1)
        assert result["status"] == "success"
        assert "responded" in result["content"][0]["text"]

    def test_feetech_ping_no_response(self, serial_mod):
        mock_ser = MagicMock()
        mock_ser.read.return_value = b""
        _mock_serial.Serial.return_value = mock_ser

        result = serial_mod.serial_tool(action="feetech_ping", port="/dev/ttyACM0", motor_id=1)
        assert result["status"] == "error"
        assert "no response" in result["content"][0]["text"]

    def test_feetech_ping_no_motor_id(self, serial_mod):
        mock_ser = MagicMock()
        _mock_serial.Serial.return_value = mock_ser
        result = serial_mod.serial_tool(action="feetech_ping", port="/dev/ttyACM0")
        assert result["status"] == "error"


class TestSerialToolMonitor:
    def test_monitor(self, serial_mod):
        mock_ser = MagicMock()
        mock_ser.in_waiting = 0  # No data available
        _mock_serial.Serial.return_value = mock_ser

        # Patch time.time to make the monitor exit quickly
        start = time.time()
        with patch.object(serial_mod, "time", create=True) as mock_time:
            mock_time.time.side_effect = [start, start + 6.0]  # Instant timeout
            mock_time.sleep = time.sleep
            result = serial_mod.serial_tool(action="monitor", port="/dev/ttyACM0")
        assert result["status"] == "success"
        assert "Monitored" in result["content"][0]["text"]


class TestSerialToolErrors:
    def test_unknown_action(self, serial_mod):
        result = serial_mod.serial_tool(action="invalid_action", port="/dev/ttyACM0")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_serial_exception(self, serial_mod):
        # Use the pre-defined SerialException from our mock
        _mock_serial.Serial.side_effect = _mock_serial.SerialException("port busy")
        result = serial_mod.serial_tool(action="read", port="/dev/ttyACM0")
        assert result["status"] == "error"
        assert "error" in result["content"][0]["text"].lower() or "Serial" in result["content"][0]["text"]
        _mock_serial.Serial.side_effect = None


# ═════════════════════════════════════════════════════════════════════════════
# robot_mesh Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRobotMeshPeers:
    def test_peers_empty(self, mesh_mod):
        mock_zenoh = MagicMock()
        mock_zenoh.get_peers.return_value = []
        mock_zenoh._LOCAL_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="peers")
        assert result["status"] == "success"
        assert "No peers" in result["content"][0]["text"]

    def test_peers_with_local_and_remote(self, mesh_mod):
        local_robot = MagicMock()
        local_robot.peer_type = "robot"
        local_robots = {"so100-abc": local_robot}
        remote_peers = [
            {
                "peer_id": "reachy-xyz",
                "type": "robot",
                "hostname": "rpi4",
                "age": 5,
                "task_status": "running",
                "instruction": "pick up cube",
            },
        ]

        mock_zenoh = MagicMock()
        mock_zenoh.get_peers.return_value = remote_peers
        mock_zenoh._LOCAL_ROBOTS = local_robots

        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="peers")
        assert result["status"] == "success"
        assert "so100-abc" in result["content"][0]["text"]
        assert "reachy-xyz" in result["content"][0]["text"]


class TestRobotMeshTell:
    def test_tell_no_target(self, mesh_mod):
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"r1": MagicMock()}
        mock_zenoh.get_peers.return_value = []
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="tell")
        assert result["status"] == "error"
        assert "target" in result["content"][0]["text"]

    def test_tell_no_mesh(self, mesh_mod):
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {}
        mock_zenoh.get_peers.return_value = []
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="tell", target="r1", instruction="go")
        assert result["status"] == "error"
        assert "No local robots" in result["content"][0]["text"]

    def test_tell_success(self, mesh_mod):
        mesh_instance = MagicMock()
        mesh_instance.send.return_value = {"status": "ok"}
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"local1": mesh_instance}
        mock_zenoh.get_peers.return_value = []

        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="tell", target="remote1", instruction="pick up cube")
        assert result["status"] == "success"


class TestRobotMeshSend:
    def test_send_no_target(self, mesh_mod):
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"r1": MagicMock()}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="send")
        assert result["status"] == "error"

    def test_send_success(self, mesh_mod):
        mesh_instance = MagicMock()
        mesh_instance.send.return_value = {"status": "ok"}
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"local1": mesh_instance}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="send", target="r2", command='{"action": "status"}')
        assert result["status"] == "success"


class TestRobotMeshBroadcast:
    def test_broadcast_success(self, mesh_mod):
        mesh_instance = MagicMock()
        mesh_instance.broadcast.return_value = [{"peer": "r1", "status": "ok"}]
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"local1": mesh_instance}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="broadcast")
        assert result["status"] == "success"
        assert "1 responses" in result["content"][0]["text"]


class TestRobotMeshStop:
    def test_stop_success(self, mesh_mod):
        mesh_instance = MagicMock()
        mesh_instance.send.return_value = {"stopped": True}
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"local1": mesh_instance}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="stop", target="r1")
        assert result["status"] == "success"

    def test_stop_no_target(self, mesh_mod):
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"r1": MagicMock()}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="stop")
        assert result["status"] == "error"


class TestRobotMeshEmergencyStop:
    def test_estop(self, mesh_mod):
        mesh_instance = MagicMock()
        mesh_instance.emergency_stop.return_value = [{"peer": "r1"}, {"peer": "r2"}]
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"local1": mesh_instance}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="emergency_stop")
        assert result["status"] == "success"
        assert "2 responses" in result["content"][0]["text"]


class TestRobotMeshStatus:
    def test_status(self, mesh_mod):
        mesh_instance = MagicMock()
        mesh_instance.peer_type = "robot"
        mesh_instance.alive = True
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"local1": mesh_instance}
        mock_zenoh.get_peers.return_value = []
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="status")
        assert result["status"] == "success"
        assert "local1" in result["content"][0]["text"]


class TestRobotMeshSubscribe:
    def test_subscribe_no_target(self, mesh_mod):
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"r1": MagicMock()}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="subscribe")
        assert result["status"] == "error"

    def test_subscribe_success(self, mesh_mod):
        mesh_instance = MagicMock()
        mesh_instance.subscribe.return_value = "reachy_mini/*"
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"local1": mesh_instance}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="subscribe", target="reachy_mini/*")
        assert result["status"] == "success"
        assert "Subscribed" in result["content"][0]["text"]


class TestRobotMeshWatch:
    def test_watch_no_target(self, mesh_mod):
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"r1": MagicMock()}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="watch")
        assert result["status"] == "error"

    def test_watch_success(self, mesh_mod):
        mesh_instance = MagicMock()
        mesh_instance.on_stream.return_value = "vla_stream_r1"
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"local1": mesh_instance}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="watch", target="r1")
        assert result["status"] == "success"
        assert "Watching" in result["content"][0]["text"]


class TestRobotMeshInbox:
    def test_inbox(self, mesh_mod):
        mesh_instance = MagicMock()
        mesh_instance.inbox = {"reachy_mini/*": [(time.time(), {"pos": [0.1, 0.2]})]}
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {"local1": mesh_instance}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="inbox")
        assert result["status"] == "success"
        assert "1 subscriptions" in result["content"][0]["text"]


class TestRobotMeshUnknownAction:
    def test_unknown_action(self, mesh_mod):
        mock_zenoh = MagicMock()
        mock_zenoh._LOCAL_ROBOTS = {}
        mock_zenoh.get_peers.return_value = []
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_zenoh}):
            result = mesh_mod.robot_mesh(action="invalid_action")
        assert result["status"] == "error"

    def test_import_error(self, mesh_mod):
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": None}):
            result = mesh_mod.robot_mesh(action="peers")
        assert result["status"] == "error"
        assert "unavailable" in result["content"][0]["text"].lower() or "error" in result["content"][0]["text"].lower()


# ═════════════════════════════════════════════════════════════════════════════
# pose_tool Tests (PoseManager, MotorController, pose_tool)
# ═════════════════════════════════════════════════════════════════════════════


class TestPoseManager:
    @pytest.fixture
    def pm(self, tmp_path, pose_mod):
        return pose_mod.PoseManager("test_robot", storage_dir=tmp_path)

    def test_init(self, pm):
        assert pm.robot_id == "test_robot"
        assert pm.poses == {}

    def test_store_and_get_pose(self, pm):
        pose = pm.store_pose("home", {"shoulder": 0.0, "elbow": 0.0}, "Home position")
        assert pose.name == "home"
        assert pm.get_pose("home") is not None
        assert pm.get_pose("home").description == "Home position"

    def test_list_poses(self, pm):
        pm.store_pose("home", {"shoulder": 0.0})
        pm.store_pose("pick", {"shoulder": 45.0})
        assert set(pm.list_poses()) == {"home", "pick"}

    def test_delete_pose(self, pm):
        pm.store_pose("home", {"shoulder": 0.0})
        assert pm.delete_pose("home") is True
        assert pm.get_pose("home") is None
        assert pm.delete_pose("nonexistent") is False

    def test_persistence(self, tmp_path, pose_mod):
        pm1 = pose_mod.PoseManager("test_robot", storage_dir=tmp_path)
        pm1.store_pose("home", {"shoulder": 0.0}, "Home")

        pm2 = pose_mod.PoseManager("test_robot", storage_dir=tmp_path)
        assert "home" in pm2.list_poses()
        assert pm2.get_pose("home").description == "Home"

    def test_validate_pose_no_bounds(self, pm, pose_mod):
        pose = pose_mod.RobotPose(name="test", positions={"shoulder": 0.0}, timestamp=time.time())
        valid, msg = pm.validate_pose(pose)
        assert valid is True

    def test_validate_pose_within_bounds(self, pm, pose_mod):
        pose = pose_mod.RobotPose(
            name="test",
            positions={"shoulder": 45.0},
            timestamp=time.time(),
            safety_bounds={"shoulder": (-90.0, 90.0)},
        )
        valid, msg = pm.validate_pose(pose)
        assert valid is True

    def test_validate_pose_out_of_bounds(self, pm, pose_mod):
        pose = pose_mod.RobotPose(
            name="test",
            positions={"shoulder": 100.0},
            timestamp=time.time(),
            safety_bounds={"shoulder": (-90.0, 90.0)},
        )
        valid, msg = pm.validate_pose(pose)
        assert valid is False
        assert "outside bounds" in msg


class TestRobotPose:
    def test_to_dict(self, pose_mod):
        pose = pose_mod.RobotPose(name="test", positions={"shoulder": 0.0}, timestamp=123.4)
        d = pose.to_dict()
        assert d["name"] == "test"
        assert d["timestamp"] == 123.4

    def test_from_dict(self, pose_mod):
        d = {
            "name": "test",
            "positions": {"shoulder": 0.0},
            "timestamp": 123.4,
            "description": None,
            "safety_bounds": None,
        }
        pose = pose_mod.RobotPose.from_dict(d)
        assert pose.name == "test"


class TestMotorController:
    @pytest.fixture
    def ctrl(self, pose_mod):
        return pose_mod.MotorController("/dev/ttyACM0")

    def test_degrees_to_position_regular(self, ctrl):
        # shoulder_pan: range (-180, 180), resolution 4095
        pos = ctrl.degrees_to_position("shoulder_pan", 0.0)
        assert pos == 2047 or pos == 2048  # Midpoint

    def test_degrees_to_position_gripper(self, ctrl):
        pos = ctrl.degrees_to_position("gripper", 50.0)
        assert pos == 2047 or pos == 2048

    def test_degrees_to_position_unknown_motor(self, ctrl):
        with pytest.raises(ValueError, match="Unknown motor"):
            ctrl.degrees_to_position("unknown_motor", 0.0)

    def test_degrees_to_position_clamping(self, ctrl):
        # Beyond max range should clamp
        pos_max = ctrl.degrees_to_position("shoulder_pan", 999.0)
        assert pos_max == 4095

    def test_position_to_degrees_regular(self, ctrl):
        # 0 position should be -180 for shoulder_pan
        deg = ctrl.position_to_degrees("shoulder_pan", 0)
        assert deg == -180.0

    def test_position_to_degrees_gripper(self, ctrl):
        deg = ctrl.position_to_degrees("gripper", 4095)
        assert abs(deg - 100.0) < 0.1

    def test_position_to_degrees_unknown(self, ctrl):
        with pytest.raises(ValueError):
            ctrl.position_to_degrees("unknown", 0)

    def test_connect(self, ctrl):
        _mock_serial.Serial.return_value = MagicMock()
        _mock_serial.Serial.side_effect = None
        ok, err = ctrl.connect()
        assert ok is True
        assert err == ""

    def test_connect_fail(self, ctrl):
        _mock_serial.Serial.side_effect = Exception("port busy")
        ok, err = ctrl.connect()
        assert ok is False
        assert "port busy" in err
        _mock_serial.Serial.side_effect = None

    def test_disconnect(self, ctrl):
        ctrl.serial_conn = MagicMock()
        ctrl.serial_conn.is_open = True
        ctrl.disconnect()
        ctrl.serial_conn.close.assert_called_once()

    def test_disconnect_not_connected(self, ctrl):
        ctrl.serial_conn = None
        ctrl.disconnect()  # Should not raise

    def test_build_feetech_packet(self, ctrl):
        packet = ctrl.build_feetech_packet(1, 0x01, [])  # Ping
        assert packet[0:2] == b"\xff\xff"
        assert packet[2] == 1  # motor_id

    def test_move_motor(self, ctrl):
        ctrl.serial_conn = MagicMock()
        ctrl.serial_conn.is_open = True
        result = ctrl.move_motor("shoulder_pan", 0.0)
        assert result is True
        ctrl.serial_conn.write.assert_called_once()

    def test_move_motor_not_connected(self, ctrl):
        ctrl.serial_conn = None
        assert ctrl.move_motor("shoulder_pan", 0.0) is False

    def test_read_motor_position(self, ctrl):
        ctrl.serial_conn = MagicMock()
        ctrl.serial_conn.is_open = True
        # Simulate response: header + id + len + err + pos_low + pos_high + checksum
        ctrl.serial_conn.read.return_value = b"\xff\xff\x01\x04\x00\x00\x08\xf2"
        pos = ctrl.read_motor_position("shoulder_pan")
        assert pos is not None

    def test_read_motor_not_connected(self, ctrl):
        ctrl.serial_conn = None
        assert ctrl.read_motor_position("shoulder_pan") is None

    def test_read_all_positions(self, ctrl):
        ctrl.serial_conn = MagicMock()
        ctrl.serial_conn.is_open = True
        ctrl.serial_conn.read.return_value = b"\xff\xff\x01\x04\x00\x00\x08\xf2"
        positions = ctrl.read_all_positions()
        assert isinstance(positions, dict)

    def test_incremental_move(self, ctrl):
        ctrl.serial_conn = MagicMock()
        ctrl.serial_conn.is_open = True
        ctrl.serial_conn.read.return_value = b"\xff\xff\x01\x04\x00\x00\x08\xf2"
        result = ctrl.incremental_move("shoulder_pan", 5.0)
        assert result is True

    def test_incremental_move_cant_read(self, ctrl):
        ctrl.serial_conn = MagicMock()
        ctrl.serial_conn.is_open = True
        ctrl.serial_conn.read.return_value = b""  # No response
        result = ctrl.incremental_move("shoulder_pan", 5.0)
        assert result is False


# ── pose_tool @tool function Tests ───────────────────────────────────────────


class TestPoseToolFunction:
    def test_list_poses_empty(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            pm = MagicMock()
            pm.list_poses.return_value = []
            MockPM.return_value = pm
            result = pose_mod.pose_tool(action="list_poses")
        assert result["status"] == "success"
        assert "No poses" in result["content"][0]["text"]

    def test_list_poses_with_data(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            pm = MagicMock()
            pm.list_poses.return_value = ["home"]
            pm.get_pose.return_value = pose_mod.RobotPose(
                name="home", positions={"shoulder": 0.0}, timestamp=time.time(), description="Home"
            )
            MockPM.return_value = pm
            result = pose_mod.pose_tool(action="list_poses")
        assert result["status"] == "success"
        assert "home" in result["content"][0]["text"]

    def test_show_pose(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            pm = MagicMock()
            pm.get_pose.return_value = pose_mod.RobotPose(
                name="home", positions={"shoulder": 0.0, "elbow": 45.0}, timestamp=1000.0, description="Home position"
            )
            MockPM.return_value = pm
            result = pose_mod.pose_tool(action="show_pose", pose_name="home")
        assert result["status"] == "success"
        assert "shoulder" in result["content"][0]["text"]

    def test_show_pose_not_found(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            pm = MagicMock()
            pm.get_pose.return_value = None
            MockPM.return_value = pm
            result = pose_mod.pose_tool(action="show_pose", pose_name="nope")
        assert result["status"] == "error"

    def test_show_pose_no_name(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            MockPM.return_value = MagicMock()
            result = pose_mod.pose_tool(action="show_pose")
        assert result["status"] == "error"

    def test_delete_pose(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            pm = MagicMock()
            pm.delete_pose.return_value = True
            MockPM.return_value = pm
            result = pose_mod.pose_tool(action="delete_pose", pose_name="home")
        assert result["status"] == "success"

    def test_delete_pose_not_found(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            pm = MagicMock()
            pm.delete_pose.return_value = False
            MockPM.return_value = pm
            result = pose_mod.pose_tool(action="delete_pose", pose_name="nope")
        assert result["status"] == "error"

    def test_connect_success(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            MockPM.return_value = MagicMock()
            with patch.object(pose_mod, "MotorController") as MockMC:
                mc = MagicMock()
                mc.connect.return_value = (True, "")
                MockMC.return_value = mc
                result = pose_mod.pose_tool(action="connect", port="/dev/ttyACM0")
        assert result["status"] == "success"

    def test_connect_failure(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            MockPM.return_value = MagicMock()
            with patch.object(pose_mod, "MotorController") as MockMC:
                mc = MagicMock()
                mc.connect.return_value = (False, "port busy")
                MockMC.return_value = mc
                result = pose_mod.pose_tool(action="connect", port="/dev/ttyACM0")
        assert result["status"] == "error"

    def test_no_port_for_motor_action(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            MockPM.return_value = MagicMock()
            result = pose_mod.pose_tool(action="connect", port=None)
        assert result["status"] == "error"
        assert "port required" in result["content"][0]["text"]

    def test_emergency_stop(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            MockPM.return_value = MagicMock()
            with patch.object(pose_mod, "MotorController") as MockMC:
                MockMC.return_value = MagicMock()
                result = pose_mod.pose_tool(action="emergency_stop", port="/dev/ttyACM0")
        assert result["status"] == "success"
        assert "Emergency stop" in result["content"][0]["text"]

    def test_read_position(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            MockPM.return_value = MagicMock()
            with patch.object(pose_mod, "MotorController") as MockMC:
                mc = MagicMock()
                mc.connect.return_value = (True, "")
                mc.read_motor_position.return_value = 45.0
                MockMC.return_value = mc
                result = pose_mod.pose_tool(
                    action="read_position",
                    port="/dev/ttyACM0",
                    motor_name="shoulder_pan",
                )
        assert result["status"] == "success"
        assert "45.0" in result["content"][0]["text"]

    def test_read_position_no_motor(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            MockPM.return_value = MagicMock()
            with patch.object(pose_mod, "MotorController") as MockMC:
                MockMC.return_value = MagicMock()
                result = pose_mod.pose_tool(action="read_position", port="/dev/ttyACM0")
        assert result["status"] == "error"

    def test_move_motor(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            MockPM.return_value = MagicMock()
            with patch.object(pose_mod, "MotorController") as MockMC:
                mc = MagicMock()
                mc.connect.return_value = (True, "")
                mc.move_motor.return_value = True
                MockMC.return_value = mc
                result = pose_mod.pose_tool(
                    action="move_motor",
                    port="/dev/ttyACM0",
                    motor_name="shoulder_pan",
                    position=45.0,
                )
        assert result["status"] == "success"

    def test_move_motor_no_params(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            MockPM.return_value = MagicMock()
            with patch.object(pose_mod, "MotorController") as MockMC:
                MockMC.return_value = MagicMock()
                result = pose_mod.pose_tool(action="move_motor", port="/dev/ttyACM0")
        assert result["status"] == "error"

    def test_reset_to_home(self, pose_mod):
        with patch.object(pose_mod, "PoseManager") as MockPM:
            MockPM.return_value = MagicMock()
            with patch.object(pose_mod, "MotorController") as MockMC:
                mc = MagicMock()
                mc.connect.return_value = (True, "")
                mc.move_multiple_motors.return_value = True
                MockMC.return_value = mc
                result = pose_mod.pose_tool(action="reset_to_home", port="/dev/ttyACM0")
        assert result["status"] == "success"
        assert "home" in result["content"][0]["text"].lower()


# ═════════════════════════════════════════════════════════════════════════════
# teleoperator Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestTeleoperatorTool:
    @pytest.fixture(autouse=True)
    def _reset_session(self, teleop_mod):
        """Reset global session state before each test."""
        teleop_mod._ACTIVE_SESSION.update(
            {
                "running": False,
                "thread": None,
                "robot": None,
                "teleop": None,
                "record_session": None,
                "mode": None,
                "stats": {},
            }
        )

    def test_status_no_session(self, teleop_mod):
        result = teleop_mod.teleoperator(action="status")
        assert result["status"] == "success"
        assert "No active session" in result["content"][0]["text"]

    def test_stop_no_session(self, teleop_mod):
        result = teleop_mod.teleoperator(action="stop")
        assert result["status"] == "success"
        assert "No active session" in result["content"][0]["text"]

    def test_discard_no_session(self, teleop_mod):
        result = teleop_mod.teleoperator(action="discard")
        assert result["status"] == "error"

    def test_unknown_action(self, teleop_mod):
        result = teleop_mod.teleoperator(action="unknown")
        assert result["status"] == "error"

    def test_teleop_already_running(self, teleop_mod):
        teleop_mod._ACTIVE_SESSION["running"] = True
        result = teleop_mod.teleoperator(action="teleop")
        assert result["status"] == "error"
        assert "already running" in result["content"][0]["text"]

    def test_record_already_running(self, teleop_mod):
        teleop_mod._ACTIVE_SESSION["running"] = True
        result = teleop_mod.teleoperator(action="record")
        assert result["status"] == "error"
        assert "already running" in result["content"][0]["text"]
