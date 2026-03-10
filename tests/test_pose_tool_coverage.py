"""Coverage-gap tests for strands_robots/tools/pose_tool.py.

Targets the 107 uncovered lines: MotorController serial operations,
pose_tool @tool actions that need a mocked serial connection (read_all,
store_pose, load_pose, move_multiple, incremental_move, reset_to_home failures,
read_position failures, unknown action), PoseManager error paths,
RobotPose.from_dict, validate_pose with safety bounds.

All serial I/O is mocked. CPU-only, no hardware required.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure strands is available (mock if not) ─────────────────────────────
try:
    import strands

    HAS_STRANDS = hasattr(strands, "Agent")
except ImportError:
    import types

    _mock_strands = types.ModuleType("strands")
    _mock_strands.tool = lambda f: f
    sys.modules["strands"] = _mock_strands
    HAS_STRANDS = False

# Mock serial before importing pose_tool
_mock_serial = MagicMock()
_mock_serial_tools = MagicMock()
_mock_serial_tools_list_ports = MagicMock()
sys.modules.setdefault("serial", _mock_serial)
sys.modules.setdefault("serial.tools", _mock_serial_tools)
sys.modules.setdefault("serial.tools.list_ports", _mock_serial_tools_list_ports)

from strands_robots.tools.pose_tool import (  # noqa: E402
    MotorController,
    PoseManager,
    RobotPose,
    pose_tool,
)

# ═══════════════════════════════════════════════════════════════════════════
# RobotPose unit tests — from_dict, to_dict round-trip, safety_bounds
# ═══════════════════════════════════════════════════════════════════════════


class TestRobotPose:
    def test_from_dict_round_trip(self):
        original = RobotPose(
            name="test",
            positions={"j1": 10.0, "j2": 20.0},
            timestamp=1234567890.0,
            description="Test pose",
            safety_bounds={"j1": (-90, 90)},
        )
        d = original.to_dict()
        restored = RobotPose.from_dict(d)
        assert restored.name == "test"
        assert restored.positions == {"j1": 10.0, "j2": 20.0}
        assert restored.description == "Test pose"
        assert restored.safety_bounds == {"j1": (-90, 90)}  # asdict preserves tuples

    def test_from_dict_minimal(self):
        d = {"name": "min", "positions": {"a": 0.0}, "timestamp": 0.0}
        pose = RobotPose.from_dict(d)
        assert pose.name == "min"
        assert pose.description is None
        assert pose.safety_bounds is None


# ═══════════════════════════════════════════════════════════════════════════
# PoseManager — error paths (_load_poses corrupt file, _save_poses failure)
# ═══════════════════════════════════════════════════════════════════════════


class TestPoseManagerErrors:
    def test_load_corrupt_json(self, tmp_path):
        """_load_poses with corrupt JSON falls back to empty dict."""
        pose_file = tmp_path / "bad_robot_poses.json"
        pose_file.write_text("{not valid json")
        pm = PoseManager("bad_robot", storage_dir=tmp_path)
        assert pm.poses == {}

    def test_save_failure(self, tmp_path):
        """_save_poses logs error on write failure."""
        pm = PoseManager("test", storage_dir=tmp_path)
        pm.poses["x"] = RobotPose(name="x", positions={"a": 1.0}, timestamp=0)
        # Make the file read-only directory to cause write failure
        with patch("builtins.open", side_effect=PermissionError("denied")):
            pm._save_poses()  # should not raise

    def test_validate_pose_out_of_bounds(self, tmp_path):
        pm = PoseManager("test", storage_dir=tmp_path)
        pose = RobotPose(
            name="bad",
            positions={"j1": 200.0},
            timestamp=0,
            safety_bounds={"j1": (-90, 90)},
        )
        valid, msg = pm.validate_pose(pose)
        assert valid is False
        assert "j1" in msg
        assert "200.0" in msg

    def test_validate_pose_within_bounds(self, tmp_path):
        pm = PoseManager("test", storage_dir=tmp_path)
        pose = RobotPose(
            name="ok",
            positions={"j1": 45.0},
            timestamp=0,
            safety_bounds={"j1": (-90, 90)},
        )
        valid, msg = pm.validate_pose(pose)
        assert valid is True

    def test_validate_pose_no_bounds(self, tmp_path):
        pm = PoseManager("test", storage_dir=tmp_path)
        pose = RobotPose(name="ok", positions={"j1": 9999.0}, timestamp=0)
        valid, msg = pm.validate_pose(pose)
        assert valid is True
        assert "No safety bounds" in msg


# ═══════════════════════════════════════════════════════════════════════════
# MotorController — pure logic (degrees_to_position, position_to_degrees,
# build_feetech_packet, connect/disconnect, move_motor, read_motor,
# read_all, smooth_move, incremental_move)
# ═══════════════════════════════════════════════════════════════════════════


class TestMotorControllerLogic:
    """Test MotorController without real serial."""

    def make_controller(self):
        return MotorController("/dev/fake", baudrate=1000000)

    def test_degrees_to_position_regular_joint(self):
        mc = self.make_controller()
        # shoulder_pan range: (-180, 180), mid = 0 → should be ~2048
        pos = mc.degrees_to_position("shoulder_pan", 0.0)
        assert pos == 2047 or pos == 2048  # midpoint of 4095

    def test_degrees_to_position_clamped(self):
        mc = self.make_controller()
        # Beyond range should clamp
        pos_max = mc.degrees_to_position("shoulder_pan", 999.0)
        pos_clamp = mc.degrees_to_position("shoulder_pan", 180.0)
        assert pos_max == pos_clamp

    def test_degrees_to_position_gripper(self):
        mc = self.make_controller()
        pos = mc.degrees_to_position("gripper", 50.0)
        assert pos == 2047 or pos == 2048  # 50% of 4095

    def test_degrees_to_position_unknown_motor(self):
        mc = self.make_controller()
        with pytest.raises(ValueError, match="Unknown motor"):
            mc.degrees_to_position("nonexistent", 0)

    def test_position_to_degrees_regular(self):
        mc = self.make_controller()
        # Full range: position 0 → -180°, position 4095 → 180°
        deg = mc.position_to_degrees("shoulder_pan", 0)
        assert deg == pytest.approx(-180.0)
        deg = mc.position_to_degrees("shoulder_pan", 4095)
        assert deg == pytest.approx(180.0)

    def test_position_to_degrees_gripper(self):
        mc = self.make_controller()
        deg = mc.position_to_degrees("gripper", 4095)
        assert deg == pytest.approx(100.0)

    def test_position_to_degrees_unknown_motor(self):
        mc = self.make_controller()
        with pytest.raises(ValueError, match="Unknown motor"):
            mc.position_to_degrees("nonexistent", 0)

    def test_build_feetech_packet(self):
        mc = self.make_controller()
        packet = mc.build_feetech_packet(1, 0x03, [0x2A, 0x00, 0x08])
        assert isinstance(packet, bytes)
        assert packet[0] == 0xFF
        assert packet[1] == 0xFF
        assert packet[2] == 1  # motor_id
        # Verify checksum
        checksum = (~sum(packet[2:-1])) & 0xFF
        assert packet[-1] == checksum

    def test_connect_success(self):
        mc = self.make_controller()
        mock_serial = MagicMock()
        with patch("strands_robots.tools.pose_tool.serial.Serial", return_value=mock_serial):
            ok, err = mc.connect()
        assert ok is True
        assert err == ""
        assert mc.serial_conn is mock_serial

    def test_connect_failure(self):
        mc = self.make_controller()
        with patch("strands_robots.tools.pose_tool.serial.Serial", side_effect=Exception("port busy")):
            ok, err = mc.connect()
        assert ok is False
        assert "port busy" in err

    def test_disconnect(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        mc.disconnect()
        mc.serial_conn.close.assert_called_once()

    def test_disconnect_not_connected(self):
        mc = self.make_controller()
        mc.serial_conn = None
        mc.disconnect()  # should not raise

    def test_move_motor_not_connected(self):
        mc = self.make_controller()
        mc.serial_conn = None
        result = mc.move_motor("shoulder_pan", 0.0)
        assert result is False

    def test_move_motor_connected(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        result = mc.move_motor("shoulder_pan", 45.0)
        assert result is True
        mc.serial_conn.write.assert_called_once()

    def test_move_motor_write_error(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        mc.serial_conn.write.side_effect = Exception("write error")
        result = mc.move_motor("shoulder_pan", 45.0)
        assert result is False

    def test_read_motor_position_not_connected(self):
        mc = self.make_controller()
        mc.serial_conn = None
        result = mc.read_motor_position("shoulder_pan")
        assert result is None

    def test_read_motor_position_short_response(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        mc.serial_conn.read.return_value = b"\x00\x01"  # too short
        result = mc.read_motor_position("shoulder_pan")
        assert result is None

    def test_read_motor_position_success(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        # Build a valid response: 7+ bytes, position at bytes 5-6
        # Position = 2048 → 0x00 0x08 (low, high)
        response = bytes([0xFF, 0xFF, 0x01, 0x04, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00])
        mc.serial_conn.read.return_value = response
        with patch("time.sleep"):
            result = mc.read_motor_position("shoulder_pan")
        assert result is not None
        assert isinstance(result, float)

    def test_read_motor_position_error(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        mc.serial_conn.write.side_effect = Exception("read error")
        result = mc.read_motor_position("shoulder_pan")
        assert result is None

    def test_read_all_positions(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        response = bytes([0xFF, 0xFF, 0x01, 0x04, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00])
        mc.serial_conn.read.return_value = response
        with patch("time.sleep"):
            positions = mc.read_all_positions()
        assert len(positions) == 6  # all motors
        for name in mc.motor_configs:
            assert name in positions

    def test_move_multiple_non_smooth(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        result = mc.move_multiple_motors({"shoulder_pan": 0.0, "gripper": 50.0}, smooth=False)
        assert result is True
        assert mc.serial_conn.write.call_count == 2

    def test_move_multiple_smooth(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        # read_all_positions needs valid responses
        response = bytes([0xFF, 0xFF, 0x01, 0x04, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00])
        mc.serial_conn.read.return_value = response
        with patch("time.sleep"):
            result = mc._smooth_move({"shoulder_pan": 10.0}, steps=2, step_delay=0)
        assert result is True

    def test_incremental_move_success(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        response = bytes([0xFF, 0xFF, 0x01, 0x04, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00])
        mc.serial_conn.read.return_value = response
        with patch("time.sleep"):
            result = mc.incremental_move("shoulder_pan", 5.0)
        assert result is True

    def test_incremental_move_read_fail(self):
        mc = self.make_controller()
        mc.serial_conn = MagicMock()
        mc.serial_conn.is_open = True
        mc.serial_conn.read.return_value = b""  # empty = read fail
        with patch("time.sleep"):
            result = mc.incremental_move("shoulder_pan", 5.0)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════
# pose_tool @tool function — uncovered action branches
# ═══════════════════════════════════════════════════════════════════════════


class TestPoseToolUncoveredActions:
    """Test the actions that have 0 coverage in the @tool function."""

    def _mock_controller(
        self,
        connect_ok=True,
        read_positions=None,
        move_ok=True,
        move_multiple_ok=True,
        incremental_ok=True,
        read_single=45.0,
    ):
        """Build a mock MotorController with configurable behavior."""
        mc = MagicMock()
        mc.connect.return_value = (connect_ok, "" if connect_ok else "port busy")
        mc.disconnect.return_value = None
        mc.read_all_positions.return_value = (
            read_positions
            if read_positions is not None
            else {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 50.0,
            }
        )
        mc.read_motor_position.return_value = read_single
        mc.move_motor.return_value = move_ok
        mc.move_multiple_motors.return_value = move_multiple_ok
        mc.incremental_move.return_value = incremental_ok
        return mc

    # ── read_all ─────────────────────────────────────────────────────

    def test_read_all_success(self):
        mc = self._mock_controller()
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="read_all", port="/dev/fake")
        assert result["status"] == "success"
        assert "shoulder_pan" in result["content"][0]["text"]
        assert "gripper" in result["content"][0]["text"]
        assert "positions" in result

    def test_read_all_connect_fail(self):
        mc = self._mock_controller(connect_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="read_all", port="/dev/fake")
        assert result["status"] == "error"

    def test_read_all_empty(self):
        mc = self._mock_controller(read_positions={})
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="read_all", port="/dev/fake")
        assert result["status"] == "error"

    # ── read_position (gripper unit display) ─────────────────────────

    def test_read_position_gripper_uses_percent(self):
        mc = self._mock_controller(read_single=50.0)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="read_position", port="/dev/fake", motor_name="gripper")
        assert result["status"] == "success"
        assert "%" in result["content"][0]["text"]

    def test_read_position_connect_fail(self):
        mc = self._mock_controller(connect_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="read_position", port="/dev/fake", motor_name="shoulder_pan")
        assert result["status"] == "error"

    def test_read_position_returns_none(self):
        mc = self._mock_controller(read_single=None)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="read_position", port="/dev/fake", motor_name="shoulder_pan")
        assert result["status"] == "error"

    # ── store_pose ───────────────────────────────────────────────────

    def test_store_pose_success(self, tmp_path):
        mc = self._mock_controller()
        pm = PoseManager("test", storage_dir=tmp_path)
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="store_pose",
                    port="/dev/fake",
                    pose_name="my_pose",
                    description="Test pose",
                )
        assert result["status"] == "success"
        assert "my_pose" in result["content"][0]["text"]
        assert "pose" in result  # pose data included

    def test_store_pose_no_name(self):
        mc = self._mock_controller()
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="store_pose", port="/dev/fake")
        assert result["status"] == "error"
        assert "pose_name required" in result["content"][0]["text"]

    def test_store_pose_connect_fail(self):
        mc = self._mock_controller(connect_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="store_pose", port="/dev/fake", pose_name="x")
        assert result["status"] == "error"

    def test_store_pose_read_fail(self):
        mc = self._mock_controller(read_positions={})
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="store_pose", port="/dev/fake", pose_name="x")
        assert result["status"] == "error"
        assert "Failed to read" in result["content"][0]["text"]

    # ── load_pose ────────────────────────────────────────────────────

    def test_load_pose_success(self, tmp_path):
        pm = PoseManager("test", storage_dir=tmp_path)
        pm.store_pose("home", {"shoulder_pan": 0.0, "gripper": 0.0})
        mc = self._mock_controller()
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="load_pose", port="/dev/fake", pose_name="home")
        assert result["status"] == "success"
        assert "home" in result["content"][0]["text"]

    def test_load_pose_not_found(self, tmp_path):
        pm = PoseManager("test", storage_dir=tmp_path)
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=MagicMock()):
                result = pose_tool(action="load_pose", port="/dev/fake", pose_name="nope")
        assert result["status"] == "error"

    def test_load_pose_no_name(self):
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=MagicMock()):
                result = pose_tool(action="load_pose", port="/dev/fake")
        assert result["status"] == "error"

    def test_load_pose_validation_fail(self, tmp_path):
        pm = PoseManager("test", storage_dir=tmp_path)
        pm.store_pose("bad", {"j1": 200.0}, safety_bounds={"j1": (-90, 90)})
        mc = self._mock_controller()
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="load_pose", port="/dev/fake", pose_name="bad")
        assert result["status"] == "error"
        assert "validation" in result["content"][0]["text"].lower()

    def test_load_pose_connect_fail(self, tmp_path):
        pm = PoseManager("test", storage_dir=tmp_path)
        pm.store_pose("home", {"j1": 0.0})
        mc = self._mock_controller(connect_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="load_pose", port="/dev/fake", pose_name="home")
        assert result["status"] == "error"

    def test_load_pose_move_fail(self, tmp_path):
        pm = PoseManager("test", storage_dir=tmp_path)
        pm.store_pose("home", {"j1": 0.0})
        mc = self._mock_controller(move_multiple_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="load_pose", port="/dev/fake", pose_name="home")
        assert result["status"] == "error"

    # ── move_multiple ────────────────────────────────────────────────

    def test_move_multiple_success(self):
        mc = self._mock_controller()
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="move_multiple",
                    port="/dev/fake",
                    positions={"shoulder_pan": 10.0, "gripper": 75.0},
                )
        assert result["status"] == "success"
        assert "gripper" in result["content"][0]["text"]

    def test_move_multiple_no_positions(self):
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=MagicMock()):
                result = pose_tool(action="move_multiple", port="/dev/fake")
        assert result["status"] == "error"
        assert "positions dict required" in result["content"][0]["text"]

    def test_move_multiple_connect_fail(self):
        mc = self._mock_controller(connect_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="move_multiple",
                    port="/dev/fake",
                    positions={"j1": 0.0},
                )
        assert result["status"] == "error"

    def test_move_multiple_move_fail(self):
        mc = self._mock_controller(move_multiple_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="move_multiple",
                    port="/dev/fake",
                    positions={"j1": 0.0},
                )
        assert result["status"] == "error"

    # ── incremental_move ─────────────────────────────────────────────

    def test_incremental_move_success(self):
        mc = self._mock_controller()
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="incremental_move",
                    port="/dev/fake",
                    motor_name="shoulder_pan",
                    delta=5.0,
                )
        assert result["status"] == "success"
        assert "+5.0" in result["content"][0]["text"]

    def test_incremental_move_negative(self):
        mc = self._mock_controller()
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="incremental_move",
                    port="/dev/fake",
                    motor_name="gripper",
                    delta=-10.0,
                )
        assert result["status"] == "success"
        assert "-10.0" in result["content"][0]["text"]
        assert "%" in result["content"][0]["text"]

    def test_incremental_move_no_params(self):
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=MagicMock()):
                result = pose_tool(action="incremental_move", port="/dev/fake")
        assert result["status"] == "error"
        assert "motor_name and delta required" in result["content"][0]["text"]

    def test_incremental_move_connect_fail(self):
        mc = self._mock_controller(connect_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="incremental_move",
                    port="/dev/fake",
                    motor_name="j1",
                    delta=1.0,
                )
        assert result["status"] == "error"

    def test_incremental_move_fail(self):
        mc = self._mock_controller(incremental_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="incremental_move",
                    port="/dev/fake",
                    motor_name="shoulder_pan",
                    delta=1.0,
                )
        assert result["status"] == "error"

    # ── move_motor (failure path) ────────────────────────────────────

    def test_move_motor_connect_fail(self):
        mc = self._mock_controller(connect_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="move_motor",
                    port="/dev/fake",
                    motor_name="j1",
                    position=10.0,
                )
        assert result["status"] == "error"

    def test_move_motor_fail(self):
        mc = self._mock_controller(move_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="move_motor",
                    port="/dev/fake",
                    motor_name="shoulder_pan",
                    position=10.0,
                )
        assert result["status"] == "error"

    def test_move_motor_gripper_unit(self):
        mc = self._mock_controller()
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(
                    action="move_motor",
                    port="/dev/fake",
                    motor_name="gripper",
                    position=75.0,
                )
        assert result["status"] == "success"
        assert "%" in result["content"][0]["text"]

    # ── reset_to_home failure ────────────────────────────────────────

    def test_reset_to_home_connect_fail(self):
        mc = self._mock_controller(connect_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="reset_to_home", port="/dev/fake")
        assert result["status"] == "error"

    def test_reset_to_home_move_fail(self):
        mc = self._mock_controller(move_multiple_ok=False)
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            with patch("strands_robots.tools.pose_tool.MotorController", return_value=mc):
                result = pose_tool(action="reset_to_home", port="/dev/fake")
        assert result["status"] == "error"

    # ── unknown action ───────────────────────────────────────────────

    def test_unknown_action(self):
        with patch("strands_robots.tools.pose_tool.PoseManager"):
            result = pose_tool(action="bogus_action", port="/dev/fake")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    # ── exception handler ────────────────────────────────────────────

    def test_exception_handler(self):
        """Test the catch-all exception handler inside the try block."""
        pm = MagicMock()
        pm.list_poses.side_effect = RuntimeError("boom")
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            result = pose_tool(action="list_poses")
        assert result["status"] == "error"
        assert "boom" in result["content"][0]["text"]
