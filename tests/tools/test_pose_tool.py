"""Behavior tests for the ``pose_tool`` agent tool.

Exercises the hardware-free logic with the serial layer mocked so the tests run
without a robot attached:

- ``RobotPose`` dataclass round-trips to and from dicts.
- ``PoseManager`` persists, retrieves, lists, deletes, and validates poses
  against safety bounds, surviving a reload from disk.
- ``MotorController`` builds correct Feetech protocol packets and converts
  between degrees and raw servo positions (including gripper percentage units),
  with the serial connection mocked.
- ``pose_tool`` dispatches every action branch and returns the
  ``{"status", "content"}`` contract on both success and error paths.

Also pins the project's "no emojis / ASCII-only in user-facing strings" rule:
every returned ``text`` field must be plain ASCII.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import serial

import strands_robots.tools.pose_tool as pose_mod
from strands_robots.tools.pose_tool import (
    MotorController,
    PoseManager,
    RobotPose,
    pose_tool,
)


def _texts(result: dict[str, Any]) -> str:
    """Concatenate all content ``text`` fields from a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _assert_ascii(result: dict[str, Any]) -> None:
    """Every user-facing text must be plain ASCII (no emojis, no degree sign)."""
    text = _texts(result)
    assert text.isascii(), f"non-ASCII characters in tool output: {text!r}"


# --------------------------------------------------------------------------- #
# Fake serial layer
# --------------------------------------------------------------------------- #
class FakeSerial:
    """Minimal stand-in for ``serial.Serial`` recording writes and serving reads."""

    def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.writes: list[bytes] = []
        self._read_queue: list[bytes] = []
        self.is_open = True

    def queue_read(self, data: bytes) -> None:
        self._read_queue.append(data)

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def read(self, n: int = 1) -> bytes:
        if self._read_queue:
            return self._read_queue.pop(0)
        return b""

    def close(self) -> None:
        self.is_open = False


@pytest.fixture
def fake_serial(monkeypatch):
    """Patch ``serial.Serial`` to return a single shared FakeSerial instance."""
    instances: list[FakeSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> FakeSerial:
        fs = FakeSerial(port, baudrate, timeout)
        instances.append(fs)
        return fs

    monkeypatch.setattr(serial, "Serial", _ctor)
    return instances


@pytest.fixture
def cwd_tmp(tmp_path, monkeypatch):
    """Run with cwd in a temp dir so PoseManager persists under tmp."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- #
# RobotPose
# --------------------------------------------------------------------------- #
def test_robot_pose_dict_round_trip() -> None:
    pose = RobotPose(
        name="grip",
        positions={"gripper": 50.0, "wrist_roll": 10.0},
        timestamp=123.0,
        description="closed",
        safety_bounds={"gripper": (0.0, 100.0)},
    )
    restored = RobotPose.from_dict(pose.to_dict())
    assert restored == pose
    assert restored.positions == {"gripper": 50.0, "wrist_roll": 10.0}


# --------------------------------------------------------------------------- #
# PoseManager
# --------------------------------------------------------------------------- #
def test_pose_manager_store_get_list_delete(cwd_tmp) -> None:
    mgr = PoseManager("arm_a")
    assert mgr.list_poses() == []

    mgr.store_pose("home", {"shoulder_pan": 0.0}, description="rest")
    assert mgr.list_poses() == ["home"]
    got = mgr.get_pose("home")
    assert got is not None and got.description == "rest"
    assert got.positions == {"shoulder_pan": 0.0}

    assert mgr.delete_pose("home") is True
    assert mgr.get_pose("home") is None
    assert mgr.delete_pose("home") is False


def test_pose_manager_persists_across_reload(cwd_tmp) -> None:
    PoseManager("arm_b").store_pose("ready", {"elbow_flex": 30.0})
    # A fresh manager for the same robot id loads the stored pose from disk.
    reloaded = PoseManager("arm_b")
    pose = reloaded.get_pose("ready")
    assert pose is not None
    assert pose.positions == {"elbow_flex": 30.0}


def test_pose_manager_load_corrupt_file_is_resilient(cwd_tmp) -> None:
    mgr = PoseManager("arm_c")
    mgr.pose_file.write_text("{ this is not valid json", encoding="utf-8")
    # Re-loading a corrupt file must not raise; it falls back to empty.
    recovered = PoseManager("arm_c")
    assert recovered.list_poses() == []


def test_pose_manager_validate_within_and_outside_bounds() -> None:
    mgr = PoseManager.__new__(PoseManager)  # avoid disk I/O; validate is pure
    ok_pose = RobotPose(
        name="ok",
        positions={"shoulder_pan": 10.0},
        timestamp=0.0,
        safety_bounds={"shoulder_pan": (-90.0, 90.0)},
    )
    valid, msg = mgr.validate_pose(ok_pose)
    assert valid is True

    bad_pose = RobotPose(
        name="bad",
        positions={"shoulder_pan": 200.0},
        timestamp=0.0,
        safety_bounds={"shoulder_pan": (-90.0, 90.0)},
    )
    valid, msg = mgr.validate_pose(bad_pose)
    assert valid is False
    assert "outside bounds" in msg

    no_bounds = RobotPose(name="nb", positions={"x": 5.0}, timestamp=0.0)
    valid, msg = mgr.validate_pose(no_bounds)
    assert valid is True


# --------------------------------------------------------------------------- #
# MotorController
# --------------------------------------------------------------------------- #
def test_feetech_packet_header_and_checksum() -> None:
    ctrl = MotorController("/dev/null")
    packet = ctrl.build_feetech_packet(1, 0x03, [0x2A, 0x00, 0x08])
    assert packet[0] == 0xFF and packet[1] == 0xFF
    assert packet[2] == 1  # motor id
    assert packet[3] == len([0x2A, 0x00, 0x08]) + 2  # length
    assert packet[4] == 0x03  # instruction
    # Checksum is the bitwise inverse of the sum of bytes from index 2 onward.
    expected = ~sum(packet[2:-1]) & 0xFF
    assert packet[-1] == expected


def test_degrees_position_round_trip_joint() -> None:
    ctrl = MotorController("/dev/null")
    # Mid-range degree maps near mid-resolution and back.
    pos = ctrl.degrees_to_position("shoulder_pan", 0.0)
    assert pos == pytest.approx(4095 // 2, abs=2)
    deg = ctrl.position_to_degrees("shoulder_pan", pos)
    assert deg == pytest.approx(0.0, abs=0.2)


def test_degrees_to_position_clamps_out_of_range() -> None:
    ctrl = MotorController("/dev/null")
    # shoulder_lift range is (-90, 90); 999 deg clamps to the max position.
    assert ctrl.degrees_to_position("shoulder_lift", 999.0) == 4095
    assert ctrl.degrees_to_position("shoulder_lift", -999.0) == 0


def test_gripper_uses_percentage_units() -> None:
    ctrl = MotorController("/dev/null")
    half = ctrl.degrees_to_position("gripper", 50.0)
    assert half == pytest.approx(4095 * 0.5, abs=1)
    assert ctrl.position_to_degrees("gripper", half) == pytest.approx(50.0, abs=0.1)


def test_unknown_motor_raises() -> None:
    ctrl = MotorController("/dev/null")
    with pytest.raises(ValueError, match="Unknown motor"):
        ctrl.degrees_to_position("not_a_motor", 0.0)
    with pytest.raises(ValueError, match="Unknown motor"):
        ctrl.position_to_degrees("not_a_motor", 0)


def test_connect_disconnect_and_move(fake_serial) -> None:
    ctrl = MotorController("/dev/ttyTEST")
    ok, err = ctrl.connect()
    assert ok is True and err == ""
    assert ctrl.move_motor("shoulder_pan", 0.0) is True
    # A position write packet was emitted to the serial bus.
    assert fake_serial[0].writes
    ctrl.disconnect()
    assert fake_serial[0].is_open is False


def test_move_motor_without_connection_returns_false() -> None:
    ctrl = MotorController("/dev/ttyTEST")
    assert ctrl.move_motor("shoulder_pan", 0.0) is False
    assert ctrl.read_motor_position("shoulder_pan") is None


def test_read_motor_position_decodes_response(fake_serial) -> None:
    ctrl = MotorController("/dev/ttyTEST")
    ctrl.connect()
    # 7-byte response with position low/high at indices 5 and 6.
    fake_serial[0].queue_read(bytes([0xFF, 0xFF, 0x01, 0x04, 0x00, 0x00, 0x08]))
    deg = ctrl.read_motor_position("shoulder_pan")
    assert deg is not None
    # position 0x0800 == 2048 -> roughly mid-range for a -180..180 joint.
    assert deg == pytest.approx(0.0, abs=1.0)


# --------------------------------------------------------------------------- #
# pose_tool dispatch
# --------------------------------------------------------------------------- #
def test_pose_tool_list_empty_is_ascii(cwd_tmp) -> None:
    result = pose_tool(action="list_poses", robot_id="empty_arm")
    assert result["status"] == "success"
    assert result["poses"] == []
    _assert_ascii(result)


def test_pose_tool_show_and_list_after_store(cwd_tmp) -> None:
    PoseManager("disp_arm").store_pose("ready", {"gripper": 40.0, "wrist_flex": 5.0}, description="staged")
    listed = pose_tool(action="list_poses", robot_id="disp_arm")
    assert listed["status"] == "success"
    assert any(p["name"] == "ready" for p in listed["poses"])
    _assert_ascii(listed)

    shown = pose_tool(action="show_pose", robot_id="disp_arm", pose_name="ready")
    assert shown["status"] == "success"
    assert shown["pose"]["description"] == "staged"
    _assert_ascii(shown)


def test_pose_tool_show_missing_pose_errors(cwd_tmp) -> None:
    result = pose_tool(action="show_pose", robot_id="disp_arm", pose_name="ghost")
    assert result["status"] == "error"
    assert "not found" in _texts(result)


def test_pose_tool_show_requires_pose_name(cwd_tmp) -> None:
    result = pose_tool(action="show_pose", robot_id="disp_arm")
    assert result["status"] == "error"
    assert "pose_name required" in _texts(result)


def test_pose_tool_delete(cwd_tmp) -> None:
    PoseManager("del_arm").store_pose("tmp", {"gripper": 0.0})
    ok = pose_tool(action="delete_pose", robot_id="del_arm", pose_name="tmp")
    assert ok["status"] == "success"
    missing = pose_tool(action="delete_pose", robot_id="del_arm", pose_name="tmp")
    assert missing["status"] == "error"


def test_pose_tool_requires_port_for_motor_ops(cwd_tmp) -> None:
    result = pose_tool(action="connect", robot_id="hw_arm", port=None)
    assert result["status"] == "error"
    assert "port required" in _texts(result)


def test_pose_tool_connect_success_is_ascii(cwd_tmp, fake_serial) -> None:
    result = pose_tool(action="connect", robot_id="hw_arm", port="/dev/ttyTEST")
    assert result["status"] == "success"
    _assert_ascii(result)


def test_pose_tool_connect_failure(cwd_tmp, monkeypatch) -> None:
    def _boom(*a, **k):
        raise OSError("no device")

    monkeypatch.setattr(serial, "Serial", _boom)
    result = pose_tool(action="connect", robot_id="hw_arm", port="/dev/ttyTEST")
    assert result["status"] == "error"
    assert "no device" in _texts(result)


def test_pose_tool_move_motor_requires_args(cwd_tmp, fake_serial) -> None:
    result = pose_tool(action="move_motor", robot_id="hw_arm", port="/dev/ttyTEST")
    assert result["status"] == "error"
    assert "required" in _texts(result)


def test_pose_tool_move_motor_success_is_ascii(cwd_tmp, fake_serial) -> None:
    result = pose_tool(
        action="move_motor",
        robot_id="hw_arm",
        port="/dev/ttyTEST",
        motor_name="shoulder_pan",
        position=10.0,
    )
    assert result["status"] == "success"
    _assert_ascii(result)


def test_pose_tool_emergency_stop_is_ascii(cwd_tmp) -> None:
    result = pose_tool(action="emergency_stop", robot_id="hw_arm")
    assert result["status"] == "success"
    _assert_ascii(result)


def test_pose_tool_unknown_action(cwd_tmp) -> None:
    result = pose_tool(action="fly", robot_id="hw_arm")
    assert result["status"] == "error"
    assert "Unknown action" in _texts(result)


def test_pose_tool_reset_to_home_is_ascii(cwd_tmp, fake_serial) -> None:
    result = pose_tool(action="reset_to_home", robot_id="hw_arm", port="/dev/ttyTEST")
    assert result["status"] == "success"
    _assert_ascii(result)
    assert "home_positions" in result


def test_module_source_is_ascii() -> None:
    """Regression: the whole module must be ASCII-only (no emojis / degree sign)."""
    src = Path(pose_mod.__file__).read_text(encoding="utf-8")
    assert src.isascii(), "pose_tool.py contains non-ASCII characters"
