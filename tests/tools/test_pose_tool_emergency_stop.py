# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``pose_tool(action="emergency_stop")`` must de-energize the arm, or say it did not.

The handler used to be::

    if action == "emergency_stop":
        # This would require torque disable in real implementation
        return {"status": "success", "content": [{"text": "Emergency stop executed (torque disabled)"}]}

No code ran. It reported success unconditionally, for every port, connected or
not -- a fabricated safety confirmation on the one action an operator or agent
reaches for when an arm is moving and must stop. The tests here assert against
the BYTES that reach the bus rather than the status text, because the status
text was exactly what was already correct.

``Torque_Enable`` is address 40 (``0x28``, 1 byte) on the Feetech STS/SMS
control table, per ``lerobot.motors.feetech.tables``.

No real serial port is opened: ``serial.Serial`` is replaced by the recording
``FakeSerial`` from the sibling module, and every call passes an explicit fake
``port`` -- ``pose_tool``'s ``port`` defaults to ``/dev/ttyACM0``, so a portless
call would reach a real arm plugged into the machine running the suite.
"""

from __future__ import annotations

import ast
from pathlib import Path

import serial

from strands_robots.tools.pose_tool import MotorController, pose_tool

from .conftest import FakeSerial

#: This repository's test tree, located from this file rather than from the
#: working directory. ``Path("tests")`` named whichever directory the process
#: happened to start in, so the hazard sweep below graded a tree that was not
#: there and reported it clean.
_TEST_TREE = Path(__file__).resolve().parents[2] / "tests"
_PORT = "/dev/ttyTEST"
_TORQUE_ENABLE_ADDR = 0x28
_MOTOR_IDS = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}


def _torque_writes(writes: list[bytes]) -> list[int]:
    """Motor ids that received a ``Torque_Enable = 0`` write, in order."""
    return [
        w[2]
        for w in writes
        if len(w) >= 7 and w[:2] == b"\xff\xff" and w[4] == 0x03 and w[5] == _TORQUE_ENABLE_ADDR and w[6] == 0x00
    ]


class TestEmergencyStopReachesTheBus:
    def test_every_motor_is_de_energized(self, cwd_tmp, fake_serial):
        """The call writes Torque_Enable=0 to all six motors.

        Pre-fix ``fake_serial`` recorded zero writes and the port was never even
        opened, while the tool reported success.
        """
        result = pose_tool(action="emergency_stop", robot_id="arm", port=_PORT)

        assert result["status"] == "success"
        assert len(fake_serial) == 1, "the bus must actually be opened"
        assert _torque_writes(fake_serial[0].writes) == list(_MOTOR_IDS.values())

    def test_packets_are_well_formed_feetech_writes(self, cwd_tmp, fake_serial):
        """Each packet is a checksummed INST_WRITE of 0 to address 40."""
        pose_tool(action="emergency_stop", robot_id="arm", port=_PORT)

        writes = fake_serial[0].writes
        assert len(writes) == len(_MOTOR_IDS)
        for motor_id, packet in zip(_MOTOR_IDS.values(), writes, strict=True):
            body = [motor_id, 4, 0x03, _TORQUE_ENABLE_ADDR, 0x00]
            assert packet == bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF])

    def test_the_port_is_released(self, cwd_tmp, fake_serial):
        """A stop must not leave the bus held open against the next attempt."""
        pose_tool(action="emergency_stop", robot_id="arm", port=_PORT)

        assert fake_serial[0].is_open is False

    def test_success_text_warns_the_arm_goes_limp(self, cwd_tmp, fake_serial):
        """De-energizing drops whatever is held; the report must say so.

        An operator who reads "stopped" and expects the arm to hold position
        gets a falling arm and a dropped payload.
        """
        result = pose_tool(action="emergency_stop", robot_id="arm", port=_PORT)

        text = " ".join(c["text"] for c in result["content"] if "text" in c).lower()
        assert "limp" in text
        assert "fall" in text


class TestEmergencyStopReportsFailure:
    def test_unreachable_bus_is_not_reported_as_a_stop(self, cwd_tmp, monkeypatch):
        """A port that cannot be opened must report the arm was NOT released.

        Pre-fix this returned success -- the arm could be unplugged, powered
        down, or held by another process and the tool still said it had stopped.
        """

        def _boom(port, baudrate, timeout=1.0):
            raise serial.SerialException("could not open port")

        monkeypatch.setattr(serial, "Serial", _boom)

        result = pose_tool(action="emergency_stop", robot_id="arm", port=_PORT)

        assert result["status"] == "error"
        text = " ".join(c["text"] for c in result["content"] if "text" in c)
        assert "NOT de-energized" in text

    def test_a_failed_write_names_the_joints_still_driven(self, cwd_tmp, fake_serial, monkeypatch):
        """A motor whose write fails is named, and the rest are still attempted.

        Giving up at the first failure would leave later joints driven while
        reporting nothing about them.
        """
        controller = MotorController(_PORT)
        connected, _ = controller.connect()
        assert connected

        failing = {"elbow_flex", "gripper"}
        real_write = controller.serial_conn.write
        ids_to_names = {v: k for k, v in _MOTOR_IDS.items()}

        def _write(data: bytes) -> None:
            if ids_to_names.get(data[2]) in failing:
                raise serial.SerialException("bus write failed")
            real_write(data)

        monkeypatch.setattr(controller.serial_conn, "write", _write)

        failed = controller.disable_torque()

        assert sorted(failed) == sorted(failing)
        # The four healthy joints were still released.
        assert _torque_writes(fake_serial[0].writes) == [
            motor_id for name, motor_id in _MOTOR_IDS.items() if name not in failing
        ]

    def test_a_closed_bus_claims_no_motor_was_released(self, cwd_tmp):
        """With no open connection, every motor is reported as still driven.

        Returning an empty "nothing failed" list would read as a complete stop.
        """
        controller = MotorController(_PORT)

        assert sorted(controller.disable_torque()) == sorted(_MOTOR_IDS)

    def test_partial_failure_surfaces_as_a_tool_error(self, cwd_tmp, monkeypatch):
        """The tool reports status=error, not success, when a joint is left driven."""

        def _partial(self) -> list[str]:
            return ["gripper"]

        monkeypatch.setattr(MotorController, "disable_torque", _partial)
        monkeypatch.setattr(serial, "Serial", lambda port, baudrate, timeout=1.0: FakeSerial(port, baudrate, timeout))

        result = pose_tool(action="emergency_stop", robot_id="arm", port=_PORT)

        assert result["status"] == "error"
        text = " ".join(c["text"] for c in result["content"] if "text" in c)
        assert "gripper" in text
        assert "INCOMPLETE" in text


def test_no_test_calls_emergency_stop_on_the_default_port():
    """No test anywhere may run ``emergency_stop`` without an explicit port.

    ``pose_tool``'s ``port`` defaults to ``/dev/ttyACM0``. Now that the handler
    really opens the bus and writes ``Torque_Enable=0``, a portless call
    de-energizes whatever arm is attached to the machine running the suite --
    it silently passes on a developer box with nothing plugged in and drops a
    live arm on one that has hardware. Walk the whole test tree by AST so the
    hazard cannot be reintroduced in a file nobody thinks to check.

    The tree is located from this file rather than from the working directory.
    ``Path("tests")`` resolved against wherever the process started, so a run
    from any other directory swept zero files and reported the tree clean - the
    one direction a safety guard must never fail in. ``scanned`` carries that
    proof: a sweep that saw nothing fails here instead of passing.
    """
    offenders = []
    scanned = 0
    for path in sorted(_TEST_TREE.rglob("*.py")):
        scanned += 1
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name != "pose_tool":
                continue
            kwargs = {kw.arg for kw in node.keywords}
            action = next(
                (kw.value.value for kw in node.keywords if kw.arg == "action" and isinstance(kw.value, ast.Constant)),
                None,
            )
            if action == "emergency_stop" and "port" not in kwargs:
                offenders.append(f"{path}:{node.lineno}")

    assert scanned > 100, (
        f"the sweep read {scanned} test modules, so it graded almost nothing - it is rooted at "
        f"{_TEST_TREE}, which is not this repository's test tree"
    )
    assert not offenders, f"emergency_stop called without an explicit port: {offenders}"
