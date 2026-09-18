"""The Microduck driver against a real robotd socket - the "works as-is" proof.

Every test here drives a genuine :class:`MockRobotd` over a real ``AF_UNIX``
socket, so the transport, the NDJSON framing, the id correlation and the wire
bytes are all exercised - not mocked away. A frame that drifts from the
``duck-ipc-proto`` contract fails a test rather than shipping a driver that
"works in sim, silently no-ops on the robot".
"""

from __future__ import annotations

import logging
import time

import pytest

from strands_robots.drivers.base import HardwareDriver, missing_driver_members
from strands_robots.drivers.microduck import (
    HARDWARE_JOINT_NAMES,
    LOCOMOTION_JOINT_NAMES,
    MICRODUCK_API_VERSION,
    MOUTH_INDEX,
    MicroduckDriver,
)
from strands_robots.policies.microduck import MICRODUCK_JOINT_NAMES
from tests.mocks.microduck_robotd import STATE_PARAMS, MockRobotd


def _connected_driver(server: MockRobotd, **kwargs) -> MicroduckDriver:
    driver = MicroduckDriver(tool_name="microduck", port=server.path, timeout=2.0, **kwargs)
    reason = driver.connect_eagerly()
    assert reason is None, reason
    return driver


def test_driver_satisfies_the_hardware_driver_surface() -> None:
    assert not missing_driver_members(MicroduckDriver)
    assert isinstance(MicroduckDriver(port="/tmp/nope.sock"), HardwareDriver)


def test_hello_handshake_connects_on_a_matching_version() -> None:
    with MockRobotd(api_version=MICRODUCK_API_VERSION) as server:
        driver = _connected_driver(server)
        try:
            assert driver.is_connected
            assert "hello" in server.methods
        finally:
            driver.cleanup()


def test_hello_connects_across_a_version_skew_and_logs_both_versions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # duck-ipc-proto: "No daemon refuses a call because this number differs" -
    # robotd bumps API_VERSION for additive methods and refuses a moved shape
    # by name on the one call that cannot be served. A gate here refused every
    # robotd built after the pin (28 on main against a pin of 16).
    with MockRobotd(api_version=MICRODUCK_API_VERSION + 12) as server:
        driver = MicroduckDriver(port=server.path, timeout=2.0)
        with caplog.at_level(logging.WARNING, logger="strands_robots.drivers.microduck"):
            reason = driver.connect_eagerly()
        try:
            assert reason is None, reason
            assert driver.is_connected
            skew = [r.getMessage() for r in caplog.records if "api_version" in r.getMessage()]
            assert skew, "the skew must be logged"
            assert str(MICRODUCK_API_VERSION + 12) in skew[0] and str(MICRODUCK_API_VERSION) in skew[0]
        finally:
            driver.cleanup()


def test_subscribe_stream_delivers_state_and_maps_15_to_14_joints() -> None:
    with MockRobotd() as server:
        driver = _connected_driver(server)
        try:
            # Wait for at least two streamed frames to arrive.
            deadline = time.time() + 3.0
            while driver.read_state()["status"] != "success" and time.time() < deadline:
                time.sleep(0.02)
            state = driver.read_state()
            assert state["status"] == "success", state
            body = state["content"][0]["json"]
            # Wire-key fidelity: the renamed fields survived as move/loop.
            assert body["move"]["applied"] == [0.15, 0.0, 0.0]
            assert body["loop"]["hz"] == 49.8
            assert body["policy"] == "walk"
            # 15 -> 14 mouth drop: index 9 is gone, the rest keep their values.
            joints = body["joints"]
            assert len(joints) == 14 and "mouth" not in joints
            assert tuple(joints.keys()) == LOCOMOTION_JOINT_NAMES
            # joints[] was 0..14; dropping index 9 shifts 10->right_hip_yaw.
            assert joints["neck_pitch"] == 5.0
            assert joints["right_hip_yaw"] == 10.0
            # get_observation (the mesh joint read) sees the same 14.
            assert driver.get_observation() == joints
        finally:
            driver.cleanup()


def test_send_twist_emits_the_exact_robot_move_notification_bytes() -> None:
    with MockRobotd() as server:
        driver = _connected_driver(server)
        try:
            result = driver.send_action({"vx": 0.1, "vy": 0.0, "vyaw": 0.0})
            assert result["status"] == "success", result
            # Let the server thread record the line.
            deadline = time.time() + 2.0
            while b"robot.move" not in b"".join(server.received) and time.time() < deadline:
                time.sleep(0.02)
            moves = [r for r in server.received if b'"robot.move"' in r]
            assert moves, server.methods
            assert moves[-1] == (b'{"jsonrpc":"2.0","method":"robot.move","params":{"vx":0.1,"vy":0.0,"vyaw":0.0}}\n')
        finally:
            driver.cleanup()


def test_send_skill_emits_a_robot_do_request_and_consumes_the_reply() -> None:
    with MockRobotd() as server:
        driver = _connected_driver(server)
        try:
            result = driver.send_action({"skill": "kick_left"})
            assert result["status"] == "success", result
            sent = result["content"][0]["json"]["sent"]
            assert sent[-1]["method"] == "robot.do"
            assert sent[-1]["result"] == {"accepted": True}
            dos = [r for r in server.received if b'"robot.do"' in r]
            assert dos, server.methods
            assert b'"skill":"kick_left"' in dos[-1] and b'"id":' in dos[-1]
        finally:
            driver.cleanup()


def test_an_unknown_skill_is_refused_at_the_door() -> None:
    with MockRobotd() as server:
        driver = _connected_driver(server)
        try:
            result = driver.send_action({"skill": "backflip"})
            assert result["status"] == "error"
            assert "unknown skill" in result["content"][0]["text"]
            assert not [r for r in server.received if b"robot.do" in r]
        finally:
            driver.cleanup()


def test_torque_relax_and_emergency_stop_are_discrete_intents() -> None:
    with MockRobotd() as server:
        driver = _connected_driver(server)
        try:
            assert driver.enable_torque(True)["status"] == "success"
            assert driver.relax()["status"] == "success"
            assert driver.emergency_stop()["status"] == "success"
            assert b'"robot.enable"' in b"".join(server.received)
            assert b'"robot.relax"' in b"".join(server.received)
            assert b'"robot.stop"' in b"".join(server.received)
        finally:
            driver.cleanup()


def test_disconnect_is_idempotent() -> None:
    with MockRobotd() as server:
        driver = _connected_driver(server)
        driver.cleanup()
        driver.cleanup()  # second call must not raise
        assert driver.is_connected is False


def test_the_14_locomotion_joints_are_the_15_minus_mouth() -> None:
    """The wire map's 15 joints minus ``mouth`` are the policy's 14, in order.

    The driver derives :data:`LOCOMOTION_JOINT_NAMES` from its own
    :data:`HARDWARE_JOINT_NAMES` rather than importing the policy constant, so
    the agreement between the two layers is a property to check rather than an
    assignment: a rename on either side, or a permutation of the ONNX tensor
    order, parts the wire map from the policy contract and fails here.
    """
    assert LOCOMOTION_JOINT_NAMES == tuple(name for i, name in enumerate(HARDWARE_JOINT_NAMES) if i != MOUTH_INDEX)
    assert LOCOMOTION_JOINT_NAMES == MICRODUCK_JOINT_NAMES
    assert HARDWARE_JOINT_NAMES[MOUTH_INDEX] == "mouth"
    assert len(STATE_PARAMS["joints"]) == 15
