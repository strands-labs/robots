"""Integration tests for the AWS IoT MQTT transport.

Requires:
    - awsiotsdk installed (``pip install 'strands-robots[mesh-iot]'``)
    - AWS credentials with iot:* in the configured region
    - Either:
        * ``STRANDS_IOT_ENDPOINT`` + ``STRANDS_IOT_THING_NAME`` env vars set
          AND certs in ``$STRANDS_IOT_CERT_DIR/{thing}.{cert.pem,private.key}``
        * OR run ``scripts/iot_provision_test_things.py`` first to bootstrap
          the test fleet.

These tests skip cleanly when the prerequisites are absent so they don't break
the default ``hatch run test`` flow. Run explicitly with:

    hatch run test-integ tests_integ/test_iot_transport.py
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

awsiot = pytest.importorskip("awsiot", reason="awsiotsdk not installed")
awscrt = pytest.importorskip("awscrt", reason="awscrt not installed")

from strands_robots.mesh.transport import IotMqttTransport  # noqa: E402

# Skip whole module if env not configured

_ENDPOINT = os.getenv("STRANDS_IOT_ENDPOINT", "")
_CERT_DIR = Path(os.getenv("STRANDS_IOT_CERT_DIR") or Path.home() / ".strands_robots" / "iot")
_ROBOT_THING_A = os.getenv("STRANDS_IOT_TEST_ROBOT_A", "so100-spike-01")
_ROBOT_THING_B = os.getenv("STRANDS_IOT_TEST_ROBOT_B", "so100-spike-02")
_OPERATOR_THING = os.getenv("STRANDS_IOT_TEST_OPERATOR", "bedrock-agent-spike-01")


def _have_certs(thing: str) -> bool:
    return (
        (_CERT_DIR / f"{thing}.cert.pem").exists()
        and (_CERT_DIR / f"{thing}.private.key").exists()
        and (_CERT_DIR / "AmazonRootCA1.pem").exists()
    )


pytestmark = pytest.mark.skipif(
    not _ENDPOINT or not _have_certs(_ROBOT_THING_A) or not _have_certs(_OPERATOR_THING),
    reason=(
        f"AWS IoT integration not configured. Set STRANDS_IOT_ENDPOINT and "
        f"provide certs in {_CERT_DIR} for {_ROBOT_THING_A!r} and "
        f"{_OPERATOR_THING!r}, or run scripts/iot_provision_test_things.py."
    ),
)


# Tests


def test_robot_connects_and_publishes_presence():
    """A robot Thing can publish to its own /presence retained at QoS 1."""
    received: list[tuple[str, dict]] = []
    received_event = threading.Event()

    # Operator subscribes to all robot presence
    operator = IotMqttTransport(
        thing_name=_OPERATOR_THING,
        endpoint=_ENDPOINT,
        cert_dir=str(_CERT_DIR),
    )
    assert operator.connect(), "operator failed to connect"

    def on_presence(sample):
        topic = sample.key_expr
        payload = json.loads(sample.payload.to_bytes().decode())
        received.append((topic, payload))
        if payload.get("robot_id") == _ROBOT_THING_A:
            received_event.set()

    sub = operator.declare_subscriber("strands/+/presence", on_presence)
    time.sleep(0.5)  # let SUBACK settle

    # Robot publishes presence
    robot = IotMqttTransport(
        thing_name=_ROBOT_THING_A,
        endpoint=_ENDPOINT,
        cert_dir=str(_CERT_DIR),
    )
    assert robot.connect(), "robot failed to connect"

    presence_payload = {
        "v": 1,
        "robot_id": _ROBOT_THING_A,
        "robot_type": "so100",
        "hostname": "integ-test",
        "timestamp": time.time(),
    }
    robot.put(f"strands/{_ROBOT_THING_A}/presence", presence_payload)

    try:
        assert received_event.wait(10), f"Operator did not receive presence within 10s. Got: {[t for t, _ in received]}"
        # Verify the payload roundtripped intact
        matching = [p for t, p in received if p.get("robot_id") == _ROBOT_THING_A]
        assert matching
        assert matching[-1]["robot_type"] == "so100"
        assert matching[-1]["v"] == 1
    finally:
        sub.undeclare()
        robot.close()
        operator.close()


def test_operator_to_robot_rpc_roundtrip():
    """Operator publishes /cmd, robot dispatches, response reaches operator."""
    cmd_received = threading.Event()
    response_received = threading.Event()
    response_payload: dict = {}
    cmd_payload: dict = {}

    # Robot side
    robot = IotMqttTransport(
        thing_name=_ROBOT_THING_A,
        endpoint=_ENDPOINT,
        cert_dir=str(_CERT_DIR),
    )
    assert robot.connect()

    def on_cmd(sample):
        topic = sample.key_expr
        if topic != f"strands/{_ROBOT_THING_A}/cmd":
            return
        try:
            data = json.loads(sample.payload.to_bytes().decode())
        except Exception:
            return
        sender = data.get("sender_id", "")
        turn = data.get("turn_id", "")
        if not turn or sender == _ROBOT_THING_A:
            return
        cmd_payload.update(data)
        cmd_received.set()
        # Send response
        robot.put(
            f"strands/{sender}/response/{turn}",
            {
                "type": "response",
                "responder_id": _ROBOT_THING_A,
                "turn_id": turn,
                "result": {"ok": True, "echo": data.get("command")},
                "timestamp": time.time(),
            },
        )

    cmd_sub = robot.declare_subscriber(f"strands/{_ROBOT_THING_A}/cmd", on_cmd)

    # Operator side
    operator = IotMqttTransport(
        thing_name=_OPERATOR_THING,
        endpoint=_ENDPOINT,
        cert_dir=str(_CERT_DIR),
    )
    assert operator.connect()

    def on_response(sample):
        try:
            data = json.loads(sample.payload.to_bytes().decode())
        except Exception:
            return
        if data.get("type") != "response":
            return
        response_payload.update(data)
        response_received.set()

    resp_sub = operator.declare_subscriber(f"strands/{_OPERATOR_THING}/response/**", on_response)
    time.sleep(0.5)  # let subscriptions settle

    # Operator → Robot RPC
    turn_id = "integtest1"
    operator.put(
        f"strands/{_ROBOT_THING_A}/cmd",
        {
            "sender_id": _OPERATOR_THING,
            "turn_id": turn_id,
            "command": {"action": "status"},
            "timestamp": time.time(),
        },
    )

    try:
        assert cmd_received.wait(10), "Robot did not receive cmd"
        assert cmd_payload["turn_id"] == turn_id
        assert cmd_payload["command"]["action"] == "status"

        assert response_received.wait(10), "Operator did not receive response"
        assert response_payload["turn_id"] == turn_id
        assert response_payload["responder_id"] == _ROBOT_THING_A
        assert response_payload["result"]["ok"] is True
    finally:
        cmd_sub.undeclare()
        resp_sub.undeclare()
        robot.close()
        operator.close()


def test_camera_topic_is_dropped_silently():
    """Camera topics MUST NOT be published over MQTT (128 KB cap, cost)."""
    robot = IotMqttTransport(
        thing_name=_ROBOT_THING_A,
        endpoint=_ENDPOINT,
        cert_dir=str(_CERT_DIR),
    )
    assert robot.connect()

    operator = IotMqttTransport(
        thing_name=_OPERATOR_THING,
        endpoint=_ENDPOINT,
        cert_dir=str(_CERT_DIR),
    )
    assert operator.connect()

    received_camera: list = []

    def on_msg(sample):
        if "/camera/" in sample.key_expr:
            received_camera.append(sample.key_expr)

    sub = operator.declare_subscriber("strands/+/camera/+", on_msg)
    time.sleep(0.5)

    # Robot tries to publish a camera frame — should be dropped client-side
    robot.put(
        f"strands/{_ROBOT_THING_A}/camera/wrist",
        {"v": 1, "data": "fake-jpeg-base64", "shape": [480, 640, 3]},
    )
    time.sleep(2)  # wait for round-trip if it were going to happen

    try:
        assert received_camera == [], f"Camera frame leaked over MQTT! topics={received_camera}"
    finally:
        sub.undeclare()
        robot.close()
        operator.close()


def test_input_topic_is_dropped_silently():
    """Teleop input topics MUST stay LAN-only (50 Hz, fatal latency over WAN)."""
    robot = IotMqttTransport(
        thing_name=_ROBOT_THING_A,
        endpoint=_ENDPOINT,
        cert_dir=str(_CERT_DIR),
    )
    assert robot.connect()

    operator = IotMqttTransport(
        thing_name=_OPERATOR_THING,
        endpoint=_ENDPOINT,
        cert_dir=str(_CERT_DIR),
    )
    assert operator.connect()

    received: list = []

    def on_msg(sample):
        if "/input/" in sample.key_expr:
            received.append(sample.key_expr)

    sub = operator.declare_subscriber("strands/+/input/+", on_msg)
    time.sleep(0.5)

    robot.put(
        f"strands/{_ROBOT_THING_A}/input/leader",
        {"action": {"j0": 0.5}, "seq": 1, "v": 1},
    )
    time.sleep(2)

    try:
        assert received == [], f"Input topic leaked over MQTT! topics={received}"
    finally:
        sub.undeclare()
        robot.close()
        operator.close()
