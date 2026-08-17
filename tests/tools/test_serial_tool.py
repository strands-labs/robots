"""Behavior tests for the ``serial_tool`` agent tool.

Covers every action branch (``list_ports``, ``send``, ``read``, ``send_read``,
the Feetech servo commands, ``monitor``) plus the error paths, with the serial
layer mocked so the tests run hardware-free. Also pins the project's
"no emojis in user-facing strings" rule for this tool: every returned ``text``
must be plain ASCII.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

import pytest
import serial

from strands_robots.tools.serial_tool import serial_tool
from tests.tool_result_contract import tool_json


def _texts(result: dict[str, Any]) -> str:
    """Concatenate all content ``text`` fields from a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []))


class FakeSerial:
    """Minimal stand-in for ``serial.Serial`` recording writes and serving reads."""

    def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.writes: list[bytes] = []
        self.closed = False
        self._read_queue: list[bytes] = []
        self.in_waiting = 0

    def queue_read(self, data: bytes) -> None:
        self._read_queue.append(data)

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def read(self, n: int = 1) -> bytes:
        if self._read_queue:
            return self._read_queue.pop(0)
        return b""

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_serial_factory(monkeypatch):
    """Patch ``serial.Serial`` to return a configurable FakeSerial instance."""
    created: list[FakeSerial] = []

    def _make(reads: list[bytes] | None = None, in_waiting: int = 0):
        def _ctor(port, baudrate, timeout=1.0):
            inst = FakeSerial(port, baudrate, timeout)
            for chunk in reads or []:
                inst.queue_read(chunk)
            inst.in_waiting = in_waiting
            created.append(inst)
            return inst

        monkeypatch.setattr(serial, "Serial", _ctor)
        return created

    return _make


def test_list_ports_formats_discovered_ports(monkeypatch):
    port = SimpleNamespace(
        device="/dev/ttyACM0",
        name="ttyACM0",
        description="SO-100 arm",
        manufacturer="1a86",
        vid=0x1A86,
        pid=0x7523,
        serial_number="5AB0181806",
    )
    monkeypatch.setattr(serial.tools.list_ports, "comports", lambda: [port])

    result = serial_tool(action="list_ports")

    assert result["status"] == "success"
    assert tool_json(result)["ports"][0]["device"] == "/dev/ttyACM0"
    text = _texts(result)
    assert "Found 1 serial ports" in text
    assert "/dev/ttyACM0" in text
    assert text.isascii()


def test_missing_port_returns_error():
    result = serial_tool(action="send", data="hello")
    assert result["status"] == "error"
    assert "Port parameter required" in _texts(result)


def test_send_hex_data(fake_serial_factory):
    created = fake_serial_factory()
    result = serial_tool(action="send", port="/dev/ttyACM0", hex_data="FF FF 01 04")
    assert result["status"] == "success"
    assert created[0].writes == [bytes.fromhex("FFFF0104")]
    assert created[0].closed
    assert _texts(result).isascii()


def test_send_string_data(fake_serial_factory):
    created = fake_serial_factory()
    result = serial_tool(action="send", port="/dev/ttyACM0", data="PING")
    assert result["status"] == "success"
    assert created[0].writes == [b"PING"]


def test_send_without_payload_errors(fake_serial_factory):
    created = fake_serial_factory()
    result = serial_tool(action="send", port="/dev/ttyACM0")
    assert result["status"] == "error"
    assert "No data or hex_data" in _texts(result)
    assert created[0].closed


def test_read_formats_hex_and_ascii(fake_serial_factory):
    created = fake_serial_factory(reads=[b"AB\x01"])
    result = serial_tool(action="read", port="/dev/ttyACM0", read_bytes=8)
    assert result["status"] == "success"
    assert tool_json(result)["length"] == 3
    assert tool_json(result)["raw_data"] == b"AB\x01".hex()
    text = _texts(result)
    assert "41 42 01" in text
    assert "AB\\x01" in text
    assert text.isascii()
    assert created[0].closed


def test_send_read_round_trip(fake_serial_factory, monkeypatch):
    monkeypatch.setattr("strands_robots.tools.serial_tool.time.sleep", lambda *_: None)
    created = fake_serial_factory(reads=[b"OK"])
    result = serial_tool(action="send_read", port="/dev/ttyACM0", data="Q")
    assert result["status"] == "success"
    assert created[0].writes == [b"Q"]
    text = _texts(result)
    assert "Sent string: Q" in text
    assert "4F 4B" in text
    assert text.isascii()


def test_send_read_hex_and_missing_payload(fake_serial_factory, monkeypatch):
    monkeypatch.setattr("strands_robots.tools.serial_tool.time.sleep", lambda *_: None)
    created = fake_serial_factory(reads=[b""])
    ok = serial_tool(action="send_read", port="/dev/ttyACM0", hex_data="01 02")
    assert ok["status"] == "success"
    assert created[0].writes == [bytes.fromhex("0102")]

    fake_serial_factory()
    err = serial_tool(action="send_read", port="/dev/ttyACM0")
    assert err["status"] == "error"
    assert "No data to send" in _texts(err)


def test_feetech_position_builds_packet(fake_serial_factory):
    created = fake_serial_factory()
    result = serial_tool(action="feetech_position", port="/dev/ttyACM0", motor_id=1, position=2048)
    assert result["status"] == "success"
    packet = created[0].writes[0]
    # Header, motor id, instruction WRITE (0x03), Goal_Position addr (0x2A)
    assert packet[:3] == bytes([0xFF, 0xFF, 0x01])
    assert packet[4] == 0x03
    assert packet[5] == 0x2A
    # Checksum is the last byte and validates the packet body.
    assert packet[-1] == (~sum(packet[2:-1]) & 0xFF)
    text = _texts(result)
    assert "deg" in text
    assert text.isascii()


def test_feetech_position_requires_args(fake_serial_factory):
    created = fake_serial_factory()
    result = serial_tool(action="feetech_position", port="/dev/ttyACM0", motor_id=1)
    assert result["status"] == "error"
    assert "motor_id and position required" in _texts(result)
    assert created[0].closed


def test_feetech_velocity_builds_packet(fake_serial_factory):
    created = fake_serial_factory()
    result = serial_tool(action="feetech_velocity", port="/dev/ttyACM0", motor_id=2, velocity=100)
    assert result["status"] == "success"
    packet = created[0].writes[0]
    assert packet[5] == 0x2E  # Goal_Velocity address
    assert _texts(result).isascii()


def test_feetech_velocity_requires_args(fake_serial_factory):
    fake_serial_factory()
    result = serial_tool(action="feetech_velocity", port="/dev/ttyACM0", velocity=100)
    assert result["status"] == "error"
    assert "motor_id and velocity required" in _texts(result)


def test_feetech_ping_success(fake_serial_factory, monkeypatch):
    monkeypatch.setattr("strands_robots.tools.serial_tool.time.sleep", lambda *_: None)
    fake_serial_factory(reads=[bytes([0xFF, 0xFF, 0x01, 0x02, 0x00, 0x00])])
    result = serial_tool(action="feetech_ping", port="/dev/ttyACM0", motor_id=1)
    assert result["status"] == "success"
    assert "responded" in _texts(result)
    assert _texts(result).isascii()


def test_feetech_ping_no_response(fake_serial_factory, monkeypatch):
    monkeypatch.setattr("strands_robots.tools.serial_tool.time.sleep", lambda *_: None)
    fake_serial_factory(reads=[b"\x00"])
    result = serial_tool(action="feetech_ping", port="/dev/ttyACM0", motor_id=1)
    assert result["status"] == "error"
    assert "no response" in _texts(result)
    assert _texts(result).isascii()


def test_feetech_ping_requires_motor_id(fake_serial_factory):
    created = fake_serial_factory()
    result = serial_tool(action="feetech_ping", port="/dev/ttyACM0")
    assert result["status"] == "error"
    assert "motor_id required" in _texts(result)
    assert created[0].closed


def test_monitor_collects_chunks(fake_serial_factory, monkeypatch):
    """The monitor window is a duration; each record's timestamp is a stamp.

    Those are different clocks and the test drives both: the five-second window
    is bounded on ``time.monotonic()``, so a wall-clock step cannot cut the
    window short or hold the port open past it, while the ``timestamp`` on each
    returned record stays on the wall clock because a reader correlates it with
    other logs and seconds of process uptime would not.
    """
    monkeypatch.setattr("strands_robots.tools.serial_tool.time.sleep", lambda *_: None)
    # Scripted so the window admits exactly one read and then expires. Chained
    # with a repeat rather than a fixed-length iterator so the script bounds the
    # window instead of the number of clock reads.
    window = itertools.chain([0.0, 0.0], itertools.repeat(10.0))
    monkeypatch.setattr("strands_robots.tools.serial_tool.time.monotonic", lambda: next(window))
    monkeypatch.setattr("strands_robots.tools.serial_tool.time.time", lambda: 1_700_000_000.5)
    created = fake_serial_factory(reads=[b"hi"], in_waiting=2)
    result = serial_tool(action="monitor", port="/dev/ttyACM0")
    assert result["status"] == "success"
    assert len(tool_json(result)["monitor_data"]) == 1
    assert tool_json(result)["monitor_data"][0]["data"] == b"hi".hex()
    assert tool_json(result)["monitor_data"][0]["timestamp"] == 1_700_000_000.5, (
        "the record timestamp must come from the wall clock, not from the monotonic clock that bounds the window"
    )
    assert _texts(result).isascii()
    assert created[0].closed


def test_unknown_action_returns_error(fake_serial_factory):
    created = fake_serial_factory()
    result = serial_tool(action="bogus", port="/dev/ttyACM0")
    assert result["status"] == "error"
    text = _texts(result)
    assert "Unknown action: bogus" in text
    assert "list_ports" in text
    assert created[0].closed


def test_serial_exception_is_caught(monkeypatch):
    def _raise(*_a, **_k):
        raise serial.SerialException("port busy")

    monkeypatch.setattr(serial, "Serial", _raise)
    result = serial_tool(action="send", port="/dev/ttyACM0", data="x")
    assert result["status"] == "error"
    assert "Serial error: port busy" in _texts(result)


def test_generic_exception_is_caught(monkeypatch):
    def _raise(*_a, **_k):
        raise ValueError("boom")

    monkeypatch.setattr(serial, "Serial", _raise)
    result = serial_tool(action="send", port="/dev/ttyACM0", data="x")
    assert result["status"] == "error"
    assert "Error: boom" in _texts(result)
