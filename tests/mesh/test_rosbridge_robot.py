"""Behavior tests for RosbridgeRobot - a ROS1/rosbridge robot as a strands robot.

roslibpy-free: the module's ``use_rosbridge`` reference is monkeypatched with
a recorder, so forwarding, clamps, the trailing-zero safety rule, and the
Curiosity wiring are exercised with no rosbridge server anywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

import strands_robots.mesh.rosbridge_robot as rbr_mod
from strands_robots.mesh.rosbridge_robot import RosbridgeRobot


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return {"status": "success", "content": [{"text": "ok"}]}


@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(rbr_mod, "use_rosbridge", recorder)
    return recorder


def _rover(**overrides: Any) -> RosbridgeRobot:
    kwargs: dict[str, Any] = {"node_name": "rover", "cmd_vel_topic": "/cmd_vel", "odom_topic": "/odom"}
    kwargs.update(overrides)
    return RosbridgeRobot(**kwargs)


# Construction ----------------------------------------------------------------


def test_invalid_names_rejected() -> None:
    with pytest.raises(ValueError, match="invalid cmd_vel_topic"):
        _rover(cmd_vel_topic="/cmd vel")
    with pytest.raises(ValueError, match="invalid host"):
        _rover(host="bad host")
    with pytest.raises(ValueError, match="invalid port"):
        _rover(port=0)


@pytest.mark.parametrize("field", ["max_linear", "max_angular", "max_duration", "publish_rate"])
def test_nonpositive_or_nonfinite_numerics_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _rover(**{field: 0.0})
    with pytest.raises(ValueError, match=field):
        _rover(**{field: float("nan")})


def test_from_curiosity_wiring_pinned() -> None:
    rover = RosbridgeRobot.from_curiosity()
    assert rover.node_name == "curiosity"
    assert rover.cmd_vel_topic == "/curiosity_mars_rover/ackermann_drive_controller/cmd_vel"
    assert rover.odom_topic == "/curiosity_mars_rover/odom"
    assert rover.cmd_vel_type == "geometry_msgs/Twist"
    assert (rover.host, rover.port) == ("localhost", 9090)
    assert (rover.max_linear, rover.max_angular) == (2.0, 1.0)
    assert (rover.max_duration, rover.publish_rate) == (30.0, 10.0)


def test_from_curiosity_accepts_overrides() -> None:
    rover = RosbridgeRobot.from_curiosity(host="sim.local", max_linear=1.0)
    assert rover.host == "sim.local"
    assert rover.max_linear == 1.0
    assert rover.odom_topic == "/curiosity_mars_rover/odom"  # untouched


# drive -------------------------------------------------------------------------


def test_drive_forwards_clamped_twist(rec: _Recorder) -> None:
    rover = _rover(max_linear=2.0, max_angular=1.0)
    result = rover.drive(linear=5.0, angular=-3.0)
    assert result["status"] == "success"
    call = rec.calls[0]
    assert call["action"] == "publish"
    assert call["topic"] == "/cmd_vel"
    assert call["type"] == "geometry_msgs/Twist"
    assert call["host"] == "localhost" and call["port"] == 9090
    assert call["fields"] == {"linear": {"x": 2.0}, "angular": {"z": -1.0}}  # clamped both axes
    assert call["count"] == 1 and call["rate"] == 10.0
    assert len(rec.calls) == 1  # single-shot: no trailing zero


def test_drive_rejects_nonfinite(rec: _Recorder) -> None:
    for kwargs in (
        {"linear": float("nan")},
        {"angular": float("inf")},
        {"linear": 1.0, "duration": float("nan")},
    ):
        kwargs_typed: dict[str, Any] = kwargs
        assert _rover().drive(**kwargs_typed)["status"] == "error"
    assert rec.calls == []


def test_drive_rejects_bad_durations(rec: _Recorder) -> None:
    for bad in (0.0, -1.0, 31.0):  # max_duration default 30.0
        result = _rover().drive(linear=1.0, duration=bad)
        assert result["status"] == "error"
        assert "duration" in result["content"][0]["text"]
    assert rec.calls == []


def test_drive_duration_publishes_n_then_trailing_zero(rec: _Recorder) -> None:
    _rover().drive(linear=1.0, duration=2.0)  # 10 Hz -> 20 messages
    assert len(rec.calls) == 2
    assert rec.calls[0]["count"] == 20
    assert rec.calls[1]["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}
    assert rec.calls[1]["count"] == 1


def test_drive_short_duration_still_gets_trailing_zero(rec: _Recorder) -> None:
    _rover().drive(linear=1.0, duration=0.05)  # rounds to a single message, still timed
    assert len(rec.calls) == 2


def test_drive_trailing_zero_even_when_publish_errors(rec: _Recorder) -> None:
    failure = {"status": "error", "content": [{"text": "use_rosbridge: publish failed"}]}
    rec.responses = [failure]
    result = _rover().drive(linear=1.0, duration=1.0)
    assert result is failure
    assert len(rec.calls) == 2


def test_drive_sustained_zero_command_has_no_trailing_zero(rec: _Recorder) -> None:
    _rover().drive(linear=0.0, duration=1.0)
    assert len(rec.calls) == 1


# stop / observations -------------------------------------------------------------


def test_stop_publishes_zero(rec: _Recorder) -> None:
    _rover().stop()
    assert rec.calls[0]["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}


def test_get_pose_echoes_odom(rec: _Recorder) -> None:
    _rover().get_pose(timeout=3.0)
    call = rec.calls[0]
    assert call["action"] == "echo"
    assert call["topic"] == "/odom"
    assert call["timeout"] == 3.0 and call["count"] == 1


def test_get_scan_conditional(rec: _Recorder) -> None:
    result = _rover().get_scan()
    assert result["status"] == "error"
    assert "no scan_topic" in result["content"][0]["text"]
    assert rec.calls == []
    _rover(scan_topic="/scan").get_scan()
    assert rec.calls[0]["topic"] == "/scan"


# tools -------------------------------------------------------------------------


def test_tools_naming_and_conditionals() -> None:
    with_scan = {t.tool_name for t in _rover(scan_topic="/scan").tools}
    without = {t.tool_name for t in _rover().tools}
    assert with_scan == {"drive_rover", "stop_rover", "get_pose_rover", "get_scan_rover"}
    assert without == {"drive_rover", "stop_rover", "get_pose_rover"}


def test_drive_tool_description_discloses_latch_and_limits() -> None:
    drive_tool = next(t for t in _rover().tools if t.tool_name == "drive_rover")
    desc = drive_tool.tool_spec["description"]
    assert "latches" in desc
    assert "2.0" in desc  # max_linear disclosed


def test_tools_forward(rec: _Recorder) -> None:
    tools: dict[str, Any] = {t.tool_name: t for t in _rover(scan_topic="/scan").tools}
    tools["drive_rover"](linear=1.0)
    assert rec.calls[-1]["action"] == "publish"
    tools["get_pose_rover"]()
    assert rec.calls[-1]["topic"] == "/odom"


def test_exported_from_mesh() -> None:
    from strands_robots.mesh import RosbridgeRobot as exported

    assert exported is RosbridgeRobot
