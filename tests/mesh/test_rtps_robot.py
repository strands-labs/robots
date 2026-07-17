"""Behavior tests for RtpsRobot - forwards to use_rtps, mocked here.

ROS-free and cyclonedds-free: the forwarded ``use_rtps`` symbol is patched, so
the tests assert the robot builds the right RTPS calls (topic, type, Twist field
mapping, duration->count) and exposes correctly-named per-instance tools.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.mesh.rtps_robot import RtpsRobot


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"status": "success", "content": [{"text": "ok"}]}


@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr("strands_robots.mesh.rtps_robot.use_rtps", recorder)
    return recorder


def test_drive_maps_twist_fields(rec: _Recorder) -> None:
    robot = RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel")
    robot.drive(linear=2.0, angular=1.5)
    call = rec.calls[0]
    assert call["action"] == "publish"
    assert call["topic"] == "/turtle1/cmd_vel"
    assert call["type"] == "geometry_msgs/msg/Twist"
    assert call["fields"] == {"linear": {"x": 2.0}, "angular": {"z": 1.5}}


def test_drive_duration_to_count(rec: _Recorder) -> None:
    robot = RtpsRobot.from_rtps(node_name="bot", cmd_vel_topic="/cmd_vel", publish_rate=10.0)
    robot.drive(linear=1.0, duration=1.5)
    assert rec.calls[0]["count"] == 15  # round(1.5 * 10)


def test_stop_publishes_zero(rec: _Recorder) -> None:
    robot = RtpsRobot.from_rtps(node_name="bot", cmd_vel_topic="/cmd_vel")
    robot.stop()
    assert rec.calls[0]["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}
    assert rec.calls[0]["count"] == 1


def test_advertise_forwards(rec: _Recorder) -> None:
    robot = RtpsRobot.from_rtps(node_name="bot", cmd_vel_topic="/cmd_vel")
    robot.advertise()
    assert rec.calls[0]["action"] == "advertise"
    assert rec.calls[0]["topic"] == "/cmd_vel"


def test_tools_are_uniquely_named(rec: _Recorder) -> None:
    robot = RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel")
    names = {t.tool_name for t in robot.tools}
    assert names == {"drive_turtlesim", "stop_turtlesim"}


def test_invalid_topic_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="invalid cmd_vel_topic"):
        RtpsRobot.from_rtps(node_name="bot", cmd_vel_topic="not absolute")


def test_generated_drive_tool_publishes_over_rtps(rec: _Recorder) -> None:
    """The agent-facing drive_<node> tool forwards to the RTPS publish path.

    ``.tools`` builds per-instance ``@tool`` closures; invoking the drive tool
    must produce the same Twist publish call as :meth:`RtpsRobot.drive`, so an
    agent driving via the tool and code driving via the method are equivalent.
    """
    robot = RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel")
    drive_tool: Any = next(t for t in robot.tools if t.tool_name == "drive_turtlesim")

    result = drive_tool(linear=2.0, angular=1.5)

    assert result["status"] == "success"
    call = rec.calls[0]
    assert call["action"] == "publish"
    assert call["topic"] == "/turtle1/cmd_vel"
    assert call["type"] == "geometry_msgs/msg/Twist"
    assert call["fields"] == {"linear": {"x": 2.0}, "angular": {"z": 1.5}}


def test_generated_stop_tool_publishes_zero(rec: _Recorder) -> None:
    """The agent-facing stop_<node> tool publishes a single zero-velocity Twist."""
    robot = RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel")
    stop_tool: Any = next(t for t in robot.tools if t.tool_name == "stop_turtlesim")

    result = stop_tool()

    assert result["status"] == "success"
    call = rec.calls[0]
    assert call["action"] == "publish"
    assert call["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}
    assert call["count"] == 1


def test_repr_shows_wiring() -> None:
    """``repr`` surfaces the node name, cmd_vel topic, and interface type."""
    robot = RtpsRobot.from_rtps(node_name="turtlesim", cmd_vel_topic="/turtle1/cmd_vel")
    text = repr(robot)
    assert text == (
        "RtpsRobot(node_name='turtlesim', cmd_vel_topic='/turtle1/cmd_vel', cmd_vel_type='geometry_msgs/msg/Twist')"
    )
