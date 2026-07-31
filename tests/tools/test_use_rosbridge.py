"""Behavior tests for the ``use_rosbridge`` agent tool.

rosbridge is a WebSocket JSON transport (roslibpy) - pure pip, no sourced ROS
environment. These tests run roslibpy-free: a fake ``roslibpy`` module is
injected via ``sys.modules`` (fake Ros/Topic/Service record all traffic), so
connection caching, every action dispatch, the validation layer, and the
structured error contract are exercised with nothing installed.
"""

from __future__ import annotations

import sys
import types as _types
from typing import Any

import pytest

import strands_robots.tools.use_rosbridge as rb_mod

use_rosbridge = rb_mod.use_rosbridge


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


class _FakeTopic:
    def __init__(self, ros: Any, name: str, message_type: str) -> None:
        self.ros, self.name, self.message_type = ros, name, message_type
        self.advertised = False
        self.unadvertised = False
        self.unsubscribed = False
        self.published: list[dict[str, Any]] = []
        ros.topics.append(self)

    def advertise(self) -> None:
        self.advertised = True

    def unadvertise(self) -> None:
        self.unadvertised = True

    def publish(self, msg: dict[str, Any]) -> None:
        self.published.append(dict(msg))

    def subscribe(self, cb: Any) -> None:
        for m in list(type(self.ros).scripted_messages.get(self.name, [])):
            cb(m)

    def unsubscribe(self) -> None:
        self.unsubscribed = True


class _FakeService:
    def __init__(self, ros: Any, name: str, service_type: str) -> None:
        self.ros, self.name, self.service_type = ros, name, service_type

    def call(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.ros.service_calls.append((self.name, self.service_type, dict(request), timeout))
        responses = type(self.ros).scripted_responses
        if self.name in responses:
            return responses[self.name]
        raise RuntimeError(f"no scripted response for {self.name}")


class _FakeRos:
    instances: list[_FakeRos] = []
    fail_next_connect = False
    scripted_responses: dict[str, dict[str, Any]] = {}
    scripted_messages: dict[str, list[dict[str, Any]]] = {}

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self.host, self.port = host, port
        self.is_connected = False
        self.terminated = False
        self.topics: list[_FakeTopic] = []
        self.service_calls: list[tuple[str, str, dict[str, Any], float | None]] = []
        _FakeRos.instances.append(self)

    def run(self, timeout: float | None = None) -> None:
        if _FakeRos.fail_next_connect:
            raise RuntimeError("connection refused")
        self.is_connected = True

    def terminate(self) -> None:
        self.terminated = True
        self.is_connected = False


@pytest.fixture
def fake_roslibpy(monkeypatch: pytest.MonkeyPatch) -> _types.ModuleType:
    _FakeRos.instances = []
    _FakeRos.fail_next_connect = False
    _FakeRos.scripted_responses = {}
    _FakeRos.scripted_messages = {}
    mod = _types.ModuleType("roslibpy")
    mod.Ros = _FakeRos  # type: ignore[attr-defined]
    mod.Topic = _FakeTopic  # type: ignore[attr-defined]
    mod.Service = _FakeService  # type: ignore[attr-defined]
    mod.Message = dict  # type: ignore[attr-defined]
    mod.ServiceRequest = dict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "roslibpy", mod)
    monkeypatch.setattr(rb_mod._backend, "_connections", {})
    monkeypatch.setattr(rb_mod._backend, "_available", None)
    return mod


# Validation ------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["/cmd vel", "/x|y", "../etc", "/a$(x)"])
def test_invalid_topic_rejected(bad: str) -> None:
    result = use_rosbridge(action="echo", topic=bad)
    assert result["status"] == "error"
    assert "invalid topic" in _texts(result)


def test_ros1_two_segment_type_enforced() -> None:
    # ROS1 types are pkg/Name; a ROS2-style pkg/msg/Name must be rejected so
    # agents get a correcting error instead of a silent rosbridge failure.
    result = use_rosbridge(action="publish", topic="/cmd_vel", type="geometry_msgs/msg/Twist")
    assert result["status"] == "error"
    assert "invalid interface type" in _texts(result)


def test_valid_ros1_type_accepted_shapewise(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="publish", topic="/cmd_vel", type="geometry_msgs/Twist")
    assert result["status"] == "success"


@pytest.mark.parametrize("bad_host", ["bad host", "h;st", ""])
def test_invalid_host_rejected(bad_host: str) -> None:
    result = use_rosbridge(action="status", host=bad_host)
    assert result["status"] == "error"
    assert "invalid host" in _texts(result)


@pytest.mark.parametrize("bad_port", [0, -1, 70000])
def test_invalid_port_rejected(bad_port: int) -> None:
    result = use_rosbridge(action="status", port=bad_port)
    assert result["status"] == "error"
    assert "invalid port" in _texts(result)


# status / availability ---------------------------------------------------------


def test_status_reports_missing_roslibpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "roslibpy", None)  # import raises deterministically
    monkeypatch.setattr(rb_mod._backend, "_available", None)
    monkeypatch.setattr(rb_mod._backend, "_connections", {})
    result = use_rosbridge(action="status")
    assert result["status"] == "success"
    assert "backend: none" in _texts(result)
    assert "strands-robots[rosbridge]" in _texts(result)


def test_status_connects_and_reports(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="status", host="sim.local", port=9091)
    assert result["status"] == "success"
    assert "connected to ws://sim.local:9091" in _texts(result)


def test_status_reports_unreachable_bridge(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.fail_next_connect = True  # type: ignore[attr-defined]
    result = use_rosbridge(action="status")
    assert result["status"] == "success"
    assert "not connected" in _texts(result)
    assert "rosbridge_server" in _texts(result)


def test_actions_error_without_roslibpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "roslibpy", None)
    monkeypatch.setattr(rb_mod._backend, "_available", None)
    monkeypatch.setattr(rb_mod._backend, "_connections", {})
    result = use_rosbridge(action="list_topics")
    assert result["status"] == "error"
    assert "strands-robots[rosbridge]" in _texts(result)


# Connection cache --------------------------------------------------------------


def test_connection_cached_per_host_port(fake_roslibpy: _types.ModuleType) -> None:
    use_rosbridge(action="status")
    use_rosbridge(action="status")
    use_rosbridge(action="status", port=9091)
    hosts = [(r.host, r.port) for r in fake_roslibpy.Ros.instances]  # type: ignore[attr-defined]
    assert hosts == [("localhost", 9090), ("localhost", 9091)]  # second call reused the first


def test_stale_connection_kept_and_reused_after_recovery(fake_roslibpy: _types.ModuleType) -> None:
    use_rosbridge(action="status")
    first = fake_roslibpy.Ros.instances[0]  # type: ignore[attr-defined]
    first.is_connected = False  # dropped WebSocket, bridge still down
    result = use_rosbridge(action="status", timeout=0.2)
    assert result["status"] == "success"
    assert "did not reconnect" in _texts(result)
    # The entry is never terminated NOR discarded - its factory keeps
    # retrying, and a fresh Ros after churn is unreliable in-process.
    assert not first.terminated
    # Bridge comes back: the SAME object reconnects and is reused.
    first.is_connected = True
    result = use_rosbridge(action="status")
    assert "connected to" in _texts(result)
    assert len(fake_roslibpy.Ros.instances) == 1  # type: ignore[attr-defined]


def test_stale_connection_reused_when_factory_reconnects(fake_roslibpy: _types.ModuleType) -> None:
    use_rosbridge(action="status")
    first = fake_roslibpy.Ros.instances[0]  # type: ignore[attr-defined]

    class _Flapping:
        # is_connected reads False twice (initial check + first wait poll),
        # then True - simulating the auto-reconnecting factory recovering.
        def __init__(self) -> None:
            self.reads = 0

        def __get__(self, obj: Any, objtype: Any = None) -> bool:
            self.reads += 1
            return self.reads > 2

    type(first).is_connected = _Flapping()  # type: ignore[assignment]
    try:
        result = use_rosbridge(action="status", timeout=1.0)
        assert "connected to" in _texts(result)
        assert len(fake_roslibpy.Ros.instances) == 1  # type: ignore[attr-defined]  # same object reused
    finally:
        del type(first).is_connected  # restore instance-attribute behavior


def test_failed_dial_cached_and_recovers(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.fail_next_connect = True  # type: ignore[attr-defined]
    result = use_rosbridge(action="status", timeout=0.2)
    assert "not connected" in _texts(result)
    fake_roslibpy.Ros.fail_next_connect = False  # type: ignore[attr-defined]
    # The never-connected Ros stays cached; when its factory succeeds the
    # same object is reused - no second dial is ever attempted.
    orphan = fake_roslibpy.Ros.instances[0]  # type: ignore[attr-defined]
    orphan.is_connected = True
    result = use_rosbridge(action="status")
    assert "connected to" in _texts(result)
    assert len(fake_roslibpy.Ros.instances) == 1  # type: ignore[attr-defined]


def test_unknown_action_errors(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="warp_drive")
    assert result["status"] == "error"
    assert "unknown action" in _texts(result)


def test_unknown_action_rejected_before_any_connection(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.fail_next_connect = True  # type: ignore[attr-defined]
    result = use_rosbridge(action="warp_drive")
    assert result["status"] == "error"
    assert "unknown action" in _texts(result)
    assert fake_roslibpy.Ros.instances == []  # type: ignore[attr-defined]  # never dialed


# rosapi-backed introspection ---------------------------------------------------


def test_list_topics_formats_sorted_pairs(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/rosapi/topics"] = {  # type: ignore[attr-defined]
        "topics": ["/zeta", "/curiosity_mars_rover/odom"],
        "types": ["std_msgs/String", "nav_msgs/Odometry"],
    }
    result = use_rosbridge(action="list_topics")
    assert result["status"] == "success"
    lines = _texts(result).splitlines()
    assert lines[0] == "/curiosity_mars_rover/odom [nav_msgs/Odometry]"  # sorted
    assert "/zeta [std_msgs/String]" in lines[1]


def test_list_services_sorted(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/rosapi/services"] = {  # type: ignore[attr-defined]
        "services": ["/b_srv", "/a_srv"],
    }
    result = use_rosbridge(action="list_services")
    assert _texts(result).splitlines() == ["/a_srv", "/b_srv"]


def test_rosapi_absence_is_actionable(fake_roslibpy: _types.ModuleType) -> None:
    # No scripted response -> the fake raises like a timed-out service; the
    # tool must convert that into a structured, named error.
    result = use_rosbridge(action="list_topics")
    assert result["status"] == "error"
    assert "/rosapi/topics" in _texts(result)


# echo / service_call ------------------------------------------------------------


def test_echo_autoresolves_type_and_caps_count(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/rosapi/topic_type"] = {"type": "nav_msgs/Odometry"}  # type: ignore[attr-defined]
    fake_roslibpy.Ros.scripted_messages["/curiosity_mars_rover/odom"] = [  # type: ignore[attr-defined]
        {"pose": {"pose": {"position": {"x": 1.0}}}},
        {"pose": {"pose": {"position": {"x": 2.0}}}},
        {"pose": {"pose": {"position": {"x": 3.0}}}},
    ]
    result = use_rosbridge(action="echo", topic="/curiosity_mars_rover/odom", count=2)
    assert result["status"] == "success"
    assert "nav_msgs/Odometry" in _texts(result)
    assert '"x": 1.0' in _texts(result) and '"x": 2.0' in _texts(result)
    assert '"x": 3.0' not in _texts(result)  # capped at count
    ros = fake_roslibpy.Ros.instances[0]  # type: ignore[attr-defined]
    assert ros.topics[-1].unsubscribed  # subscription torn down in finally


def test_echo_unresolvable_type_errors(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/rosapi/topic_type"] = {"type": ""}  # type: ignore[attr-defined]
    result = use_rosbridge(action="echo", topic="/ghost")
    assert result["status"] == "error"
    assert "cannot resolve type" in _texts(result)


def test_echo_requires_topic(fake_roslibpy: _types.ModuleType) -> None:
    assert use_rosbridge(action="echo")["status"] == "error"


def test_echo_empty_result_discloses_timeout(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/rosapi/topic_type"] = {"type": "nav_msgs/Odometry"}  # type: ignore[attr-defined]
    result = use_rosbridge(action="echo", topic="/silent", timeout=0.1)
    assert result["status"] == "success"
    assert "no messages within" in _texts(result)


def test_publish_traffic_and_unadvertise(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(
        action="publish",
        topic="/cmd_vel",
        type="geometry_msgs/Twist",
        fields={"linear": {"x": 1.5}, "angular": {"z": 0.0}},
        count=3,
    )
    assert result["status"] == "success"
    ros = fake_roslibpy.Ros.instances[0]  # type: ignore[attr-defined]
    pub = ros.topics[-1]
    assert pub.advertised and pub.unadvertised
    assert pub.published == [{"linear": {"x": 1.5}, "angular": {"z": 0.0}}] * 3


def test_service_call_returns_response(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/gazebo/reset_world"] = {"ok": True}  # type: ignore[attr-defined]
    result = use_rosbridge(action="service_call", service="/gazebo/reset_world", type="std_srvs/Empty")
    assert result["status"] == "success"
    assert '"ok": true' in _texts(result)


def test_service_call_requires_service_and_type(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="service_call", service="/gazebo/reset_world")
    assert result["status"] == "error"
    assert "requires service and type" in _texts(result)


def test_publish_rejects_nonpositive_count(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="publish", topic="/cmd_vel", type="geometry_msgs/Twist", count=0)
    assert result["status"] == "error"
    assert "count" in _texts(result)
