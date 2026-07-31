"""A rosbridge port that is an ``int`` subclass dials like the equal plain int.

The WebSocket URL builder behind ``roslibpy`` gates the port on type identity,
not ``isinstance``::

    assert port is None or (type(port) == int and port in range(0, 65535))
        - autobahn/websocket/util.py:85 (autobahn 26.7.1)

So every ``int`` subclass was refused at every value - including the default
9090 - with a bare ``AssertionError`` carrying an empty message, raised from the
client constructor and therefore outside the ``try`` that converts a failed
dial. ``use_rosbridge`` normalizes the port to a plain ``int`` at the one place
a client is constructed, which is also the place the connection cache is keyed.

Four things have to stay true together, and they are grouped that way below:

1. The value dials, through the tool and through ``RosbridgeRobot``, and nothing
   escapes the tool envelope.
2. It dials in either call order. This is the half that makes the defect look
   intermittent: the cache is keyed on ``(host, port)`` and an ``IntEnum``
   hashes equal to its value, so any earlier plain-int call to the same
   host and port reused the cached client and never reached the URL builder.
3. The shared 16-bit domain is untouched - it is ``isinstance``-based on
   purpose, and this is a type-identity defect, not a range one.
4. The premise: the installed dependency really does gate on identity, and the
   normalization really does produce a value it accepts. If a future autobahn
   uses ``isinstance``, these say so rather than leaving an unexplained
   ``int()`` behind.
"""

from __future__ import annotations

import enum
import sys
import types as _types
from typing import Any

import pytest

import strands_robots.tools.use_rosbridge as rb_mod
from strands_robots.mesh.rosbridge_robot import RosbridgeRobot
from strands_robots.utils import tcp_port_error

use_rosbridge = rb_mod.use_rosbridge


# The realistic way an int subclass reaches the tool: a port named in a settings
# module rather than typed at the call site.
class _Port(enum.IntEnum):
    ROSBRIDGE = 9090
    ALT = 9091


class _IntSubclass(int):
    """A plain subclass, to show the defect is the subclassing and not the enum."""


PORT_FLAVOURS: list[Any] = [_Port.ROSBRIDGE, _IntSubclass(9090)]


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


class _RecordingRos:
    """Stands in for ``roslibpy.Ros``, reproducing its port gate.

    The gate is the point of this double: it is what turns "the tool passed an
    IntEnum straight through" from something invisible into a failure. Mirrors
    the completed double in test_use_rosbridge.py.
    """

    instances: list[_RecordingRos] = []

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        if not (port is None or (type(port) is int and port in range(0, 65535))):
            raise AssertionError
        self.host, self.port = host, port
        self.is_connected = False
        self.topics: list[Any] = []
        self.service_calls: list[Any] = []
        _RecordingRos.instances.append(self)

    def run(self, timeout: float | None = None) -> None:
        self.is_connected = True

    def terminate(self) -> None:  # pragma: no cover - never called by the tool
        self.is_connected = False


class _RecordingTopic:
    """Enough of ``roslibpy.Topic`` for the echo path the bridge drives."""

    def __init__(self, ros: Any, name: str, message_type: str) -> None:
        self.ros, self.name, self.message_type = ros, name, message_type
        ros.topics.append(self)

    def subscribe(self, cb: Any) -> None:
        cb({"pose": {"x": 1.0}})

    def unsubscribe(self) -> None:
        pass

    def advertise(self) -> None:
        pass

    def unadvertise(self) -> None:
        pass

    def publish(self, msg: dict[str, Any]) -> None:
        pass


@pytest.fixture
def fake_roslibpy(monkeypatch: pytest.MonkeyPatch) -> _types.ModuleType:
    """Inject a fake ``roslibpy`` and reset the process-wide connection cache.

    Resetting the cache is load-bearing here rather than hygiene: a leaked
    connection from an earlier test is exactly the masking this file is about.
    """
    _RecordingRos.instances = []
    mod = _types.ModuleType("roslibpy")
    mod.Ros = _RecordingRos  # type: ignore[attr-defined]
    mod.Topic = _RecordingTopic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "roslibpy", mod)
    monkeypatch.setattr(rb_mod._backend, "_available", True)
    monkeypatch.setattr(rb_mod._backend, "_connections", {})
    return mod


class TestAnIntSubclassPortDials:
    @pytest.mark.parametrize("port", PORT_FLAVOURS)
    def test_status_succeeds_and_nothing_escapes_the_envelope(
        self, port: Any, fake_roslibpy: _types.ModuleType
    ) -> None:
        result = use_rosbridge(action="status", host="127.0.0.1", port=port, timeout=0.05)

        assert result["status"] == "success"
        assert "connected to ws://127.0.0.1:9090" in _texts(result)

    @pytest.mark.parametrize("port", PORT_FLAVOURS)
    def test_the_client_is_constructed_with_a_plain_int(self, port: Any, fake_roslibpy: _types.ModuleType) -> None:
        use_rosbridge(action="status", host="127.0.0.1", port=port, timeout=0.05)

        (ros,) = _RecordingRos.instances
        assert type(ros.port) is int
        assert ros.port == 9090

    @pytest.mark.parametrize("action", sorted(rb_mod._ACTIONS))
    def test_every_action_dials_rather_than_raising(self, action: str, fake_roslibpy: _types.ModuleType) -> None:
        """Every action begins with a dial, so every action leaked this.

        The action set is read from the module rather than typed out, so a
        seventh action cannot be added without this covering it. What is
        asserted is narrow and uniform: the call returns a result dict rather
        than raising, and it got far enough to build a client - which is the
        step that used to fail. What the action then does with the connection
        is each action's own test's business, not this file's.
        """
        result = use_rosbridge(
            action=action,
            host="127.0.0.1",
            port=_Port.ROSBRIDGE,
            timeout=0.05,
            topic="/odom" if action in {"echo", "publish"} else None,
            service="/rosapi/topics" if action == "service_call" else None,
            type="nav_msgs/Odometry" if action in {"echo", "publish"} else None,
            count=1,
        )

        assert isinstance(result, dict)
        (ros,) = _RecordingRos.instances
        assert (type(ros.port), ros.port) == (int, 9090)

    def test_the_cache_holds_one_connection_per_host_and_port(self, fake_roslibpy: _types.ModuleType) -> None:
        use_rosbridge(action="status", host="127.0.0.1", port=_Port.ROSBRIDGE, timeout=0.05)
        use_rosbridge(action="status", host="127.0.0.1", port=9090, timeout=0.05)
        use_rosbridge(action="status", host="127.0.0.1", port=_Port.ALT, timeout=0.05)

        assert len(_RecordingRos.instances) == 2
        assert {(r.host, r.port) for r in _RecordingRos.instances} == {("127.0.0.1", 9090), ("127.0.0.1", 9091)}
        assert all(type(key[1]) is int for key in rb_mod._backend._connections)


class TestNeitherCallOrderMatters:
    """The cache made the defect order-dependent, so both orders are pinned.

    Before the fix, "enum then plain" raised and "plain then enum" succeeded,
    because the second call reused a client the first had already built. A
    reproducer that happened to touch the port with a plain int first reported
    that everything worked.
    """

    def test_subclass_first_then_plain(self, fake_roslibpy: _types.ModuleType) -> None:
        first = use_rosbridge(action="status", host="127.0.0.1", port=_Port.ROSBRIDGE, timeout=0.05)
        second = use_rosbridge(action="status", host="127.0.0.1", port=9090, timeout=0.05)

        assert (first["status"], second["status"]) == ("success", "success")

    def test_plain_first_then_subclass(self, fake_roslibpy: _types.ModuleType) -> None:
        first = use_rosbridge(action="status", host="127.0.0.1", port=9090, timeout=0.05)
        second = use_rosbridge(action="status", host="127.0.0.1", port=_Port.ROSBRIDGE, timeout=0.05)

        assert (first["status"], second["status"]) == ("success", "success")


def _rover(port: Any) -> RosbridgeRobot:
    return RosbridgeRobot(
        node_name="rover",
        cmd_vel_topic="/cmd_vel",
        odom_topic="/odom",
        # Named so the echo path needs no rosapi type lookup - this file is
        # about the port, and a Service double would only add surface.
        odom_type="nav_msgs/Odometry",
        host="127.0.0.1",
        port=port,
    )


class TestRosbridgeRobotInheritsTheFix:
    """The bridge forwards all of its I/O through the tool, so one site covers it."""

    def test_it_constructs_with_an_int_subclass_port(self) -> None:
        assert _rover(_Port.ROSBRIDGE).port == 9090

    def test_its_traffic_reaches_the_client_as_a_plain_int(self, fake_roslibpy: _types.ModuleType) -> None:
        """Drives real forwarding: ``get_pose`` calls the tool, which dials.

        Before the fix this raised the bare ``AssertionError`` out of
        ``get_pose`` - a method annotated ``-> dict[str, Any]``.
        """
        result = _rover(_Port.ROSBRIDGE).get_pose(timeout=0.05)

        assert result["status"] == "success"
        (ros,) = _RecordingRos.instances
        assert type(ros.port) is int
        assert ros.port == 9090


class TestTheSharedPortDomainIsUnchanged:
    """This is a type-identity defect, so the range domain must not move.

    Narrowing ``tcp_port_error`` to ``type(port) is int`` would reject the very
    values this fix makes work, and it is shared with surfaces that have no
    WebSocket under them.
    """

    @pytest.mark.parametrize("port", PORT_FLAVOURS)
    def test_it_still_accepts_an_int_subclass(self, port: Any) -> None:
        assert tcp_port_error(port, "port", "status") is None

    def test_it_still_rejects_bool_and_out_of_range(self) -> None:
        assert tcp_port_error(True, "port", "status") is not None
        assert tcp_port_error(0, "port", "status") is not None
        assert tcp_port_error(70000, "port", "status") is not None


class TestThePremiseTheNormalizationRestsOn:
    """Pins the claim about the dependency that ``int(port)`` exists to satisfy.

    Skipped rather than faked when the extra is not installed: a premise about
    a dependency is worth nothing measured against a double.
    """

    def test_the_url_builder_gates_the_port_on_type_identity(self) -> None:
        create_url = pytest.importorskip("autobahn.websocket.util").create_url

        with pytest.raises(AssertionError) as excinfo:
            create_url("127.0.0.1", port=_Port.ROSBRIDGE)

        assert str(excinfo.value) == ""
        assert create_url("127.0.0.1", port=9090)

    def test_the_normalized_value_is_one_the_builder_accepts(self) -> None:
        create_url = pytest.importorskip("autobahn.websocket.util").create_url

        assert create_url("127.0.0.1", port=int(_Port.ROSBRIDGE)) == create_url("127.0.0.1", port=9090)

    def test_the_client_refuses_it_at_construction_not_at_dial(self) -> None:
        """Why the tool's ``try`` around the dial never converted this.

        ``_RosbridgeBackend.connect`` wraps ``ros.run(...)`` in an ``except
        Exception`` that reports an unreachable bridge. That would have caught
        an ``AssertionError`` - the gate simply fires earlier, in the
        constructor, one line above the try.
        """
        roslibpy = pytest.importorskip("roslibpy")

        with pytest.raises(AssertionError):
            roslibpy.Ros(host="127.0.0.1", port=_Port.ROSBRIDGE)
