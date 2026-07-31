#!/usr/bin/env python3
"""Universal rosbridge tool - any ROS graph over a WebSocket, no ROS install.

Where :func:`strands_robots.tools.use_ros.use_ros` speaks to a ROS 2 graph
through in-process ``rclpy`` (requiring a sourced distro), ``use_rosbridge``
speaks the rosbridge JSON protocol over a WebSocket via ``roslibpy`` - a pure
pip dependency. That gives it two properties no other transport here has:

* **ROS 1 robots** (rosbridge_suite ships for ROS1 and ROS2 alike) - e.g. the
  NASA Curiosity rover Gazebo simulation (ROS1 Noetic).
* **No ROS environment on this machine** - the agent can run on macOS, CI, or
  any laptop and drive a robot across the network.

Requirements:
    ``pip install "strands-robots[rosbridge]"`` (roslibpy). The robot side
    runs ``rosbridge_server`` (with ``rosapi``) - standard in every rosbridge
    install. rosbridge is unauthenticated by default: use on trusted networks.

Graph introspection uses the ``rosapi`` node's services. Interface types are
ROS1-style two-segment names (``geometry_msgs/Twist``); field payloads are
plain JSON dicts, exactly as rosbridge transmits them.

Actions:
    status         - roslibpy availability + connectivity to host:port.
    list_topics    - topics with their types (rosapi /rosapi/topics).
    list_services  - services (rosapi /rosapi/services).
    echo           - subscribe and return up to N messages as JSON. Type
                     auto-resolved via rosapi when omitted.
    publish        - advertise, publish N messages built from ``fields``,
                     unadvertise.
    service_call   - call a service with a JSON request dict.

Examples:
    use_rosbridge(action="status", host="192.168.1.20")
    use_rosbridge(action="list_topics")
    use_rosbridge(action="echo", topic="/curiosity_mars_rover/odom", count=1)
    use_rosbridge(action="publish",
                  topic="/curiosity_mars_rover/ackermann_drive_controller/cmd_vel",
                  type="geometry_msgs/Twist",
                  fields={"linear": {"x": 1.0}, "angular": {"z": 0.0}})
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from strands import tool

from strands_robots.tools._numeric_options import numeric_option_error
from strands_robots.utils import tcp_port_error

logger = logging.getLogger(__name__)

# Graph names: same allowlist posture as use_ros. Types are ROS1 two-segment.
_NAME_RE = re.compile(r"^[A-Za-z0-9_/~]+$")
_TYPE_RE = re.compile(r"^[A-Za-z0-9_]+/[A-Za-z0-9_]+$")
_HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# The top of the 16-bit port space is a legal TCP port that this transport cannot
# address. autobahn builds the WebSocket URL behind roslibpy with
# ``assert port is None or (type(port) == int and port in range(0, 65535))``
# (``autobahn/websocket/util.py``), and ``range(0, 65535)`` stops one short of
# 65535 - so the shared owner accepts the port, the kernel would bind it, and the
# transport then refuses it with a bare ``assert`` carrying an empty message,
# raised out of a function annotated ``-> dict[str, Any]``. An agent driving the
# tool got an exception where every other refusal is a result dict, and the
# exception named neither the tool nor the parameter.
#
# The narrower domain is therefore declared here and refused ahead of the backend
# probe, for the same reason the numeric options are: the caller learns the same
# thing whether or not roslibpy is installed, and no socket is dialed first. It is
# deliberately narrower than ``tcp_port_error``, which keeps the whole port space
# because that is what a port *is* - this bound belongs to one transport, not to
# the domain, and lives beside the transport that has it.
#
# Under ``python -O`` the assert is stripped and the transport carries 65535, but
# refusing it uniformly is the honest contract: the tool cannot promise a port
# whose acceptance depends on an interpreter flag.
_TRANSPORT_MAX_PORT = 65534


def _transport_port_error(port: int, param: str, context: str) -> str | None:
    """Error text when the rosbridge transport cannot address ``port``.

    Applied after :func:`strands_robots.utils.tcp_port_error`, which establishes
    that ``port`` is an ``int`` in the 16-bit space, so this only has to place it
    against the transport's own ceiling. Shared with
    :class:`strands_robots.mesh.rosbridge_robot.RosbridgeRobot`, which reaches
    this transport through this module, so the two cannot disagree about which
    ports it can carry.

    Args:
        port: A port already accepted by the shared 16-bit domain.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            requested action for an agent tool, or the class name for a
            constructor parameter.

    Returns:
        An error message, or ``None`` when the transport can address the port.
    """
    if port > _TRANSPORT_MAX_PORT:
        return (
            f"{context}: port {port!r} is a legal TCP port that the rosbridge WebSocket "
            f"transport cannot address (it addresses 1-{_TRANSPORT_MAX_PORT}; autobahn's "
            "URL builder excludes the top of the range)"
        )
    return None


_INSTALL_HINT = (
    "roslibpy is not importable - install the rosbridge extra: "
    'pip install "strands-robots[rosbridge]". The rosbridge transport is pure '
    "pip (WebSocket); no ROS environment is needed on this machine."
)

_ACTIONS = frozenset({"status", "list_topics", "list_services", "echo", "publish", "service_call"})

# Which numeric options each action actually consumes. Unlike the rclpy
# transport, whose graph introspection reads no caller budget, EVERY action here
# begins with a timeout-bounded WebSocket dial, so every action reads
# ``timeout``. The guard below is driven by this table rather than validating
# the whole signature unconditionally, so a caller is never refused for a value
# the requested action never looks at.
_ACTION_NUMERIC_OPTIONS: dict[str, tuple[str, ...]] = {
    "status": ("timeout",),
    "list_topics": ("timeout",),
    "list_services": ("timeout",),
    "echo": ("timeout", "count"),
    "service_call": ("timeout",),
    "publish": ("timeout", "count", "rate"),
}


class _RosbridgeBackend:
    """Process-wide cache of live roslibpy connections, keyed by (host, port).

    Tool calls are stateless; the WebSocket underneath is reused across calls
    (the rosbridge analogue of use_ros's single long-lived rclpy node). All
    access is serialised through ``lock``.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._connections: dict[tuple[str, int], Any] = {}
        self._available: bool | None = None

    def available(self) -> bool:
        if self._available is None:
            try:
                import roslibpy  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def connect(self, host: str, port: int, timeout: float) -> Any:
        """Return a live connection to host:port, dialing or awaiting reconnect.

        One roslibpy.Ros is created per (host, port) for the process lifetime
        and NEVER discarded: its ReconnectingClientFactory re-dials a dropped
        WebSocket by itself (observed live), while a freshly constructed Ros
        in a process that has seen reconnect churn can fail to connect at all
        (roslibpy/Twisted limitation, observed live). Keeping the one object
        is therefore both the reliable and the cheap choice - a bridge that is
        down costs one retrying factory with exponential backoff, not a storm.
        Calling ros.terminate() is never an option: it stops the process-wide,
        non-restartable Twisted reactor and would break every connection in
        this process.
        """
        import roslibpy

        # The client builds its WebSocket URL before it dials, and that builder
        # gates the port on type IDENTITY rather than isinstance:
        #
        #     assert port is None or (type(port) == int and port in range(0, 65535))
        #         - autobahn/websocket/util.py:85 (autobahn 26.7.1)
        #
        # So every int SUBCLASS is refused at every value, including the default
        # 9090 - an IntEnum read from a settings module is the realistic case -
        # and the refusal is a bare AssertionError with an empty message, raised
        # from the constructor below and therefore outside the try that converts
        # a failed dial. The value is legal and dials exactly as the equal plain
        # int does, so it is normalized here rather than refused: carrying it is
        # a capability this tool already advertises through its ``port: int``.
        # Ahead of the cache read on purpose - the key is then a plain int
        # whatever flavour arrived, so one (host, port) is one connection
        # regardless of which type reached it first, and the annotation on
        # ``_connections`` is true rather than aspirational.
        port = int(port)

        ros = self._connections.get((host, port))
        if ros is None:
            ros = roslibpy.Ros(host=host, port=port)
            self._connections[(host, port)] = ros
            try:
                ros.run(timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - roslibpy raises library-specific errors; all mean "not connected yet"
                raise TimeoutError(
                    f"could not connect to rosbridge at ws://{host}:{port} within {timeout}s "
                    f"- is rosbridge_server running? ({exc})"
                ) from exc
            if not getattr(ros, "is_connected", False):
                raise TimeoutError(
                    f"could not connect to rosbridge at ws://{host}:{port} within {timeout}s "
                    "- is rosbridge_server running?"
                )
            return ros
        if getattr(ros, "is_connected", False):
            return ros
        deadline = time.time() + timeout
        while time.time() < deadline:
            if getattr(ros, "is_connected", False):
                return ros
            time.sleep(0.05)
        raise TimeoutError(
            f"rosbridge at ws://{host}:{port} did not reconnect within {timeout}s "
            "- is rosbridge_server running? (the connection keeps retrying in the background)"
        )

    @property
    def lock(self) -> threading.RLock:
        return self._lock


_backend = _RosbridgeBackend()


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": f"use_rosbridge: {text}"}]}


def _rosapi_call(ros: Any, service: str, srv_type: str, values: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Call a service over rosbridge and return the response as a plain dict."""
    import roslibpy

    svc = roslibpy.Service(ros, service, srv_type)
    request = roslibpy.ServiceRequest(dict(values))
    try:
        return dict(svc.call(request, timeout=timeout))
    except Exception as exc:  # noqa: BLE001 - roslibpy raises library-specific errors; convert at the boundary
        raise TimeoutError(
            f"service {service} call failed via rosbridge: {exc} (is rosbridge_server running with rosapi?)"
        ) from exc


def _list_topics(ros: Any, timeout: float) -> str:
    resp = _rosapi_call(ros, "/rosapi/topics", "rosapi/Topics", {}, timeout)
    pairs = sorted(zip(resp.get("topics", []), resp.get("types", [])))
    return "\n".join(f"{name} [{type_}]" for name, type_ in pairs)


def _list_services(ros: Any, timeout: float) -> str:
    resp = _rosapi_call(ros, "/rosapi/services", "rosapi/Services", {}, timeout)
    return "\n".join(sorted(resp.get("services", [])))


def _resolve_topic_type(ros: Any, topic: str, timeout: float) -> str | None:
    resp = _rosapi_call(ros, "/rosapi/topic_type", "rosapi/TopicType", {"topic": topic}, timeout)
    return resp.get("type") or None


def _echo(ros: Any, topic: str, msg_type: str, timeout: float, count: int) -> list[dict[str, Any]]:
    import roslibpy

    received: list[dict[str, Any]] = []
    done = threading.Event()

    def _on_message(message: dict[str, Any]) -> None:
        received.append(dict(message))
        if len(received) >= count:
            done.set()

    sub = roslibpy.Topic(ros, topic, msg_type)
    sub.subscribe(_on_message)
    try:
        done.wait(timeout)
    finally:
        sub.unsubscribe()
    return received[:count]


def _service_call(ros: Any, service: str, srv_type: str, fields: dict[str, Any], timeout: float) -> dict[str, Any]:
    return _rosapi_call(ros, service, srv_type, fields, timeout)


@tool
def use_rosbridge(
    action: str,
    host: str = "localhost",
    port: int = 9090,
    topic: str | None = None,
    service: str | None = None,
    type: str | None = None,
    fields: dict[str, Any] | None = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
) -> dict[str, Any]:
    """Universal rosbridge tool - ROS over a WebSocket, no ROS install needed.

    Args:
        action: One of ``status``, ``list_topics``, ``list_services``,
            ``echo``, ``publish``, ``service_call``.
        host: rosbridge server hostname or IP.
        port: rosbridge WebSocket port (default 9090).
        topic: Topic name (``echo``, ``publish``).
        service: Service name (``service_call``).
        type: ROS1 two-segment interface type, e.g. ``geometry_msgs/Twist``.
            Auto-resolved for ``echo`` when omitted.
        fields: JSON field dict (``publish`` message / ``service_call`` request).
        timeout: Seconds for the WebSocket dial, sample collection, or a
            service call. A positive finite number of seconds; every action
            dials the bridge, so every action reads it.
        count: Messages to echo or publish. A positive integer; it is consumed
            as a ``range()`` bound, so ``0`` publishes nothing and a float or a
            numeric string cannot be honored.
        rate: Publish rate in Hz. A positive finite number - the inter-message
            period is ``1 / rate``, so ``0``, a negative value, ``nan`` and
            ``inf`` all leave the burst unthrottled rather than paced.

    Returns:
        A Strands tool result dict ``{"status": ..., "content": [{"text": ...}]}``.
    """
    fields = fields or {}

    if not host or not _HOST_RE.match(host):
        return _err(f"invalid host: {host!r}")
    if (port_error := tcp_port_error(port, "port", action)) is not None:
        return _err(port_error)
    if (transport_error := _transport_port_error(port, "port", action)) is not None:
        return _err(transport_error)
    if topic is not None and not _NAME_RE.match(topic):
        return _err(f"invalid topic name: {topic!r}")
    if service is not None and not _NAME_RE.match(service):
        return _err(f"invalid service name: {service!r}")
    if type is not None and not _TYPE_RE.match(type):
        return _err(f"invalid interface type: {type!r} (expected ROS1 pkg/Name like geometry_msgs/Twist)")

    if action not in _ACTIONS:
        return _err(f"unknown action: {action}")

    # Numeric options are checked here, alongside the names and ahead of the
    # backend probe, so the same caller mistake is reported identically whether
    # or not roslibpy is installed - and so a refusal happens before the
    # WebSocket is dialed and before a publisher is advertised.
    numeric_error = numeric_option_error(action, _ACTION_NUMERIC_OPTIONS, timeout=timeout, count=count, rate=rate)
    if numeric_error:
        return _err(numeric_error)

    if action == "status":
        if not _backend.available():
            return _ok("backend: none - " + _INSTALL_HINT)
        try:
            with _backend.lock:
                _backend.connect(host, port, timeout)
        except TimeoutError as exc:
            return _ok(f"backend: roslibpy; not connected - {exc}")
        return _ok(f"backend: roslibpy; connected to ws://{host}:{port}")

    if not _backend.available():
        return _err(_INSTALL_HINT)

    try:
        with _backend.lock:
            ros = _backend.connect(host, port, timeout)

            if action == "list_topics":
                return _ok(_list_topics(ros, timeout))

            if action == "list_services":
                return _ok(_list_services(ros, timeout))

            if action == "echo":
                if not topic:
                    return _err("echo requires topic")
                msg_type = type or _resolve_topic_type(ros, topic, timeout)
                if not msg_type:
                    return _err(f"cannot resolve type for {topic}; pass type=pkg/Name")
                import json

                samples = _echo(ros, topic, msg_type, timeout, count)
                body = json.dumps(samples, indent=2, default=str)
                note = (
                    "" if samples else f"\n(no messages within {timeout}s - topic may be silent or the type mismatched)"
                )
                return _ok(f"echo {topic} ({msg_type}):\n{body}{note}")

            if action == "service_call":
                if not service or not type:
                    return _err("service_call requires service and type")
                import json

                resp = _service_call(ros, service, type, fields, timeout)
                return _ok(f"response:\n{json.dumps(resp, indent=2, default=str)}")

            if action == "publish":
                if not topic or not type:
                    return _err("publish requires topic and type")
                _publish(ros, topic, type, fields, count, rate)
                return _ok(f"published {count} message(s) to {topic}")

            # Unreachable: action is validated against _ACTIONS above. Kept as
            # a defensive fallback because mypy cannot prove the if/elif
            # chain above is exhaustive from a runtime frozenset check.
            return _err(f"unknown action: {action}")  # pragma: no cover
    except TimeoutError as exc:
        return _err(str(exc))
    except (ImportError, KeyError, AttributeError, ValueError, TypeError, OSError) as exc:
        return _err(f"{action} failed: {exc}")


def _publish(ros: Any, topic: str, msg_type: str, fields: dict[str, Any], count: int, rate: float) -> None:
    import roslibpy

    pub = roslibpy.Topic(ros, topic, msg_type)
    pub.advertise()
    try:
        time.sleep(0.2)  # settle so rosbridge registers the publisher before the first send
        period = 1.0 / rate if rate > 0 else 0.0
        for _ in range(count):
            pub.publish(roslibpy.Message(dict(fields)))
            if period:
                time.sleep(period)
    finally:
        pub.unadvertise()
