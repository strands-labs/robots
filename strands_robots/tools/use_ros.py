#!/usr/bin/env python3
"""Universal ROS 2 bridge tool - one tool for the full ROS 2 surface.

Like ``use_lerobot`` wraps the lerobot module tree, ``use_ros`` gives a Strands
agent a single, structured entry point into any ROS 2 graph reachable from this
interpreter - **entirely in-process through ``rclpy``**. There is no shelling
out to the ``ros2`` CLI and no code-generation: every action calls the ROS 2
client library directly, so message types are real Python classes, errors are
real exceptions, and a single long-lived node/executor is reused across calls.

Requirements:
    ``rclpy`` and ``rosidl_runtime_py`` must be importable in this interpreter.
    These ship with a sourced system ROS 2 distro (apt / RoboStack / conda) and
    are **not** on PyPI, so they cannot be ``pip install``ed and are not pinned
    in ``pyproject.toml`` (the ``[ros2]`` extra only carries the pip-installable
    ``cyclonedds`` RMW binding). Source a ROS 2 environment before launching the
    agent - e.g. ``source /opt/ros/jazzy/setup.bash`` - and ``rclpy`` becomes
    importable. When it is absent, every action returns a clear, actionable
    error instead of raising.

Message and service types are resolved dynamically through
``rosidl_runtime_py`` (``get_message`` / ``get_service``), so any interface
installed in the ROS 2 environment works with no static registry. Field
payloads are passed as plain JSON dicts and applied with ``set_message_fields``
- the standard ROS 2 idiom.

Actions:
    status         - report whether the in-process rclpy backend is available.
    list_topics    - list topics with their types.
    list_nodes     - list nodes.
    list_services  - list services with their types.
    info           - describe a topic (type + pub/sub counts) or service (type).
    echo           - subscribe to a topic and return N samples as JSON.
    publish        - publish N messages built from a JSON field dict.
    service_call   - call a service with a JSON request dict, return the response.
    list_actions   - list action servers with their types.
    action_send_goal - send a goal to an action server, stream feedback, and
                     return the terminal result (goal-level autonomy: Nav2
                     NavigateToPose, FollowJointTrajectory, gripper commands).
                     On timeout the goal is cancelled before returning, so a
                     robot is never left executing an orphaned goal.

Examples:
    use_ros(action="status")
    use_ros(action="list_topics")
    use_ros(action="echo", topic="/turtle1/pose", timeout=2.0, count=2)
    use_ros(action="publish", topic="/turtle1/cmd_vel",
            type="geometry_msgs/msg/Twist",
            fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}})
    use_ros(action="service_call", service="/spawn",
            type="turtlesim/srv/Spawn",
            fields={"x": 3.0, "y": 3.0, "name": "t2"})
    use_ros(action="action_send_goal", action_name="/navigate_to_pose",
            type="nav2_msgs/action/NavigateToPose",
            fields={"pose": {"header": {"frame_id": "map"},
                             "pose": {"position": {"x": 1.0, "y": 2.0}}}},
            timeout=120.0)
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from strands import tool

from strands_robots.tools._numeric_options import numeric_option_error

logger = logging.getLogger(__name__)

# Validation allowlists. ROS 2 graph names are alnum plus _ / ~ (and the {ns}
# substitution braces); interface types are pkg/(msg|srv)/Name. Rejecting
# everything else keeps untrusted, agent-supplied strings from carrying
# unexpected characters into the rclpy graph API or type-resolution layer.
_NAME_RE = re.compile(r"^[A-Za-z0-9_/~{}]+$")
_TYPE_RE = re.compile(r"^[A-Za-z0-9_]+/[A-Za-z0-9_]+/[A-Za-z0-9_]+$")

# Which numeric options each action actually consumes. An action that reads none
# of them (``status``, the ``list_*`` queries, ``info``) must not be refused for
# a value it never looks at, so the guard below is driven by this table rather
# than validating the whole signature unconditionally.
_ACTION_NUMERIC_OPTIONS: dict[str, tuple[str, ...]] = {
    "echo": ("timeout", "count"),
    "publish": ("count", "rate"),
    "service_call": ("timeout",),
    "action_send_goal": ("timeout",),
}

_INSTALL_HINT = (
    "rclpy is not importable - source a ROS 2 distro before launching the agent "
    "(e.g. 'source /opt/ros/jazzy/setup.bash'). rclpy/rosidl_runtime_py ship with "
    "a system ROS 2 install (apt / RoboStack / conda) and are not on PyPI."
)


# --------------------------------------------------------------------------
# In-process rclpy backend: a single long-lived node + executor, reused across
# tool calls and guarded by a lock (rclpy spinning is not re-entrant).
# --------------------------------------------------------------------------


class _RosBackend:
    """Lazily-initialised, process-wide rclpy node + single-threaded executor.

    rclpy.init() is global per-process, so this is a singleton. All access is
    serialised through ``_lock`` because spinning the executor from concurrent
    threads is unsafe.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._node = None
        self._executor = None
        self._available: bool | None = None

    def available(self) -> bool:
        """Return True if rclpy/rosidl_runtime_py can be imported."""
        if self._available is None:
            try:
                import rclpy  # noqa: F401
                import rosidl_runtime_py.utilities  # noqa: F401

                self._available = True
            except ImportError:
                # rclpy absent: no ROS 2 sourced in this interpreter. Callers
                # surface _INSTALL_HINT; nothing to fall back to.
                self._available = False
        return self._available

    def _ensure_node(self):
        """Initialise rclpy + the shared node/executor on first use."""
        if self._node is not None:
            return self._node
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("strands_robots_use_ros")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        return self._node

    def spin_for(self, predicate, timeout: float) -> None:
        """Spin the executor until ``predicate()`` is true or timeout elapses."""
        executor = self._executor
        if executor is None:
            raise RuntimeError("rclpy node not initialised - call _ensure_node first")
        deadline = time.time() + timeout
        while not predicate() and time.time() < deadline:
            executor.spin_once(timeout_sec=0.05)

    @property
    def lock(self) -> threading.RLock:
        return self._lock


_backend = _RosBackend()


def _get_message(type_str: str):
    from rosidl_runtime_py.utilities import get_message

    return get_message(type_str)


def _get_service(type_str: str):
    from rosidl_runtime_py.utilities import get_service

    return get_service(type_str)


def _get_action(type_str: str):
    from rosidl_runtime_py.utilities import get_action

    return get_action(type_str)


def _msg_to_dict(msg) -> dict[str, Any]:
    from rosidl_runtime_py.convert import message_to_ordereddict

    return dict(message_to_ordereddict(msg))


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": f"use_ros: {text}"}]}


# --------------------------------------------------------------------------
# Graph introspection (rclpy node API - no CLI).
# --------------------------------------------------------------------------


def _list_topics() -> str:
    node = _backend._ensure_node()
    # Let discovery settle briefly so freshly-started graphs are visible.
    _backend.spin_for(lambda: False, 0.2)
    lines = [f"{name} [{', '.join(types)}]" for name, types in sorted(node.get_topic_names_and_types())]
    return "\n".join(lines)


def _list_nodes() -> str:
    node = _backend._ensure_node()
    _backend.spin_for(lambda: False, 0.2)
    names = [
        (f"{ns.rstrip('/')}/{n}" if ns not in ("", "/") else f"/{n}")
        for n, ns in sorted(node.get_node_names_and_namespaces())
    ]
    return "\n".join(names)


def _list_services() -> str:
    node = _backend._ensure_node()
    _backend.spin_for(lambda: False, 0.2)
    lines = [f"{name} [{', '.join(types)}]" for name, types in sorted(node.get_service_names_and_types())]
    return "\n".join(lines)


def _resolve_topic_type(topic: str) -> str | None:
    node = _backend._ensure_node()
    _backend.spin_for(lambda: False, 0.2)
    for name, types in node.get_topic_names_and_types():
        if name == topic and types:
            return types[0]
    return None


def _info(target: str) -> str | None:
    node = _backend._ensure_node()
    _backend.spin_for(lambda: False, 0.2)
    for name, types in node.get_topic_names_and_types():
        if name == target:
            n_pub = node.count_publishers(target)
            n_sub = node.count_subscribers(target)
            return f"topic info {target}:\n  type(s): {', '.join(types)}\n  publishers: {n_pub}\n  subscribers: {n_sub}"
    for name, types in node.get_service_names_and_types():
        if name == target:
            return f"service info {target}:\n  type(s): {', '.join(types)}"
    return None


# --------------------------------------------------------------------------
# Pub / sub / service-call (rclpy directly - real message classes, no eval).
# --------------------------------------------------------------------------


def _echo(topic: str, msg_type: str, timeout: float, count: int) -> list[dict[str, Any]]:
    from rosidl_runtime_py.set_message import set_message_fields  # noqa: F401 (validates import path)

    node = _backend._ensure_node()
    msg_cls = _get_message(msg_type)
    received: list[dict[str, Any]] = []
    sub = node.create_subscription(msg_cls, topic, lambda m: received.append(_msg_to_dict(m)), 10)
    try:
        _backend.spin_for(lambda: len(received) >= count, timeout)
    finally:
        node.destroy_subscription(sub)
    return received[:count]


def _publish(topic: str, msg_type: str, fields: dict[str, Any], count: int, rate: float) -> None:
    from rosidl_runtime_py.set_message import set_message_fields

    node = _backend._ensure_node()
    msg_cls = _get_message(msg_type)
    pub = node.create_publisher(msg_cls, topic, 10)
    msg = msg_cls()
    set_message_fields(msg, fields)
    try:
        # Brief settle so subscribers discover the publisher before the first send.
        _backend.spin_for(lambda: False, 0.3)
        period = 1.0 / rate if rate > 0 else 0.0
        for _ in range(count):
            pub.publish(msg)
            if period:
                _backend.spin_for(lambda: False, period)
    finally:
        node.destroy_publisher(pub)


def _service_call(service: str, srv_type: str, fields: dict[str, Any], timeout: float) -> dict[str, Any]:
    from rosidl_runtime_py.set_message import set_message_fields

    node = _backend._ensure_node()
    srv_cls = _get_service(srv_type)
    client = node.create_client(srv_cls, service)
    try:
        ready = client.wait_for_service(timeout_sec=timeout)
        if not ready:
            raise TimeoutError(f"service {service} not available within {timeout}s")
        req = srv_cls.Request()
        set_message_fields(req, fields)
        future = client.call_async(req)
        _backend.spin_for(lambda: future.done(), timeout)
        if not future.done() or future.result() is None:
            raise TimeoutError(f"service call to {service} timed out after {timeout}s")
        return _msg_to_dict(future.result())
    finally:
        node.destroy_client(client)


# --------------------------------------------------------------------------
# Actions (rclpy.action - goal / feedback / result, timeout-cancelled).
# --------------------------------------------------------------------------

# Terminal GoalStatus codes -> names (action_msgs/msg/GoalStatus constants).
_GOAL_STATUS_NAMES = {
    0: "UNKNOWN",
    1: "ACCEPTED",
    2: "EXECUTING",
    3: "CANCELING",
    4: "SUCCEEDED",
    5: "CANCELED",
    6: "ABORTED",
}

# Cap on feedback samples retained per goal. Long-running goals (Nav2 emits
# feedback at control-loop rate) would otherwise grow an unbounded list and
# flood the agent's context with thousands of near-identical dicts.
_FEEDBACK_LIMIT = 5


def _list_actions() -> str:
    from rclpy.action import get_action_names_and_types

    node = _backend._ensure_node()
    _backend.spin_for(lambda: False, 0.2)
    lines = [f"{name} [{', '.join(types)}]" for name, types in sorted(get_action_names_and_types(node))]
    return "\n".join(lines)


def _action_send_goal(
    action_name: str,
    action_type: str,
    fields: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Send a goal, spin until the terminal result or ``timeout``, then return.

    A single overall deadline governs server discovery, goal acceptance, and
    result delivery. If the deadline expires after the goal was accepted, a
    cancel request is sent *before* raising, so a timed-out ``use_ros`` call
    never leaves a physical robot executing an orphaned goal - the same
    fail-safe posture as ``HardwareRosBridge``'s reject-whole command clamping.
    """
    from rclpy.action import ActionClient
    from rosidl_runtime_py.set_message import set_message_fields

    node = _backend._ensure_node()
    action_cls = _get_action(action_type)
    client = ActionClient(node, action_cls, action_name)
    deadline = time.monotonic() + timeout

    feedback: list[dict[str, Any]] = []

    def _on_feedback(fb: Any) -> None:
        # Keep first (limit - 1) plus always the most recent sample, so the
        # agent sees both how the goal started and where it currently is.
        entry = _msg_to_dict(fb.feedback)
        if len(feedback) < _FEEDBACK_LIMIT:
            feedback.append(entry)
        else:
            feedback[-1] = entry

    def _remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    goal_handle = None
    try:
        if not client.wait_for_server(timeout_sec=_remaining()):
            raise TimeoutError(f"action server {action_name} not available within {timeout}s")

        goal = action_cls.Goal()
        set_message_fields(goal, fields)

        send_future = client.send_goal_async(goal, feedback_callback=_on_feedback)
        _backend.spin_for(lambda: send_future.done(), _remaining())
        if not send_future.done():
            raise TimeoutError(f"goal to {action_name} not acknowledged within {timeout}s")
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise ValueError(f"goal rejected by action server {action_name}")

        result_future = goal_handle.get_result_async()
        _backend.spin_for(lambda: result_future.done(), _remaining())
        if not result_future.done() or result_future.result() is None:
            # Deadline hit mid-execution: cancel before surfacing the timeout
            # so the robot stops pursuing the goal.
            cancel_future = goal_handle.cancel_goal_async()
            _backend.spin_for(lambda: cancel_future.done(), 2.0)
            raise TimeoutError(f"goal to {action_name} did not finish within {timeout}s (cancel requested)")

        wrapped = result_future.result()
        status = int(wrapped.status)
        return {
            "goal_status": _GOAL_STATUS_NAMES.get(status, str(status)),
            "result": _msg_to_dict(wrapped.result),
            "feedback": feedback,
        }
    finally:
        client.destroy()


@tool
def use_ros(
    action: str,
    topic: str | None = None,
    service: str | None = None,
    action_name: str | None = None,
    type: str | None = None,
    fields: dict[str, Any] | None = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
) -> dict[str, Any]:
    """Universal ROS 2 tool - in-process rclpy, dynamic types, no shelling out.

    Args:
        action: One of ``status``, ``list_topics``, ``list_nodes``,
            ``list_services``, ``list_actions``, ``info``, ``echo``,
            ``publish``, ``service_call``, ``action_send_goal``.
        topic: Topic name (``echo``, ``publish``, ``info``).
        service: Service name (``service_call``, ``info``).
        action_name: Action server name (``action_send_goal``), e.g.
            ``/navigate_to_pose``.
        type: Fully-qualified interface type, e.g. ``geometry_msgs/msg/Twist``,
            ``turtlesim/srv/Spawn``, or ``nav2_msgs/action/NavigateToPose``.
            Auto-resolved for ``echo`` when omitted.
        fields: JSON field dict applied with ``set_message_fields`` (``publish``,
            ``service_call``, ``action_send_goal``). Booleans and nulls are
            preserved - the dict is passed straight to rclpy, never serialised
            through source.
        timeout: Seconds to wait for samples / a service / an action result.
            For ``action_send_goal`` this is the end-to-end budget (discovery +
            acceptance + execution); size it to the goal (e.g. 120 for a Nav2
            navigation), and note the goal is cancelled when it expires.
            A positive finite number of seconds.
        count: Number of messages to echo or publish. A positive integer;
            it is consumed as a ``range()`` bound, so ``0`` publishes nothing
            and a float or a numeric string cannot be honored.
        rate: Publish rate in Hz. A positive finite number - the inter-message
            period is ``1 / rate``, so ``0``, a negative value, ``nan`` and
            ``inf`` all leave the burst unthrottled rather than paced.

    Returns:
        A Strands tool result dict ``{"status": ..., "content": [{"text": ...}]}``.
    """
    fields = fields or {}

    # Validate agent-supplied names before they reach the rclpy graph/type API.
    if topic is not None and not _NAME_RE.match(topic):
        return _err(f"invalid topic name: {topic!r}")
    if service is not None and not _NAME_RE.match(service):
        return _err(f"invalid service name: {service!r}")
    if action_name is not None and not _NAME_RE.match(action_name):
        return _err(f"invalid action name: {action_name!r}")
    if type is not None and not _TYPE_RE.match(type):
        return _err(f"invalid interface type: {type!r} (expected pkg/msg/Name or pkg/srv/Name)")

    # Numeric options are checked here, alongside the names and ahead of the
    # backend probe, so the same caller mistake is reported identically whether
    # or not rclpy is installed - and so a refusal happens before a publisher
    # joins the graph.
    numeric_error = numeric_option_error(action, _ACTION_NUMERIC_OPTIONS, timeout=timeout, count=count, rate=rate)
    if numeric_error:
        return _err(numeric_error)

    if action == "status":
        if _backend.available():
            return _ok("backend: rclpy (in-process)")
        return _ok("backend: none - " + _INSTALL_HINT)

    if not _backend.available():
        return _err(_INSTALL_HINT)

    try:
        with _backend.lock:
            if action == "list_topics":
                return _ok(_list_topics())

            if action == "list_nodes":
                return _ok(_list_nodes())

            if action == "list_services":
                return _ok(_list_services())

            if action == "list_actions":
                return _ok(_list_actions())

            if action == "action_send_goal":
                if not action_name or not type:
                    return _err("action_send_goal requires action_name and type")
                import json

                outcome = _action_send_goal(action_name, type, fields, timeout)
                return _ok(f"goal to {action_name} finished:\n{json.dumps(outcome, indent=2, default=str)}")

            if action == "info":
                target = topic or service
                if not target:
                    return _err("info requires topic or service")
                out = _info(target)
                return _ok(out) if out else _err(f"no info for {target}")

            if action == "echo":
                if not topic:
                    return _err("echo requires topic")
                msg_type = type or _resolve_topic_type(topic)
                if not msg_type:
                    return _err(f"cannot resolve type for {topic}; pass type=pkg/msg/Name")
                samples = _echo(topic, msg_type, timeout, count)
                import json

                return _ok(f"echo {topic} ({msg_type}):\n{json.dumps(samples, indent=2, default=str)}")

            if action == "publish":
                if not topic or not type:
                    return _err("publish requires topic and type")
                _publish(topic, type, fields, count, rate)
                return _ok(f"published {count} message(s) to {topic}")

            if action == "service_call":
                if not service or not type:
                    return _err("service_call requires service and type")
                import json

                resp = _service_call(service, type, fields, timeout)
                return _ok(f"response:\n{json.dumps(resp, indent=2, default=str)}")

            return _err(f"unknown action: {action}")
    except TimeoutError as exc:
        return _err(str(exc))
    except (ImportError, KeyError, AttributeError, ValueError, TypeError) as exc:
        # Type resolution / field-set errors surface as a clean tool error
        # rather than a raised exception that bypasses the structured result.
        # ImportError (incl. ModuleNotFoundError) is the real failure mode when a
        # valid-shaped type names a package that is not installed: get_message ->
        # import_message_from_namespaced_type -> importlib.import_module raises it.
        return _err(f"{action} failed: {exc}")
