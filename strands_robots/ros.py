"""The in-process ``rclpy`` transport every caller in this package forwards through.

The ROS 2 mechanics behind ``use_ros``: one long-lived node and executor for the
process, the dynamic type resolution ``rosidl_runtime_py`` provides, the graph
introspection, the pub/sub/service calls and the action-goal lifecycle with its
timeout cancel. Three surfaces need them - the agent-facing
:mod:`~strands_robots.tools.use_ros` tool,
:class:`~strands_robots.mesh.ros_bridge.RosBridgedRobot` and
:class:`~strands_robots.mesh.ackermann_robot.AckermannRosRobot` - so they live
here rather than inside one of the three.

They lived in the tool, and both mesh robots imported the ``@tool`` to reach
them. That is an import from the ``drivers|mesh`` layer up into ``tools``: a
library class whose transport was only reachable through an agent entry point, so
a programmatic caller paid for the tool decorator, the tool's argument envelope
and the tool's module import to publish a ``Twist``.

Requirements:
    ``rclpy`` and ``rosidl_runtime_py`` must be importable in this interpreter.
    These ship with a sourced system ROS 2 distro (apt / RoboStack / conda) and
    are **not** on PyPI, so they cannot be ``pip install``ed and are not pinned
    in ``pyproject.toml`` (the ``[ros2]`` extra only carries the pip-installable
    ``cyclonedds`` RMW binding). Source a ROS 2 environment before launching the
    agent - e.g. ``source /opt/ros/jazzy/setup.bash`` - and ``rclpy`` becomes
    importable. When it is absent, every action returns a clear, actionable error
    instead of raising.

Message and service types are resolved dynamically through
``rosidl_runtime_py`` (``get_message`` / ``get_service``), so any interface
installed in the ROS 2 environment works with no static registry. Field payloads
are passed as plain JSON dicts and applied with ``set_message_fields`` - the
standard ROS 2 idiom.

**The operator gate is an argument, not a default.** A ``publish``, a
``service_call`` or an ``action_send_goal`` onto a blocklisted surface has to
reach :func:`strands_robots._command_gate.gate_command` whichever caller asked -
that is the whole point of one shared blocklist - so :func:`ros_action` takes the
gate as a required ``gate`` argument and consults it at one fixed point in the
flow: after the backend probe, so a graph that cannot be reached never prompts;
after the verb's own required arguments are checked, so an incomplete call is
reported without asking an operator about it; and before the executor lock, so a
human deciding does not hold the lock every other caller of this transport - the
odometry read, the scan, the second robot on the same graph - has to take. A
caller cannot forget it, and cannot move it.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import Any

#: How a caller asks its operator about one command. Called with the verb and
#: the topic, service or action name the command is aimed at, and returns a
#: refusal to report or ``None`` to let the command through - the shape
#: :func:`strands_robots._command_gate.gate_command` already returns.
CommandGate = Callable[[str, str], str | None]

#: The name every caller keys its operator prompt and audit row with. The gate
#: builds the interrupt id ``<tool>-command-approval`` and the audit source
#: ``<tool>_tool`` from it, so two spellings would split one incident's audit
#: trail in two and an operator would be asked the same question under two
#: names. A ``Twist`` on ``/cmd_vel`` is the same physical command whether an
#: agent called ``use_ros`` or a :class:`RosBridgedRobot`'s drive tool did.
GATE_TOOL = "use_ros"

# Validation allowlists. ROS 2 graph names are alnum plus _ / ~ (and the {ns}
# substitution braces); interface types are pkg/(msg|srv)/Name. Rejecting
# everything else keeps untrusted, agent-supplied strings from carrying
# unexpected characters into the rclpy graph API or type-resolution layer.
_NAME_RE = re.compile(r"^[A-Za-z0-9_/~{}]+\Z")
_TYPE_RE = re.compile(r"^[A-Za-z0-9_]+/[A-Za-z0-9_]+/[A-Za-z0-9_]+\Z")

_INSTALL_HINT = (
    "rclpy is not importable - source a ROS 2 distro before launching the agent "
    "(e.g. 'source /opt/ros/jazzy/setup.bash'). rclpy/rosidl_runtime_py ship with "
    "a system ROS 2 install (apt / RoboStack / conda) and are not on PyPI."
)


def never_gated(kind: str, target: str) -> str | None:
    """The gate for a verb that carries no command: the reads and the probes.

    :func:`ros_action` requires a gate so that no caller can command a
    blocklisted surface by forgetting one, and it consults that gate for the
    three verbs that carry a command. ``status``, ``info``, ``echo`` and the
    ``list_*`` queries only read, so none of them can move a robot - an operator
    asked about one would be asked about nothing. This is the one spelling of
    that, so a caller wiring a read-only verb cannot invent a permissive gate of
    its own.

    Args:
        kind: The verb, ignored - a read is never gated whichever one it is.
        target: The topic, service or action name, ignored for the same reason.

    Returns:
        ``None``, always: nothing to refuse.
    """
    return None


# --------------------------------------------------------------------------
# In-process rclpy backend: a single long-lived node + executor, reused across
# calls and guarded by a lock (rclpy spinning is not re-entrant).
# --------------------------------------------------------------------------


class _RosBackend:
    """Lazily-initialised, process-wide rclpy node + single-threaded executor.

    rclpy.init() is global per-process, so this is a singleton. All access is
    serialised through ``_lock`` because spinning the executor from concurrent
    threads is unsafe.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # The rclpy node and executor: typed loosely because rclpy is an optional
        # system dependency, so the concrete classes are only importable once a
        # ROS 2 distro is sourced.
        self._node: Any = None
        self._executor: Any = None
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

    def _ensure_node(self) -> Any:
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

    def spin_for(self, predicate: Callable[[], bool], timeout: float) -> None:
        """Spin the executor until ``predicate()`` is true or ``timeout`` elapses.

        The budget is measured on ``time.monotonic()``, which is the clock the
        caller measured it with: :func:`_action_send_goal` derives every budget
        it passes here from a single ``time.monotonic()`` deadline. Measuring it
        on ``time.time()`` instead put one logical deadline on two clocks, and a
        wall-clock step during the wait -- an NTP correction, ``date -s``, a VM
        resume -- moved the inner one by the size of the step: forward, the wait
        ended early with the predicate still false, so a message that was
        arriving was reported as a timeout; backward, it ran past the budget by
        the size of the step. That also reached the cancel this module sends
        before surfacing a timeout, which needs the executor pumped to transmit,
        so a truncated wait could report a cancel it had not sent.
        """
        executor = self._executor
        if executor is None:
            raise RuntimeError("rclpy node not initialised - call _ensure_node first")
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)

    @property
    def lock(self) -> threading.RLock:
        return self._lock


_backend = _RosBackend()


def _get_message(type_str: str) -> Any:
    from rosidl_runtime_py.utilities import get_message

    return get_message(type_str)


def _get_service(type_str: str) -> Any:
    from rosidl_runtime_py.utilities import get_service

    return get_service(type_str)


def _get_action(type_str: str) -> Any:
    from rosidl_runtime_py.utilities import get_action

    return get_action(type_str)


def _msg_to_dict(msg: Any) -> dict[str, Any]:
    from rosidl_runtime_py.convert import message_to_ordereddict

    return dict(message_to_ordereddict(msg))


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": f"{GATE_TOOL}: {text}"}]}


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
            return str(types[0])
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
    cancel request is sent *before* raising, so a timed-out call never leaves a
    physical robot executing an orphaned goal - the same fail-safe posture as
    ``HardwareRosBridge``'s reject-whole command clamping.
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


def ros_action(
    action: str,
    *,
    topic: str | None = None,
    service: str | None = None,
    action_name: str | None = None,
    type: str | None = None,
    fields: dict[str, Any] | None = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
    gate: CommandGate,
) -> dict[str, Any]:
    """Run one ROS 2 action against the process-wide rclpy node.

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
        timeout: Seconds to wait for samples / a service / an action result. For
            ``action_send_goal`` this is the end-to-end budget (discovery +
            acceptance + execution), and the goal is cancelled when it expires.
        count: Number of messages to echo or publish, consumed as a ``range()``
            bound.
        rate: Publish rate in Hz; the inter-message period is ``1 / rate``.
        gate: The operator gate for the three verbs that carry a command,
            consulted with the verb and the topic, service or action name it is
            aimed at. Required: a caller that forgot it would carry a command to
            a blocklisted drive surface with no prompt, which is the defect
            :mod:`strands_robots._command_gate` exists to prevent. The numeric
            domains of ``timeout`` / ``count`` / ``rate`` belong to the caller
            too - an agent tool reports a malformed option, while a
            :class:`~strands_robots.mesh._mobile_base.MobileBaseRobot` has
            already refused one at its own seam.

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

    if action == "status":
        if _backend.available():
            return _ok("backend: rclpy (in-process)")
        return _ok("backend: none - " + _INSTALL_HINT)

    if not _backend.available():
        return _err(_INSTALL_HINT)

    # The operator gate is consulted here - after the backend probe, so a graph
    # that cannot be reached never prompts, and before the lock, so a human
    # deciding does not hold the process-wide rclpy executor lock and no
    # publisher joins the graph on a refusal. The lock serialises every caller of
    # this transport, so asking under it stalled an unrelated ``echo`` - the
    # odometry read of a second robot on the same graph - for as long as the
    # operator took to answer. Each condition mirrors its verb's own
    # required-argument check below, so an incomplete call is reported without
    # asking an operator about it.
    for kind, name in (("publish", topic), ("service_call", service), ("action_send_goal", action_name)):
        if action == kind and name and type:
            refusal = gate(kind, name)
            if refusal is not None:
                return _err(refusal)

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


__all__ = ["CommandGate", "GATE_TOOL", "never_gated", "ros_action"]
