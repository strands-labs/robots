#!/usr/bin/env python3
"""Pure-RTPS ROS 2 tool - join a ROS 2 graph as a DDS participant, no rclpy.

Where ``use_ros`` is a *client* that needs a sourced ROS 2 distro (rclpy),
``use_rtps`` is a *participant* built on the pip-installable ``cyclonedds``
binding alone. It speaks RTPS - the DDS wire protocol every ROS 2 distro uses -
so it interoperates with Humble, Jazzy, Rolling, ... uniformly, with nothing
installed but a pip wheel.

The headline capability: an RTPS participant can **act as a robot**. It can
advertise and publish a topic that a real ROS 2 node (rviz, nav2, a teleop
joystick) will consume, and subscribe to command topics - indistinguishable on
the wire from physical hardware.

Actions:
    status      - report whether the cyclonedds backend is available.
    types       - list the ROS 2 message types in the local IDL bundle.
    advertise   - create a publisher for a topic (start "being" a robot output).
    publish     - publish N messages built from a JSON field dict.
    subscribe   - create a subscription and buffer samples.
    echo        - subscribe and return the next N samples as JSON.

Scope (v1): topics only. Services and actions require the ROS 2 request/reply-
over-DDS protocol and land in a focused follow-up.

Type coverage: bounded by the local IDL bundle (``strands_robots.rtps.idl``) -
publishing needs a local type definition. ``use_rtps(action="types")`` lists
what is available; anything else needs rclpy (use ``use_ros``).

Examples:
    use_rtps(action="status")
    use_rtps(action="types")
    use_rtps(action="advertise", topic="/turtle1/cmd_vel", type="geometry_msgs/msg/Twist")
    use_rtps(action="publish", topic="/turtle1/cmd_vel",
             type="geometry_msgs/msg/Twist",
             fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}}, count=15)
    use_rtps(action="echo", topic="/turtle1/cmd_vel",
             type="geometry_msgs/msg/Twist", count=2, timeout=2.0)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import threading
import time
import typing
from typing import Any

from strands import tool

from strands_robots.tools._numeric_options import numeric_option_error

logger = logging.getLogger(__name__)

# ROS 2 graph names: absolute, alnum/_ segments. Validate before mangling so a
# malformed agent-supplied name fails with a clear message.
_TOPIC_RE = re.compile(r"^/[A-Za-z0-9_/]*[A-Za-z0-9_]$")
_TYPE_RE = re.compile(r"^[A-Za-z0-9_]+/(msg|srv|action)/[A-Za-z0-9_]+$")

# Which numeric options each action actually consumes. ``status``, ``types``,
# ``advertise`` and ``subscribe`` read none of them, so the guard below is driven
# by this table rather than validating the whole signature unconditionally - a
# caller is never refused for a value the requested action never looks at.
_ACTION_NUMERIC_OPTIONS: dict[str, tuple[str, ...]] = {
    "publish": ("count", "rate"),
    "echo": ("timeout", "count"),
}


class _RtpsBackend:
    """Process-wide cyclonedds participant + per-topic readers/writers.

    A single DomainParticipant is shared (cheap, and keeps one presence on the
    graph). Writers and readers are cached per (topic, type) so repeated
    publish/echo calls reuse the same DDS entities - and a long-lived advertised
    writer keeps "being a robot" between tool calls. All access is serialised.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._participant: Any = None
        self._writers: dict[tuple[str, str], Any] = {}
        self._readers: dict[tuple[str, str], Any] = {}
        self._available: bool | None = None

    def available(self) -> bool:
        try:
            from strands_robots.rtps.idl import have_cyclonedds

            return have_cyclonedds()
        except ImportError:
            return False

    def _participant_obj(self) -> Any:
        if self._participant is None:
            from cyclonedds.domain import DomainParticipant

            self._participant = DomainParticipant()
        return self._participant

    def writer(self, ros_topic: str, ros_type: str) -> Any:
        """Get-or-create a DataWriter for (topic, type), ROS-mangled."""
        key = (ros_topic, ros_type)
        if key not in self._writers:
            from cyclonedds.pub import DataWriter
            from cyclonedds.topic import Topic

            from strands_robots.rtps.idl import get_type
            from strands_robots.rtps.mangling import dds_topic_name

            idl_cls = get_type(ros_type)
            topic = Topic(self._participant_obj(), dds_topic_name(ros_topic), idl_cls)
            self._writers[key] = DataWriter(self._participant_obj(), topic)
        return self._writers[key]

    def reader(self, ros_topic: str, ros_type: str) -> Any:
        """Get-or-create a DataReader for (topic, type), ROS-mangled."""
        key = (ros_topic, ros_type)
        if key not in self._readers:
            from cyclonedds.sub import DataReader
            from cyclonedds.topic import Topic

            from strands_robots.rtps.idl import get_type
            from strands_robots.rtps.mangling import dds_topic_name

            idl_cls = get_type(ros_type)
            topic = Topic(self._participant_obj(), dds_topic_name(ros_topic), idl_cls)
            self._readers[key] = DataReader(self._participant_obj(), topic)
        return self._readers[key]

    @property
    def lock(self) -> threading.RLock:
        return self._lock


_backend = _RtpsBackend()


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": f"use_rtps: {text}"}]}


def _sample_to_dict(sample: Any) -> Any:
    """Recursively convert an IDL dataclass sample to a plain dict."""
    if dataclasses.is_dataclass(sample) and not isinstance(sample, type):
        return {f.name: _sample_to_dict(getattr(sample, f.name)) for f in dataclasses.fields(sample)}
    if isinstance(sample, (list, tuple)):
        return [_sample_to_dict(x) for x in sample]
    return sample


def _resolve_field_types(idl_cls: Any) -> dict[str, Any]:
    """Return {field_name: resolved_type} for a dataclass.

    ``from __future__ import annotations`` (and cyclonedds's own annotations)
    make ``dataclasses.Field.type`` a *string*, so nested-message detection must
    resolve real types via ``typing.get_type_hints``. Falls back to the raw
    ``Field.type`` if hint resolution fails (e.g. exotic cyclonedds aliases).
    """
    try:
        hints = typing.get_type_hints(idl_cls)
    except Exception:
        hints = {}
    return {f.name: hints.get(f.name, f.type) for f in dataclasses.fields(idl_cls)}


def _build_sample(idl_cls: Any, fields: dict[str, Any]) -> Any:
    """Construct an IDL dataclass instance from a (possibly nested) field dict.

    Nested message fields are built recursively from their annotated dataclass
    type, so ``{"linear": {"x": 2.0}}`` becomes ``Twist(linear=Vector3(x=2.0))``.
    Unknown field names raise so a typo fails loudly rather than silently
    publishing zeros.
    """
    kwargs: dict[str, Any] = {}
    field_types = _resolve_field_types(idl_cls)
    for name, value in fields.items():
        if name not in field_types:
            raise ValueError(f"unknown field {name!r} for {idl_cls.__name__} (have: {sorted(field_types)})")
        ftype = field_types[name]
        if isinstance(value, dict) and dataclasses.is_dataclass(ftype):
            kwargs[name] = _build_sample(ftype, value)
        else:
            kwargs[name] = value
    return idl_cls(**kwargs)


@tool
def use_rtps(
    action: str,
    topic: str | None = None,
    type: str | None = None,
    fields: dict[str, Any] | None = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
) -> dict[str, Any]:
    """Pure-RTPS ROS 2 participant tool - no rclpy, all ROS 2 distros.

    Args:
        action: One of ``status``, ``types``, ``advertise``, ``publish``,
            ``subscribe``, ``echo``.
        topic: ROS 2 topic name, absolute (e.g. ``/turtle1/cmd_vel``).
        type: ROS 2 interface type in the IDL bundle (e.g.
            ``geometry_msgs/msg/Twist``). List with ``action="types"``.
        fields: JSON field dict for ``publish``; nested message fields are built
            recursively. Booleans and nulls are preserved (plain Python values).
        timeout: Seconds to wait for samples (``echo``). A positive finite
            number.
        count: Number of messages to publish or samples to echo. A positive
            integer; it is consumed as a ``range()`` bound, so ``0`` publishes
            nothing and a float or a numeric string cannot be honored.
        rate: Publish rate in Hz. A positive finite number - the inter-message
            period is ``1 / rate``, so ``0``, a negative value, ``nan`` and
            ``inf`` all leave the burst unthrottled rather than paced.

    Returns:
        A Strands tool result dict ``{"status": ..., "content": [{"text": ...}]}``.
    """
    fields = fields or {}

    if topic is not None and not _TOPIC_RE.match(topic):
        return _err(f"invalid topic name: {topic!r} (expected absolute, e.g. /turtle1/cmd_vel)")
    if type is not None and not _TYPE_RE.match(type):
        return _err(f"invalid interface type: {type!r} (expected pkg/msg/Name)")

    # Numeric options are checked here, alongside the names and ahead of the
    # backend probe, so the same caller mistake is reported identically whether
    # or not cyclonedds is installed - and so a refusal happens before a writer
    # joins the graph.
    numeric_error = numeric_option_error(action, _ACTION_NUMERIC_OPTIONS, timeout=timeout, count=count, rate=rate)
    if numeric_error:
        return _err(numeric_error)

    if action == "status":
        if _backend.available():
            return _ok("backend: cyclonedds (RTPS participant) - no rclpy / ROS 2 distro needed")
        from strands_robots.rtps.idl import _INSTALL_HINT

        return _ok("backend: none - " + _INSTALL_HINT)

    if not _backend.available():
        from strands_robots.rtps.idl import _INSTALL_HINT

        return _err(_INSTALL_HINT)

    try:
        from strands_robots.rtps.idl import REGISTRY, get_type

        if action == "types":
            return _ok("RTPS IDL bundle types:\n" + "\n".join(sorted(REGISTRY)))

        with _backend.lock:
            if action == "advertise":
                if not topic or not type:
                    return _err("advertise requires topic and type")
                get_type(type)  # validate type is in the bundle before creating the writer
                _backend.writer(topic, type)
                return _ok(f"advertised {topic} ({type}) - now a publisher on the ROS 2 graph")

            if action == "publish":
                if not topic or not type:
                    return _err("publish requires topic and type")
                idl_cls = get_type(type)
                sample = _build_sample(idl_cls, fields)
                writer = _backend.writer(topic, type)
                # Brief settle so freshly-matched readers receive the first sample.
                time.sleep(0.3)
                period = 1.0 / rate if rate > 0 else 0.0
                for _ in range(count):
                    writer.write(sample)
                    if period:
                        time.sleep(period)
                return _ok(f"published {count} message(s) to {topic} ({type})")

            if action in ("subscribe", "echo"):
                if not topic or not type:
                    return _err(f"{action} requires topic and type")
                reader = _backend.reader(topic, type)
                if action == "subscribe":
                    return _ok(f"subscribed to {topic} ({type})")
                # echo: poll the reader until count samples arrive or timeout.
                samples: list[Any] = []
                deadline = time.time() + timeout
                while len(samples) < count and time.time() < deadline:
                    for sample in reader.take(N=count - len(samples)):
                        samples.append(_sample_to_dict(sample))
                    if len(samples) < count:
                        time.sleep(0.05)
                return _ok(f"echo {topic} ({type}):\n{json.dumps(samples, indent=2, default=str)}")

            return _err(f"unknown action: {action}")
    except ImportError as exc:
        return _err(str(exc))
    except (KeyError, ValueError, AttributeError, TypeError) as exc:
        return _err(f"{action} failed: {exc}")
