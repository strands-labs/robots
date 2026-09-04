"""RTPS mesh robot - act as (or drive) a ROS 2 robot with no rclpy.

:class:`RtpsRobot` is the pure-RTPS sibling of :class:`RosBridgedRobot`. Where
the ROS bridge forwards to ``use_ros`` (which needs a sourced ROS 2 distro),
``RtpsRobot`` forwards to ``use_rtps`` - a DDS participant built on the
pip-installable ``cyclonedds`` binding alone. It therefore works on macOS,
Jetson, and CI with nothing but a pip wheel, and interoperates with every ROS 2
distro over RTPS.

Because an RTPS participant publishes real DDS samples, an :class:`RtpsRobot`
can do something the client-only bridge cannot: **act as a robot**. Advertise a
``cmd_vel`` listener or publish ``/odom`` / ``/joint_states`` and a real ROS 2
stack (rviz, nav2) treats the agent as hardware.

Typical usage::

    from strands import Agent
    from strands_robots.mesh import RtpsRobot

    turtle = RtpsRobot.from_rtps(
        node_name="turtlesim",
        cmd_vel_topic="/turtle1/cmd_vel",
    )
    turtle.drive(linear=1.0, duration=1.5)   # publishes Twist over RTPS
    agent = Agent(tools=turtle.tools)
    agent("drive forward for two seconds")

Scope mirrors ``use_rtps``: topics only, and types bounded by the IDL bundle
(``geometry_msgs/msg/Twist`` for ``drive``). This transport has no services and
no actions, so an :class:`RtpsRobot` exposes no ``init_services`` handshake and
no goal-level navigation - the base class asks the transport what it can do
rather than assuming. Pose/scan read-back needs those messages in the bundle;
until they are added, use ``RosBridgedRobot`` for echo.
"""

from __future__ import annotations

import re
from typing import Any, cast

from strands.types.tools import ToolContext

from strands_robots.mesh._mobile_base import MobileBaseRobot
from strands_robots.tools.use_rtps import use_rtps
from strands_robots.utils import partial_construction_repr

_TWIST_TYPE = "geometry_msgs/msg/Twist"
# ``use_rtps`` writes to a DDS topic directly, so a topic must be absolute -
# a stricter grammar than the ROS 2 bridge's, which also accepts relative and
# private (``~``) names for rclpy to resolve.
_RTPS_TOPIC_RE = re.compile(r"^/[A-Za-z0-9_/]*[A-Za-z0-9_]\Z")
_RTPS_NAME_RE = re.compile(r"^[A-Za-z0-9_/~]+\Z")


class _UseRtpsTransport:
    """Transport that forwards to the pure-DDS ``use_rtps`` tool.

    Resolves ``use_rtps`` through this module's globals on every call so the
    symbol stays patchable at ``strands_robots.mesh.rtps_robot.use_rtps``.

    Deliberately implements only ``publish`` and ``echo``: ``use_rtps`` has no
    service or action surface, and declaring ``service_call`` here would let a
    caller wire an ``init_services`` handshake that could never run.

    ``use_rtps`` gates its commanding actions behind the operator approval in
    ``strands_robots.tools._command_gate``, so :meth:`publish` forwards the
    ``tool_context`` the protocol carries. A transport that silently dropped an
    operator decision would be the same class of defect as one that declared a
    capability it does not have: the prompt would be unreachable and the command
    would fail closed with the blanket bypass as its only remedy.
    """

    twist_type = _TWIST_TYPE

    def publish(
        self,
        *,
        topic: str,
        type: str,
        fields: dict[str, Any],
        count: int,
        rate: float,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        # ``publish`` is a commanding action, so the operator decision has to
        # reach ``use_rtps``: the base carries the context this far for every
        # transport, and dropping it here would make the gate fail closed with
        # no prompt.
        return use_rtps(
            action="publish", topic=topic, type=type, fields=fields, count=count, rate=rate, tool_context=tool_context
        )

    def echo(self, *, topic: str, type: str | None, count: int, timeout: float) -> dict[str, Any]:
        return use_rtps(action="echo", topic=topic, type=type, count=count, timeout=timeout)

    def advertise(self, *, topic: str, type: str) -> dict[str, Any]:
        return use_rtps(action="advertise", topic=topic, type=type)


class RtpsRobot(MobileBaseRobot):
    """A ROS 2 robot driven over pure RTPS (no rclpy), exposed as a strands robot.

    The robot owns no DDS state of its own; every method forwards to
    :func:`use_rtps`, which manages the shared participant and cached writers.
    Safe to construct without cyclonedds present - errors surface only when a
    method is called and the backend is unavailable.

    Args:
        node_name: Identifier used to name this robot's agent tools
            (``drive_<node_name>`` etc.). Need not match any ROS 2 node name.
        cmd_vel_topic: Velocity-command topic to publish ``Twist`` on.
        cmd_vel_type: Interface type for ``cmd_vel_topic`` (default
            ``geometry_msgs/msg/Twist``).
        publish_rate: Default rate (Hz) for multi-message :meth:`drive` calls.
            Must be > 0 and finite: :meth:`drive` multiplies it by ``duration``
            to size the message burst and ``use_rtps`` publishes at ``1 / rate``,
            so a non-positive rate removes the pacing entirely rather than
            slowing it. Raises ``ValueError`` at construction otherwise.
        max_linear: Optional linear-velocity clamp (m/s). Unset by default -
            an RTPS peer drives arbitrary third-party robots whose limits this
            class cannot know.
        max_angular: Optional angular-velocity clamp (rad/s). Unset by default.
        max_duration: Optional cap on a single :meth:`drive` hold, in seconds.
    """

    _NAME_RE = _RTPS_NAME_RE
    _TOPIC_RE = _RTPS_TOPIC_RE
    # The two seams have different grammars here, so they carry different
    # hints. ``node_name`` names this robot's agent tools rather than a node on
    # the robot's graph, so the inherited sentence describes it correctly; a
    # topic is written to DDS directly and must be absolute.
    _TOPIC_HINT = (
        " (expected an absolute topic like /turtle1/cmd_vel - use_rtps writes to DDS "
        "directly, so a relative or ~ name has no resolver)"
    )

    def __init__(
        self,
        node_name: str,
        cmd_vel_topic: str,
        *,
        cmd_vel_type: str = _TWIST_TYPE,
        publish_rate: float = 10.0,
        max_linear: float | None = None,
        max_angular: float | None = None,
        max_duration: float | None = None,
    ) -> None:
        super().__init__(
            node_name,
            cmd_vel_topic,
            _UseRtpsTransport(),
            cmd_vel_type=cmd_vel_type,
            max_linear=max_linear,
            max_angular=max_angular,
            max_duration=max_duration,
            publish_rate=publish_rate,
        )

    @classmethod
    def from_rtps(
        cls,
        node_name: str,
        cmd_vel_topic: str,
        **kwargs: Any,
    ) -> RtpsRobot:
        """Construct an RTPS robot from ROS 2 topic wiring.

        Keyword-style alternate constructor mirroring
        :meth:`RosBridgedRobot.from_ros`. The top-level ``Robot`` is a factory
        *function* (not a class), so the alternate constructor lives here where
        it is type-safe and discoverable.
        """
        return cls(node_name, cmd_vel_topic, **kwargs)

    def advertise(self) -> dict[str, Any]:
        """Create the ``cmd_vel`` publisher up front (appear on the ROS 2 graph).

        RTPS-only: a DDS participant can announce a writer before it has
        anything to say, which is what makes an agent visible to ``ros2 topic
        list`` and rviz. No other transport has an equivalent, so this stays on
        the subclass rather than becoming a base-class capability of one.
        """
        return cast(_UseRtpsTransport, self.transport).advertise(topic=self.cmd_vel_topic, type=self.cmd_vel_type)

    def __repr__(self) -> str:
        try:
            return (
                f"RtpsRobot(node_name={self.node_name!r}, cmd_vel_topic={self.cmd_vel_topic!r}, "
                f"cmd_vel_type={self.cmd_vel_type!r})"
            )
        except AttributeError:
            return partial_construction_repr(self)
