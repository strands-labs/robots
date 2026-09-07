"""rosbridge mesh bridge - a ROS1 (or remote) robot as a strands robot, pure pip.

A :class:`RosbridgeRobot` wraps a mobile robot reachable through a rosbridge
WebSocket (``rosbridge_server`` + ``rosapi``) so an agent can drive it with the
same ``Agent(tools=robot.tools)`` pattern as every other strands robot - with
**no ROS environment on the agent's machine**. This is the bridge for ROS1
robots (rclpy cannot reach them) and for remote robots across a network.

Reference platform: the NASA Curiosity Mars rover Gazebo simulation
(ROS1 Noetic) - see :meth:`from_curiosity` and
``examples/rosbridge/curiosity_agent.py``.

All I/O forwards through :func:`strands_robots.tools.use_rosbridge.use_rosbridge`;
the class owns no transport state. rosbridge is unauthenticated by default -
use on trusted networks.

Typical usage::

    from strands import Agent
    from strands_robots.mesh import RosbridgeRobot

    rover = RosbridgeRobot.from_curiosity(host="localhost")
    rover.drive(linear=1.0, duration=3.0)
    print(rover.get_pose())

    agent = Agent(tools=rover.tools)
    agent("drive forward for three seconds, then report the odometry")
"""

from __future__ import annotations

from typing import Any

from strands import tool
from strands.types.tools import AgentTool, ToolContext

from strands_robots.mesh._mobile_base import LATCHED_VELOCITY, failed_halt_error
from strands_robots.mesh.ros_bridge import _check_topic
from strands_robots.tools.use_rosbridge import _HOST_RE, _transport_port_error, use_rosbridge
from strands_robots.utils import (
    dial_host_error,
    finite_number_error,
    partial_construction_repr,
    positive_finite_number_error,
    positive_whole_number_error,
    tcp_port_error,
)

_TWIST_TYPE = "geometry_msgs/Twist"


class RosbridgeRobot:
    """A rosbridge-reachable mobile robot exposed as a strands-controllable robot.

    Args:
        node_name: Identifier used to name this robot's agent tools.
        cmd_vel_topic: Velocity-command topic (``geometry_msgs/Twist``).
        odom_topic: Odometry/pose topic, read by :meth:`get_pose`.
        scan_topic: Optional laser-scan topic, read by :meth:`get_scan`.
        host: rosbridge server hostname or IP.
        port: rosbridge WebSocket port.
        cmd_vel_type: Interface type of ``cmd_vel_topic`` (ROS1 two-segment).
        odom_type: Interface type of ``odom_topic``; rosapi-resolved when omitted.
        scan_type: Interface type of ``scan_topic``; rosapi-resolved when omitted.
        max_linear: Linear-velocity clamp (m/s).
        max_angular: Angular-velocity clamp (rad/s).
        max_duration: Longest accepted :meth:`drive` hold; longer requests are
            rejected loudly rather than silently truncated.
        publish_rate: Command publish rate (Hz) for held :meth:`drive` calls.

    Raises:
        ValueError: When a graph name, the host or the port is malformed, or
            when any of ``max_linear``, ``max_angular``, ``max_duration`` and
            ``publish_rate`` is not a finite number greater than zero. Each
            bounds every later command, so an unusable one is refused at
            construction instead of silently reshaping the robot's limits.
    """

    def __init__(
        self,
        node_name: str,
        cmd_vel_topic: str,
        odom_topic: str,
        scan_topic: str | None = None,
        *,
        host: str = "localhost",
        port: int = 9090,
        cmd_vel_type: str = _TWIST_TYPE,
        odom_type: str | None = None,
        scan_type: str | None = None,
        max_linear: float = 2.0,
        max_angular: float = 1.0,
        max_duration: float = 30.0,
        publish_rate: float = 10.0,
    ) -> None:
        self.node_name = _check_topic("node_name", node_name)
        self.cmd_vel_topic = _check_topic("cmd_vel_topic", cmd_vel_topic)
        self.odom_topic = _check_topic("odom_topic", odom_topic)
        self.scan_topic = _check_topic("scan_topic", scan_topic) if scan_topic else None
        # Two stages, the same pair the port below has: the shared domain every
        # dialled host in this package shares, then the transport's own narrower
        # allowlist. The allowlist is a pattern, so offering it the caller's
        # value directly raised ``TypeError`` for a non-string host, out of a
        # constructor whose contract is to report a malformed host as
        # ``ValueError``.
        if (host_error := dial_host_error(host, "host", type(self).__name__)) is not None:
            raise ValueError(host_error)
        if not _HOST_RE.match(host):
            raise ValueError(f"invalid host: {host!r}")
        if (port_error := tcp_port_error(port, "port", type(self).__name__)) is not None:
            raise ValueError(port_error)
        # Refused at construction rather than at first use: this bridge forwards
        # every call through use_rosbridge, so a port the transport cannot carry
        # is a dead bridge, and the point the port is named is the only place a
        # caller can act on that.
        if (transport_error := _transport_port_error(port, "port", type(self).__name__)) is not None:
            raise ValueError(transport_error)
        self.host = host
        self.port = port
        self.cmd_vel_type = cmd_vel_type
        self.odom_type = odom_type
        self.scan_type = scan_type
        # These four values bound every command this bridge will ever send, so
        # one that cannot be honored is refused here rather than coerced. The
        # domain is the shared one the sibling bridges use, so the three cannot
        # diverge on what counts as a usable limit: a non-real value (a numeric
        # string, ``None``, a list) is reported as this documented
        # ``ValueError`` rather than escaping as a coercion error, and ``bool``
        # is rejected because ``True`` would otherwise install a silent 1.0 m/s
        # clamp or a 1 Hz publish rate.
        for label, value in (
            ("max_linear", max_linear),
            ("max_angular", max_angular),
            ("max_duration", max_duration),
            ("publish_rate", publish_rate),
        ):
            if limit_err := positive_finite_number_error(value, label, type(self).__name__):
                raise ValueError(limit_err)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.max_duration = float(max_duration)
        self.publish_rate = float(publish_rate)

    @classmethod
    def from_curiosity(
        cls,
        node_name: str = "curiosity",
        host: str = "localhost",
        port: int = 9090,
        **overrides: Any,
    ) -> RosbridgeRobot:
        """Wiring for the NASA Curiosity rover Gazebo simulation (ROS1 Noetic).

        The rover's ``ackermann_drive_controller`` consumes ``geometry_msgs/Twist``
        directly, so no client-side kinematic model is needed. Limits ported
        from the strands-robots-ros2 registry entry that first drove this sim.
        """
        wiring: dict[str, Any] = {
            "cmd_vel_topic": "/curiosity_mars_rover/ackermann_drive_controller/cmd_vel",
            "odom_topic": "/curiosity_mars_rover/odom",
            "max_linear": 2.0,
            "max_angular": 1.0,
            "max_duration": 30.0,
            "publish_rate": 10.0,
        }
        wiring.update(overrides)
        cmd_vel_topic = wiring.pop("cmd_vel_topic")
        odom_topic = wiring.pop("odom_topic")
        scan_topic = wiring.pop("scan_topic", None)
        return cls(node_name, cmd_vel_topic, odom_topic, scan_topic, host=host, port=port, **wiring)

    @staticmethod
    def _error(text: str) -> dict[str, Any]:
        return {"status": "error", "content": [{"text": text}]}

    def _publish_twist(
        self,
        linear: float,
        angular: float,
        count: int,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Publish ``count`` Twist messages, carrying the operator context.

        ``use_rosbridge`` gates a publish aimed at a safety-critical command
        surface, and ``cmd_vel`` is one, so the context has to reach it: without
        one the gate has nothing to ask an operator with and fails closed on
        every command this robot sends.

        Args:
            linear: Linear velocity for ``linear.x``.
            angular: Angular velocity for ``angular.z``.
            count: Number of messages to publish.
            tool_context: Operator context forwarded to the transport.

        Returns:
            The ``use_rosbridge`` publish result dict.
        """
        return use_rosbridge(
            action="publish",
            host=self.host,
            port=self.port,
            topic=self.cmd_vel_topic,
            type=self.cmd_vel_type,
            fields={"linear": {"x": float(linear)}, "angular": {"z": float(angular)}},
            count=count,
            rate=self.publish_rate,
            tool_context=tool_context,
        )

    def drive(
        self,
        linear: float = 0.0,
        angular: float = 0.0,
        duration: float | None = None,
        count: int = 1,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Publish a velocity command over rosbridge.

        Fleet-standard across all three mobile-base bridges: inputs are
        validated against the shared numeric domains before any side effect, a
        bare single-shot command latches until :meth:`stop`, like any raw
        cmd_vel publish, and every timed or multi-message non-zero command is
        followed by a single zero Twist - even if the main publish failed - so a
        timed drive does not leave the robot with a live velocity. That zero is a
        gated command in its own right, so when it is the call that fails the
        result says so rather than reporting the hold's success. The trailing
        zero was this bridge's alone until the shared mobile base took over the
        drive contract; the other two inherit it now, so a timed drive
        self-stops wherever it is issued.

        Not carried by every mobile base: velocities are clamped to
        ``max_linear`` and ``max_angular``, and a hold beyond ``max_duration``
        is refused.
        :class:`~strands_robots.mesh.ackermann_robot.AckermannRosRobot`
        declares both as well, as ``max_speed`` and a ``max_duration`` of its
        own, because it too wraps a platform whose limits are known - so a hold
        this bridge accepts can be refused on that car, and the reverse.
        :meth:`RosBridgedRobot.drive` and :meth:`RtpsRobot.drive` carry
        neither: neither knows the ceilings of the third-party robot it drives,
        so they declare no velocity or duration limit and publish the requested
        burst unclamped. An unset limit there means "this platform declares no
        limit", never zero.

        Args:
            linear: Forward linear velocity (m/s), mapped to ``linear.x``. Must
                be a finite number; both signs are valid (negative reverses).
            angular: Yaw angular velocity (rad/s), mapped to ``angular.z``. Must
                be a finite number; both signs are valid (negative turns the
                other way).
            duration: When given, hold the command for this many seconds by
                publishing ``round(duration * publish_rate)`` messages (at
                least one). Takes precedence over ``count``, must be > 0 and
                finite - a zero or negative hold has no message count that
                expresses it, and publishing a single velocity command anyway
                would start the robot moving - and may not exceed
                ``max_duration``.
            count: Number of messages to publish when ``duration`` is omitted.
                Must be a positive whole number; ``0`` or a negative count
                publishes nothing, so reporting a successful drive for it hides
                a command that never left the process.
            tool_context: Operator context forwarded to ``use_rosbridge``, whose
                gate prompts before a publish to a safety-critical command
                surface. Without it the gate fails closed, so a command this
                bridge could otherwise have carried is refused with no operator
                ever asked.

        Returns:
            The ``use_rosbridge`` publish result dict, or an
            ``{"status": "error"}`` result naming the parameter when a value
            cannot be honored - in which case nothing is published.
        """
        # A velocity command is the one call on this bridge that physically
        # moves the robot, so every knob it carries is checked before anything
        # reaches the wire, through the same shared domains the sibling bridges
        # use. ``use_rosbridge`` validates the topic and interface type but
        # never sees ``duration`` at all (it receives only the derived message
        # count), and the ``count`` values it does refuse are reported against
        # a transport this caller never invoked.
        cmd_err = (
            finite_number_error(linear, "linear", "drive")
            or finite_number_error(angular, "angular", "drive")
            or (
                positive_finite_number_error(duration, "duration", "drive")
                if duration is not None
                # ``duration`` supersedes ``count``, so ``count`` is only the
                # effective horizon when no duration was given; refusing a
                # ``count`` this call never reads would reject a valid command.
                else positive_whole_number_error(count, "count", "drive")
            )
        )
        if cmd_err:
            return self._error(cmd_err)
        # Ordered after the finiteness guard on purpose: every comparison
        # against ``nan`` is false, so a ceiling test cannot stand in for one.
        if duration is not None and duration > self.max_duration:
            return self._error(
                f"drive: duration {duration}s exceeds max_duration {self.max_duration}s "
                "- issue shorter commands instead of one long hold"
            )
        v = max(-self.max_linear, min(self.max_linear, float(linear)))
        w = max(-self.max_angular, min(self.max_angular, float(angular)))
        # ``duration`` is positive and finite here, so this floor is a rounding
        # rule and not a substitute for validation: a hold shorter than one
        # publish period still means "send the command once".
        n = max(1, round(duration * self.publish_rate)) if duration is not None else count
        # The trailing stop goes out from ``finally`` even when the main publish
        # raised, and its verdict is kept rather than dropped - see
        # :func:`~strands_robots.mesh._mobile_base.failed_halt_error`.
        halt: dict[str, Any] | None = None
        try:
            result = self._publish_twist(v, w, count=n, tool_context=tool_context)
        finally:
            # The trailing zero carries the same context as the command it
            # undoes: one that could not reach the gate would be refused on its
            # own and leave the robot latched at the speed of an approved hold.
            if (duration is not None or n > 1) and (v or w):
                halt = self._publish_twist(0.0, 0.0, count=1, tool_context=tool_context)
        latched = failed_halt_error(result, halt, topic=self.cmd_vel_topic, subject=LATCHED_VELOCITY)
        return self._error(latched) if latched else result

    def stop(self, tool_context: ToolContext | None = None) -> dict[str, Any]:
        """Publish a single zero Twist.

        Never gated on this bridge's own state: a halt does not depend on a
        prior command having succeeded, and there is no enable handshake to
        satisfy. It is not exempt from the transport tool's command gate, which
        is keyed on the surface rather than the payload - zero means
        "stationary" on a ``Twist`` but commands motion to the zero pose on a
        joint-command topic, so a payload-shaped carve-out could not be written
        correctly. The halt stays reachable through the same approval path as
        any other command instead, which is why it forwards the context.

        Args:
            tool_context: Operator context forwarded to ``use_rosbridge``.

        Returns:
            The ``use_rosbridge`` publish result dict.
        """
        return self._publish_twist(0.0, 0.0, count=1, tool_context=tool_context)

    def get_pose(self, timeout: float = 5.0) -> dict[str, Any]:
        """Read one odometry/pose sample from ``odom_topic``."""
        return use_rosbridge(
            action="echo",
            host=self.host,
            port=self.port,
            topic=self.odom_topic,
            type=self.odom_type,
            count=1,
            timeout=timeout,
        )

    def get_scan(self, timeout: float = 5.0) -> dict[str, Any]:
        """Read one laser-scan sample (error when no ``scan_topic`` configured)."""
        if not self.scan_topic:
            return self._error("get_scan: no scan_topic configured for this robot")
        return use_rosbridge(
            action="echo",
            host=self.host,
            port=self.port,
            topic=self.scan_topic,
            type=self.scan_type,
            count=1,
            timeout=timeout,
        )

    @property
    def tools(self) -> list[AgentTool]:
        """This robot's capabilities as named strands agent tools.

        The two tools that carry a command are declared ``@tool(context=True)``
        and forward the injected context to :meth:`drive` and :meth:`stop`, so a
        publish to a gated ``cmd_vel`` prompts the operator instead of failing
        closed on every call. The read-only tools take no context because a read
        is never gated.
        """
        suffix = self.node_name.strip("/").replace("/", "_")

        @tool(
            name=f"drive_{suffix}",
            description=(
                f"Drive the {self.node_name} robot over rosbridge (linear m/s up to "
                f"{self.max_linear}, angular rad/s up to {self.max_angular}, optional "
                "duration s). A command with duration stops automatically afterwards; "
                "without duration the last command latches until stop."
            ),
            context=True,
        )
        def drive(
            linear: float = 0.0,
            angular: float = 0.0,
            duration: float | None = None,
            tool_context: ToolContext | None = None,
        ) -> dict[str, Any]:
            return self.drive(linear=linear, angular=angular, duration=duration, tool_context=tool_context)

        @tool(
            name=f"stop_{suffix}",
            description=f"Immediately stop the {self.node_name} robot.",
            context=True,
        )
        def stop(tool_context: ToolContext | None = None) -> dict[str, Any]:
            return self.stop(tool_context=tool_context)

        @tool(name=f"get_pose_{suffix}", description=f"Read the current odometry/pose of the {self.node_name} robot.")
        def get_pose() -> dict[str, Any]:
            return self.get_pose()

        @tool(name=f"get_scan_{suffix}", description=f"Read one laser scan from the {self.node_name} robot.")
        def get_scan() -> dict[str, Any]:
            return self.get_scan()

        agent_tools: list[AgentTool] = [drive, stop, get_pose]
        if self.scan_topic:
            agent_tools.append(get_scan)
        return agent_tools

    def __repr__(self) -> str:
        try:
            return (
                f"RosbridgeRobot(node_name={self.node_name!r}, ws://{self.host}:{self.port}, "
                f"cmd_vel_topic={self.cmd_vel_topic!r}, odom_topic={self.odom_topic!r})"
            )
        except AttributeError:
            return partial_construction_repr(self)
