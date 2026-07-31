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

import math
from typing import Any

from strands import tool
from strands.types.tools import AgentTool

from strands_robots.mesh.ros_bridge import _check_topic
from strands_robots.tools.use_rosbridge import _HOST_RE, use_rosbridge

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
        if not host or not _HOST_RE.match(host):
            raise ValueError(f"invalid host: {host!r}")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError(f"invalid port: {port!r} (expected 1-65535)")
        self.host = host
        self.port = port
        self.cmd_vel_type = cmd_vel_type
        self.odom_type = odom_type
        self.scan_type = scan_type
        for label, value in (
            ("max_linear", max_linear),
            ("max_angular", max_angular),
            ("max_duration", max_duration),
            ("publish_rate", publish_rate),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be a positive finite number, got {value!r}")
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

    def _publish_twist(self, linear: float, angular: float, count: int) -> dict[str, Any]:
        return use_rosbridge(
            action="publish",
            host=self.host,
            port=self.port,
            topic=self.cmd_vel_topic,
            type=self.cmd_vel_type,
            fields={"linear": {"x": float(linear)}, "angular": {"z": float(angular)}},
            count=count,
            rate=self.publish_rate,
        )

    def drive(
        self,
        linear: float = 0.0,
        angular: float = 0.0,
        duration: float | None = None,
        count: int = 1,
    ) -> dict[str, Any]:
        """Publish a velocity command over rosbridge.

        Fleet-standard contract: ``linear`` (m/s) and ``angular`` (rad/s),
        optional ``duration`` hold (``round(duration * publish_rate)``
        messages, precedence over ``count``). Inputs are validated before any
        side effect; velocities are clamped to the constructor limits. Every
        timed or multi-message non-zero command is followed by a single zero
        Twist - even if the main publish failed - so a timed drive cannot
        leave the robot with a live velocity. A bare single-shot command
        latches until :meth:`stop`, like any raw cmd_vel publish.
        """
        for label, value in (("linear", linear), ("angular", angular)):
            if not math.isfinite(value):
                return self._error(f"drive: {label} must be a finite number, got {value!r}")
        if duration is not None:
            if not math.isfinite(duration) or duration <= 0:
                return self._error(f"drive: duration must be a positive finite number of seconds, got {duration!r}")
            if duration > self.max_duration:
                return self._error(
                    f"drive: duration {duration}s exceeds max_duration {self.max_duration}s "
                    "- issue shorter commands instead of one long hold"
                )
        v = max(-self.max_linear, min(self.max_linear, float(linear)))
        w = max(-self.max_angular, min(self.max_angular, float(angular)))
        n = max(1, round(duration * self.publish_rate)) if duration is not None else count
        try:
            return self._publish_twist(v, w, count=n)
        finally:
            if (duration is not None or n > 1) and (v or w):
                self._publish_twist(0.0, 0.0, count=1)

    def stop(self) -> dict[str, Any]:
        """Publish a single zero Twist. Never gated on anything."""
        return self._publish_twist(0.0, 0.0, count=1)

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
        """This robot's capabilities as named strands agent tools."""
        suffix = self.node_name.strip("/").replace("/", "_")

        @tool(
            name=f"drive_{suffix}",
            description=(
                f"Drive the {self.node_name} robot over rosbridge (linear m/s up to "
                f"{self.max_linear}, angular rad/s up to {self.max_angular}, optional "
                "duration s). A command with duration stops automatically afterwards; "
                "without duration the last command latches until stop."
            ),
        )
        def drive(linear: float = 0.0, angular: float = 0.0, duration: float | None = None) -> dict[str, Any]:
            return self.drive(linear=linear, angular=angular, duration=duration)

        @tool(name=f"stop_{suffix}", description=f"Immediately stop the {self.node_name} robot.")
        def stop() -> dict[str, Any]:
            return self.stop()

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
        return (
            f"RosbridgeRobot(node_name={self.node_name!r}, ws://{self.host}:{self.port}, "
            f"cmd_vel_topic={self.cmd_vel_topic!r}, odom_topic={self.odom_topic!r})"
        )
