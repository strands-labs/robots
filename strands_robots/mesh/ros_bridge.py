"""ROS 2 mesh bridge - present a remote ROS 2 robot as a strands Robot.

A :class:`RosBridgedRobot` wraps a ROS 2 mobile base (or any robot exposing a
``cmd_vel`` / odometry / scan topic trio) so that an agent can drive it and read
its state with the same ``Agent(tools=[robot])`` pattern used for simulated and
hardware robots. All ROS 2 I/O is forwarded through the
:func:`strands_robots.tools.use_ros.use_ros` tool, so the bridge stays thin and
inherits ``use_ros``'s in-process ``rclpy`` backend and its topic/type
validation. The parameters ``use_ros`` never sees are validated here: a
:meth:`RosBridgedRobot.drive` command whose velocity, hold duration or message
count cannot be honored is refused without publishing anything, and a
:meth:`RosBridgedRobot.navigate_to` goal whose pose cannot be honored is refused
without sending anything - the goal coordinates travel inside the request body,
which ``use_ros`` forwards verbatim.

Commanding a robot goes through ``use_ros``'s operator-approval gate, because
``/turtle1/cmd_vel`` and a Nav2 ``/navigate_to_pose`` are safety-critical
surfaces. The bridge therefore forwards an operator context to it: the
``drive_<node>`` / ``stop_<node>`` / ``navigate_<node>`` agent tools are declared
``@tool(context=True)`` and hand the context to :func:`use_ros`, so an agent
driving this robot prompts the operator exactly as a direct ``use_ros`` call
does. The read paths (:meth:`get_pose`, :meth:`get_scan`) are never gated.

A **programmatic** call carries no operator context, so it is refused unless the
surface is pre-approved with ``STRANDS_ROS2_COMMAND_ALLOW`` (or the gate is
bypassed with ``BYPASS_TOOL_CONSENT=true``). That includes :meth:`stop`: the gate
is keyed on the *surface*, not on the payload, so a zero-velocity halt is gated
like any other publish to ``cmd_vel``. A payload-shaped exemption would have to
know that zero means "stationary" on a ``Twist`` while zero on ``/joint_command``
commands motion to the zero pose - so the halt stays reachable through the same
three approval paths rather than through a carve-out.

Typical usage::

    import os

    from strands import Agent
    from strands_robots.mesh import RosBridgedRobot

    turtle = RosBridgedRobot.from_ros(
        node_name="turtlesim",
        cmd_vel_topic="/turtle1/cmd_vel",
        odom_topic="/turtle1/pose",
        odom_type="turtlesim/msg/Pose",
    )

    # Direct, programmatic control - no operator context to prompt, so the
    # command surface has to be pre-approved:
    os.environ["STRANDS_ROS2_COMMAND_ALLOW"] = "/turtle1/cmd_vel"
    turtle.drive(linear=1.0)
    print(turtle.get_pose())  # reads are never gated

    # Or hand the bridge to an agent as first-class tools - the command tools
    # carry the operator context, so the gate prompts instead of refusing:
    agent = Agent(tools=turtle.tools)
    agent("drive forward, then tell me the pose")
"""

from __future__ import annotations

import math
import re
from typing import Any

from strands import tool
from strands.types.tools import AgentTool, ToolContext

from strands_robots.tools.use_ros import use_ros
from strands_robots.utils import (
    finite_number_error,
    partial_construction_repr,
    positive_finite_number_error,
    positive_whole_number_error,
)

_TWIST_TYPE = "geometry_msgs/msg/Twist"
_NAV_ACTION_TYPE = "nav2_msgs/action/NavigateToPose"

# ROS 2 graph names: leading slash plus alnum / _ / ~ segments. Reject anything
# else early so a malformed topic fails at construction with a clear message
# rather than deep inside a forwarded ``use_ros`` call.
_TOPIC_RE = re.compile(r"^[A-Za-z0-9_/~]+$")


def _check_topic(label: str, value: str) -> str:
    """Validate a ROS 2 topic/node name, returning it unchanged when valid."""
    if not value or not _TOPIC_RE.match(value):
        raise ValueError(f"invalid {label}: {value!r} (expected a ROS 2 graph name like /turtle1/cmd_vel)")
    return value


class RosBridgedRobot:
    """A remote ROS 2 robot exposed as a strands-controllable robot.

    The bridge owns no ROS 2 state of its own; every method forwards to
    :func:`use_ros`. It is therefore safe to construct without a ROS 2
    environment present - errors surface only when a method is actually called
    and no backend is available.

    Args:
        node_name: Human-readable identifier for the remote robot. Used only to
            name this bridge's agent tools (``drive_<node_name>`` etc.); it does
            not need to match the ROS 2 node name.
        cmd_vel_topic: Velocity-command topic the robot subscribes to (e.g.
            ``/turtle1/cmd_vel`` or ``/cmd_vel``).
        odom_topic: Topic carrying the robot's pose/odometry (e.g.
            ``/turtle1/pose`` or ``/odom``). Read by :meth:`get_pose`.
        scan_topic: Optional laser-scan topic (e.g. ``/scan``). Read by
            :meth:`get_scan`; when omitted, :meth:`get_scan` returns an error.
        cmd_vel_type: Interface type published to ``cmd_vel_topic``. Defaults to
            ``geometry_msgs/msg/Twist``.
        odom_type: Interface type of ``odom_topic``. Optional - when omitted,
            ``use_ros`` resolves it from the live graph.
        scan_type: Interface type of ``scan_topic``. Optional - resolved from
            the live graph when omitted.
        publish_rate: Default rate (Hz) for multi-message :meth:`drive` calls.
            Must be > 0 and finite: :meth:`drive` multiplies it by ``duration``
            to size the message burst and ``use_ros`` publishes at ``1 / rate``,
            so a non-positive rate removes the pacing entirely rather than
            slowing it. Raises ``ValueError`` at construction otherwise.
        nav_action: Optional Nav2-style action server name (e.g.
            ``/navigate_to_pose``). When set, :meth:`navigate_to` sends
            goal-level navigation instead of raw velocity, and a
            ``navigate_<node_name>`` agent tool is exposed.
        nav_action_type: Action interface for ``nav_action``. Defaults to
            ``nav2_msgs/action/NavigateToPose``.
    """

    def __init__(
        self,
        node_name: str,
        cmd_vel_topic: str,
        odom_topic: str,
        scan_topic: str | None = None,
        *,
        cmd_vel_type: str = _TWIST_TYPE,
        odom_type: str | None = None,
        scan_type: str | None = None,
        publish_rate: float = 10.0,
        nav_action: str | None = None,
        nav_action_type: str = _NAV_ACTION_TYPE,
    ) -> None:
        self.node_name = _check_topic("node_name", node_name)
        self.cmd_vel_topic = _check_topic("cmd_vel_topic", cmd_vel_topic)
        self.odom_topic = _check_topic("odom_topic", odom_topic)
        self.scan_topic = _check_topic("scan_topic", scan_topic) if scan_topic else None
        self.cmd_vel_type = cmd_vel_type
        self.odom_type = odom_type
        self.scan_type = scan_type
        if rate_err := positive_finite_number_error(publish_rate, "publish_rate", type(self).__name__):
            raise ValueError(rate_err)
        self.publish_rate = float(publish_rate)
        self.nav_action = _check_topic("nav_action", nav_action) if nav_action else None
        self.nav_action_type = nav_action_type

    @classmethod
    def from_ros(
        cls,
        node_name: str,
        cmd_vel_topic: str,
        odom_topic: str,
        scan_topic: str | None = None,
        **kwargs: Any,
    ) -> RosBridgedRobot:
        """Construct a bridge from ROS 2 topic wiring.

        Convenience alternate constructor mirroring the keyword style used
        elsewhere in the library. Equivalent to calling the constructor
        directly; provided so call sites read as ``RosBridgedRobot.from_ros(
        node_name=..., cmd_vel_topic=...)``.
        """
        return cls(node_name, cmd_vel_topic, odom_topic, scan_topic, **kwargs)

    def drive(
        self,
        linear: float = 0.0,
        angular: float = 0.0,
        duration: float | None = None,
        count: int = 1,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Publish a velocity command to the robot's ``cmd_vel`` topic.

        Args:
            linear: Forward linear velocity (m/s), mapped to ``linear.x``. Must
                be a finite number; both signs are valid (negative reverses).
            angular: Yaw angular velocity (rad/s), mapped to ``angular.z``. Must
                be a finite number; both signs are valid (negative turns the
                other way).
            duration: When given, hold the command for this many seconds by
                publishing ``round(duration * publish_rate)`` messages (at least
                one). Takes precedence over ``count``, and must be > 0 and
                finite - a zero or negative hold has no message count that
                expresses it, and publishing a single velocity command anyway
                would start the robot moving.
            count: Number of messages to publish when ``duration`` is omitted.
                Must be a positive whole number; ``0`` or a negative count
                publishes nothing, so reporting a successful drive for it hides
                a command that never left the process.
            tool_context: Operator context forwarded to ``use_ros``, whose
                command gate prompts for approval on a safety-critical surface
                such as ``cmd_vel``. The ``drive_<node_name>`` agent tool passes
                the one the framework injects; a programmatic call has none, and
                the gate then refuses unless the surface is pre-approved via
                ``STRANDS_ROS2_COMMAND_ALLOW`` / ``BYPASS_TOOL_CONSENT``.

        Returns:
            The ``use_ros`` publish result dict, or an ``{"status": "error"}``
            result naming the parameter when a value cannot be honored - in
            which case nothing is published.
        """
        # A velocity command is the one call on this bridge that physically
        # moves the robot, so every knob it carries is checked before anything
        # reaches the wire. ``use_ros`` validates the topic and interface type
        # but never sees ``duration`` at all (it receives only the derived
        # message count), so the horizon knobs have no guard downstream.
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
            return {"status": "error", "content": [{"text": cmd_err}]}
        # ``duration`` is positive and finite here, so this floor is a rounding
        # rule and not a substitute for validation: a hold shorter than one
        # publish period still means "send the command once".
        n = max(1, round(duration * self.publish_rate)) if duration is not None else count
        fields = {"linear": {"x": float(linear)}, "angular": {"z": float(angular)}}
        return use_ros(
            action="publish",
            topic=self.cmd_vel_topic,
            type=self.cmd_vel_type,
            fields=fields,
            count=n,
            rate=self.publish_rate,
            tool_context=tool_context,
        )

    def stop(self, tool_context: ToolContext | None = None) -> dict[str, Any]:
        """Publish a zero-velocity command to halt the robot.

        The halt reaches ``cmd_vel`` through the same publish as :meth:`drive`,
        so it passes the same operator gate: the ``stop_<node_name>`` agent tool
        carries the operator context and prompts, while a programmatic
        ``stop()`` needs the surface pre-approved. The gate is keyed on the
        surface rather than the payload deliberately - a "zero means harmless"
        exemption holds for a ``Twist`` and not for ``/joint_command``, where
        zero commands motion to the zero pose.

        Args:
            tool_context: Operator context forwarded to ``use_ros`` (see
                :meth:`drive`).
        """
        return self.drive(linear=0.0, angular=0.0, count=1, tool_context=tool_context)

    def get_pose(self, timeout: float = 5.0) -> dict[str, Any]:
        """Read one sample from the robot's odometry/pose topic.

        Returns:
            The ``use_ros`` echo result dict for ``odom_topic``.
        """
        return use_ros(
            action="echo",
            topic=self.odom_topic,
            type=self.odom_type,
            count=1,
            timeout=timeout,
        )

    def get_scan(self, timeout: float = 5.0) -> dict[str, Any]:
        """Read one sample from the robot's laser-scan topic.

        Returns:
            The ``use_ros`` echo result dict for ``scan_topic``, or an error
            result when no ``scan_topic`` was configured.
        """
        if not self.scan_topic:
            return {
                "status": "error",
                "content": [{"text": "get_scan: no scan_topic configured for this robot"}],
            }
        return use_ros(
            action="echo",
            topic=self.scan_topic,
            type=self.scan_type,
            count=1,
            timeout=timeout,
        )

    def navigate_to(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        frame_id: str = "map",
        timeout: float = 120.0,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Send a goal-level navigation request to the robot's ``nav_action``.

        Unlike :meth:`drive`, which streams raw velocity, this delegates
        obstacle avoidance, path planning, and recovery to the robot's own
        navigation stack (Nav2 by default) and blocks until the goal reaches a
        terminal state or ``timeout`` expires - at which point ``use_ros``
        cancels the goal so the robot does not keep navigating unattended.

        Args:
            x: Goal position x in ``frame_id`` (meters). Must be a finite
                number; both signs are valid.
            y: Goal position y in ``frame_id`` (meters). Must be a finite
                number; both signs are valid.
            yaw: Goal heading in radians, encoded as a planar quaternion. Must
                be a finite number; both signs are valid (negative turns the
                other way).
            frame_id: Frame the goal pose is expressed in (default ``map``).
            timeout: End-to-end budget in seconds for the navigation goal.
                Forwarded to ``use_ros``, which refuses a non-positive or
                non-finite budget.
            tool_context: Operator context forwarded to ``use_ros``, whose
                command gate covers a Nav2-style ``/navigate_to_pose`` action
                goal as well as a ``cmd_vel`` publish (see :meth:`drive`).

        Returns:
            The ``use_ros`` action result dict (goal status, result, feedback
            samples), or an ``{"status": "error"}`` result when no
            ``nav_action`` was configured or when a pose component cannot be
            honored - in which case no goal is sent.
        """
        if not self.nav_action:
            return {
                "status": "error",
                "content": [{"text": "navigate_to: no nav_action configured for this robot"}],
            }
        # The goal pose is the part of this call ``use_ros`` never validates: it
        # checks the action name and interface type, but the coordinates travel
        # inside ``fields`` and are serialized into the request verbatim. A
        # non-finite coordinate is a valid IEEE-754 float64 on the wire, so the
        # goal is accepted and handed to a planner that cannot resolve it, and
        # ``yaw`` additionally reaches ``math.sin``/``math.cos``, which raise a
        # bare ``ValueError`` for an infinite angle - out of a method whose
        # contract is a result dict, and out of the bound ``navigate_*`` tool.
        # ``timeout`` does reach ``use_ros`` and is guarded there.
        pose_err = (
            finite_number_error(x, "x", "navigate_to")
            or finite_number_error(y, "y", "navigate_to")
            or finite_number_error(yaw, "yaw", "navigate_to")
        )
        if pose_err:
            return {"status": "error", "content": [{"text": pose_err}]}
        half = 0.5 * float(yaw)
        fields = {
            "pose": {
                "header": {"frame_id": frame_id},
                "pose": {
                    "position": {"x": float(x), "y": float(y)},
                    "orientation": {"z": math.sin(half), "w": math.cos(half)},
                },
            }
        }
        return use_ros(
            action="action_send_goal",
            action_name=self.nav_action,
            type=self.nav_action_type,
            fields=fields,
            timeout=timeout,
            tool_context=tool_context,
        )

    @property
    def tools(self) -> list[AgentTool]:
        """Return this robot's capabilities as named strands agent tools.

        The returned tools are bound to this instance and uniquely named with
        the ``node_name`` suffix so multiple bridged robots can coexist in a
        single ``Agent(tools=[...])`` call without name collisions.

        ``drive`` and ``stop`` are always both present: a velocity command with
        no ``duration`` latches until another command arrives, so a caller that
        can start motion must be able to end it without knowing that a
        zero-velocity drive is the halt idiom.

        Every tool that carries a command (``drive``, ``stop``, ``navigate``) is
        declared ``@tool(context=True)`` and forwards the injected context into
        ``use_ros``, so its operator-approval gate prompts rather than failing
        closed. The read-only tools take no context because ``use_ros`` never
        gates ``echo``.
        """
        suffix = self.node_name.strip("/").replace("/", "_")

        @tool(
            name=f"drive_{suffix}",
            description=f"Drive the {self.node_name} robot (linear/angular velocity).",
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
            description=f"Immediately stop the {self.node_name} robot (zero velocity).",
            context=True,
        )
        def stop(tool_context: ToolContext | None = None) -> dict[str, Any]:
            return self.stop(tool_context=tool_context)

        @tool(name=f"get_pose_{suffix}", description=f"Read the current pose/odometry of the {self.node_name} robot.")
        def get_pose() -> dict[str, Any]:
            return self.get_pose()

        @tool(name=f"get_scan_{suffix}", description=f"Read one laser scan from the {self.node_name} robot.")
        def get_scan() -> dict[str, Any]:
            return self.get_scan()

        @tool(
            name=f"navigate_{suffix}",
            description=(
                f"Navigate the {self.node_name} robot to a map-frame (x, y, yaw) goal using its "
                "navigation stack (planning and obstacle avoidance handled on-robot)."
            ),
            context=True,
        )
        def navigate(
            x: float,
            y: float,
            yaw: float = 0.0,
            timeout: float = 120.0,
            tool_context: ToolContext | None = None,
        ) -> dict[str, Any]:
            return self.navigate_to(x=x, y=y, yaw=yaw, timeout=timeout, tool_context=tool_context)

        agent_tools: list[AgentTool] = [drive, stop, get_pose]
        if self.scan_topic:
            agent_tools.append(get_scan)
        if self.nav_action:
            agent_tools.append(navigate)
        return agent_tools

    def __repr__(self) -> str:
        try:
            return (
                f"RosBridgedRobot(node_name={self.node_name!r}, cmd_vel_topic={self.cmd_vel_topic!r}, "
                f"odom_topic={self.odom_topic!r}, scan_topic={self.scan_topic!r})"
            )
        except AttributeError:
            return partial_construction_repr(self)
