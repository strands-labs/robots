#!/usr/bin/env python3
"""Pure-RTPS ROS 2 tool - join a ROS 2 graph as a DDS participant, no rclpy.

Where ``use_ros`` is a *client* that needs a sourced ROS 2 distro (rclpy),
``use_rtps`` is a *participant* built on the pip-installable ``cyclonedds``
binding alone. It speaks RTPS - the DDS wire protocol every ROS 2 distro uses -
so it interoperates with Humble, Jazzy, Rolling, ... uniformly, with nothing
installed but a pip wheel - on macOS, Windows and Linux x86_64. Linux aarch64
publishes no cyclonedds wheel and builds the binding against a Cyclone DDS C
install instead (``docs/rtps-integration.md#linux-aarch64-jetson``).

The headline capability: an RTPS participant can **act as a robot**. It can
advertise and publish a topic that a real ROS 2 node (rviz, nav2, a teleop
joystick) will consume, and subscribe to command topics - indistinguishable on
the wire from physical hardware.

This module is the agent-facing envelope: the numeric-option domains an agent
can get wrong, the operator gate, and the tool docstring a model reads. The
participant itself - the shared ``DomainParticipant``, the cached writers and
readers, the IDL sample builder - is
:mod:`strands_robots.rtps.participant`, which
:class:`~strands_robots.mesh.rtps_robot.RtpsRobot` publishes through as well.

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

from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from strands_robots._command_gate import gate_command
from strands_robots.rtps.participant import GATE_TOOL, _err, never_gated, rtps_action
from strands_robots.tools._numeric_options import numeric_option_error

# Every verb this tool answers. Graded first, so a misspelled action is named
# before the participant's backend probe answers it with an install recipe.
_ACTIONS: tuple[str, ...] = ("status", "types", "advertise", "publish", "subscribe", "echo")

# Which numeric options each action actually consumes. ``status``, ``types``,
# ``advertise`` and ``subscribe`` read none of them, so the guard below is driven
# by this table rather than validating the whole signature unconditionally - a
# caller is never refused for a value the requested action never looks at.
_ACTION_NUMERIC_OPTIONS: dict[str, tuple[str, ...]] = {
    "publish": ("count", "rate"),
    "echo": ("timeout", "count"),
}


@tool(context=True)
def use_rtps(
    action: str,
    topic: str | None = None,
    type: str | None = None,
    fields: dict[str, Any] | None = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
    tool_context: ToolContext | None = None,
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
        tool_context: Injected agent context, used to ask an operator before a
            ``publish`` reaches a safety-critical command surface.

    Returns:
        A Strands tool result dict ``{"status": ..., "content": [{"text": ...}]}``.
    """
    if action not in _ACTIONS:
        return _err(f"unknown action: {action!r}. Valid: {', '.join(_ACTIONS)}")

    # Numeric options are checked here, ahead of the participant's backend probe,
    # so the same caller mistake is reported identically whether or not
    # cyclonedds is installed - and so a refusal happens before a writer joins
    # the graph. They are the agent-supplied half of the call, which is why the
    # table above lives beside this tool rather than in the participant.
    numeric_error = numeric_option_error(action, _ACTION_NUMERIC_OPTIONS, timeout=timeout, count=count, rate=rate)
    if numeric_error:
        return _err(numeric_error)

    # Each verb forwards the options it reads and no others, and only ``publish``
    # is handed a gate that can reach an operator: the read paths cannot prompt
    # at all, rather than being trusted not to.
    if action == "publish":
        return rtps_action(
            action=action,
            topic=topic,
            type=type,
            fields=fields,
            count=count,
            rate=rate,
            gate=lambda target: gate_command("publish", target, tool_context=tool_context, tool=GATE_TOOL),
        )
    if action in ("subscribe", "echo"):
        return rtps_action(action=action, topic=topic, type=type, count=count, timeout=timeout, gate=never_gated)
    if action == "advertise":
        return rtps_action(action=action, topic=topic, type=type, gate=never_gated)
    return rtps_action(action=action, gate=never_gated)


__all__ = ["use_rtps"]
