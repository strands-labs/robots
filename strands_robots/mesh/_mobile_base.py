"""Transport-agnostic mobile base - one drive contract, one safety contract.

Every mobile robot in :mod:`strands_robots.mesh` answers the same agent-facing
question ("drive at this velocity for this long") and differs only in *how* the
bytes reach the robot and *what shape* the command message has. Before this
module those two axes were entangled: each transport got its own class, and
each class re-implemented validation, clamping, the duration contract, the
trailing-zero stop, the ``tools`` property, and the ``from_<preset>()`` idiom.
The duplication was not incidental - the classes were written by mirroring each
other - and it meant a safety fix had to be found and applied N times.

:class:`MobileBaseRobot` owns the invariant half:

- input validation and name checking,
- the ``drive(linear, angular, duration)`` contract and its safety semantics,
- the pre-drive :attr:`init_services` handshake,
- ``stop`` / ``get_pose`` / ``get_scan``,
- the ``tools`` property, built from the capabilities actually wired.

Subclasses supply the variant half through two seams:

- a :class:`Transport` object - *how* to talk (``publish`` / ``echo``, and
  optionally ``service_call`` / ``action_send_goal``);
- :meth:`MobileBaseRobot._cmd_fields` - *what* the command message looks like,
  which is where kinematics lives (identity for a differential-drive Twist, a
  bicycle model for an Ackermann car).

Safety semantics, stated once because they are now implemented once:

- Non-finite inputs are refused. ``nan`` sails silently through a ``min``/``max``
  clamp and would otherwise be published as a velocity.
- ``duration`` must be a positive finite number, and within :attr:`max_duration`
  when one is set. An over-long hold is refused loudly rather than silently
  truncated, and is refused *before* any side effect - including before the
  ``init_services`` handshake, so an invalid request can never be what switches
  a vehicle into a commandable state.
- Velocities are clamped to the configured limits. A limit left at ``None``
  means "this platform declares no limit", not "zero".
- Every timed or multi-message non-zero command is followed by a single zero
  command through ``try``/``finally`` - *even when the main publish raised* - so
  a timed drive does not leave a robot with a live velocity latched. Sending it
  is not the same as landing it: that stop is a second command over the same
  transport, and the tool it goes through reports a refusal as an error envelope
  rather than by raising. Its verdict is therefore read, not dropped - a hold
  that succeeded over a stop that was refused is reported as the error, because
  the robot is still moving and a caller reading ``success`` never issues
  :meth:`MobileBaseRobot.stop`.
- A bare single-shot ``drive()`` latches until :meth:`stop`, matching a raw
  ``cmd_vel`` publish. This is disclosed in the agent-facing tool description
  rather than papered over.
- :meth:`stop` is never gated on the handshake and never on limits. An emergency
  stop must not require a working service graph. It is *not* exempt from the
  transport tool's own command gate, which is keyed on the surface rather than
  the payload - see :meth:`MobileBaseRobot.stop`.
- Every command carries an optional ``tool_context`` down to the transport,
  which is what hands it to the underlying tool. The gate belongs to the tool,
  so the base's job is only to not drop the operator's decision on the way - and
  because the command tools are declared here, that holds for every transport at
  once instead of once per bridge.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, cast, runtime_checkable

from strands import tool
from strands.types.tools import AgentTool, ToolContext

from strands_robots.utils import (
    finite_number_error,
    partial_construction_repr,
    positive_finite_number_error,
    positive_whole_number_error,
)


@runtime_checkable
class Transport(Protocol):
    """How a mobile base moves bytes to and from its robot.

    Only :meth:`publish` and :meth:`echo` are required. ``service_call`` and
    ``action_send_goal`` are *optional capabilities*: ``use_rtps`` has neither
    and the rosbridge transport has no actions, so requiring all four would
    force a transport to declare methods it cannot honor. The base detects what
    is present (:meth:`MobileBaseRobot.supports`) and reflects it in the tools
    it exposes and in what it accepts at construction.
    """

    #: Interface type for the platform's velocity command, e.g.
    #: ``geometry_msgs/msg/Twist`` (ROS 2) or ``geometry_msgs/Twist`` (ROS 1).
    twist_type: str

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
        """Send ``count`` messages of ``type`` to ``topic`` at ``rate`` Hz.

        Args:
            topic: Command topic to publish to.
            type: Interface type of the published message.
            fields: Message body.
            count: Number of messages to send.
            rate: Publish rate in Hz.
            tool_context: Operator context for the underlying tool's
                command-approval gate. The gate lives in the tool, so the base
                can only carry the context this far and the transport is what
                hands it over. A transport whose tool gates its command surface
                forwards it; one whose tool has no gate would have nothing to
                forward. Every shipped graph tool gates today, so no transport
                is currently in that second case - the condition is derived from
                the tool rather than restated here so a future ungated transport
                is handled without amending this contract.
        """
        ...

    def echo(self, *, topic: str, type: str | None, count: int, timeout: float) -> dict[str, Any]:
        """Read ``count`` samples from ``topic``, resolving ``type`` when ``None``."""
        ...


@runtime_checkable
class ServiceCapable(Protocol):
    """Optional capability: the transport can make a request/response call.

    Split out of :class:`Transport` rather than declared optional on it so that
    a transport without services (``use_rtps``) is not forced to define a method
    it cannot honor, and so a caller that needs services can narrow to this
    protocol instead of reaching through ``Any``.
    """

    def service_call(
        self,
        *,
        service: str,
        type: str,
        fields: dict[str, Any],
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Call ``service`` with ``fields`` and return its structured response.

        A service can command a robot - an arming call is the whole point of
        :attr:`MobileBaseRobot.init_services` - so it carries the operator
        context on the same terms as :meth:`Transport.publish`.
        """
        ...


@runtime_checkable
class ActionCapable(Protocol):
    """Optional capability: the transport can send a long-running action goal."""

    def action_send_goal(
        self,
        *,
        action_name: str,
        type: str,
        fields: dict[str, Any],
        timeout: float,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Send a goal to ``action_name`` and block until it settles or times out.

        Carries the operator context on the same terms as
        :meth:`Transport.publish`: a navigation goal moves the robot.
        """
        ...


def positive_finite(label: str, value: Any, context: str = "MobileBaseRobot") -> float:
    """Return ``value`` as a float, raising ``ValueError`` unless positive finite.

    Delegates the domain itself to :func:`~strands_robots.utils.positive_finite_number_error`,
    the rule every other continuous knob in the codebase is measured against, so
    a velocity clamp and a control-loop frequency cannot disagree about whether
    ``True`` or a NumPy scalar is a usable number. This wrapper only converts
    that shared verdict into the ``ValueError`` a constructor owes its caller.
    """
    if error := positive_finite_number_error(value, label, context):
        raise ValueError(error)
    return float(value)


#: What is still moving when a velocity command outlived the stop that ends it.
#: Shared by every ``Twist`` bridge so the two that own their own ``drive`` say
#: the same thing about the same failure.
LATCHED_VELOCITY = "the robot may still be moving at the commanded velocity"


def failed_halt_error(
    result: dict[str, Any],
    halt: dict[str, Any] | None,
    *,
    topic: str,
    subject: str,
) -> str | None:
    """Report a trailing stop that failed under a command that succeeded.

    A timed or repeated drive owns its own stop, and that stop is a second
    command over the same transport: the tool it goes through reports a refusal
    - a declined operator approval, a rate limit, a transport failure - as an
    error envelope rather than by raising. Returning the hold's success after
    that refusal presents a robot still moving at the commanded velocity as a
    drive that stopped itself, and a caller reading ``success`` never issues the
    ``stop`` that would end it.

    The hold's own failure wins when both failed: that is the cause, and a stop
    it never got to undo is a consequence of it. ``None`` for ``halt`` means no
    stop was owed - a single-shot command latches by contract - so there is no
    verdict to read and a queued failure from some later call cannot be
    misattributed to this one.

    Args:
        result: The hold's own publish envelope.
        halt: The trailing stop's envelope, or ``None`` when none was sent.
        topic: Command topic, named so the caller knows what is still live.
        subject: What is still moving, in the platform's own vocabulary.

    Returns:
        The error text, or ``None`` when there is nothing to report.
    """
    if halt is None or halt.get("status") == "success" or result.get("status") != "success":
        return None
    cause = " ".join(block.get("text", "") for block in halt.get("content", []) if isinstance(block, dict)).strip()
    return (
        f"drive: the command was published to {topic}, but the trailing stop "
        f"failed - {subject}. Halt it with stop. "
        f"Halt failure: {cause or 'no detail reported'}"
    )


class MobileBaseRobot:
    """A mobile robot exposed as a strands-controllable robot, transport-agnostic.

    The class owns no transport state; every method forwards through
    :attr:`transport`, so an instance is safe to construct with no ROS
    environment, no DDS stack and no reachable server - failures surface as
    structured results when a method actually runs.

    Args:
        node_name: Identifier used to name this robot's agent tools
            (``drive_<node_name>`` etc.). It need not match any node name on
            the robot's own graph.
        cmd_vel_topic: Topic the robot's velocity/servo commands are published
            to.
        transport: The :class:`Transport` carrying every call.
        odom_topic: Optional odometry/pose topic, read by :meth:`get_pose`.
            Optional because platforms differ: the stock AWS DeepRacer
            publishes no odometry at all, and a class that pretended otherwise
            would be reporting a pose it cannot know.
        scan_topic: Optional laser-scan topic, read by :meth:`get_scan`.
        cmd_vel_type: Interface type of ``cmd_vel_topic``. Defaults to the
            transport's ``twist_type``.
        odom_type: Interface type of ``odom_topic``; resolved from the live
            graph by the transport when omitted.
        scan_type: Interface type of ``scan_topic``; resolved from the live
            graph by the transport when omitted.
        max_linear: Linear-velocity clamp (m/s), or ``None`` for a platform
            that declares no limit.
        max_angular: Angular-velocity clamp (rad/s), or ``None``.
        max_duration: Longest accepted :meth:`drive` hold in seconds, or
            ``None``. A longer request is refused, never truncated.
        publish_rate: Command publish rate (Hz) for held :meth:`drive` calls.
        init_services: Ordered ``{"service", "type", "fields"}`` calls that put
            the robot into a commandable state (an arm/enable handshake). Run
            once, automatically, before the first :meth:`drive`. Requires a
            transport with ``service_call``.

    Raises:
        ValueError: on a malformed name, a non-positive-finite limit, a
            malformed ``init_services`` entry, or ``init_services`` on a
            transport that cannot call services.
    """

    #: Pattern a ``node_name`` must match. Overridable per platform.
    _NAME_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_/~]+\Z")
    #: Pattern a topic/service name must match. Overridable per platform.
    _TOPIC_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_/~]+\Z")
    #: Appended to ``node_name`` validation errors to say what a good value
    #: looks like. Overridable per platform alongside :attr:`_NAME_RE`, which
    #: it must describe.
    _NAME_HINT: str = " (expected a graph name like /turtle1/cmd_vel)"
    #: Appended to topic and service validation errors. Overridable per
    #: platform alongside :attr:`_TOPIC_RE`, which it must describe. Separate
    #: from :attr:`_NAME_HINT` because the two patterns are independently
    #: overridable: a platform whose node-name and topic grammars diverge needs
    #: two sentences, since neither one is true of both seams.
    _TOPIC_HINT: str = " (expected a graph name like /turtle1/cmd_vel)"

    def __init__(
        self,
        node_name: str,
        cmd_vel_topic: str,
        transport: Transport,
        *,
        odom_topic: str | None = None,
        scan_topic: str | None = None,
        cmd_vel_type: str | None = None,
        odom_type: str | None = None,
        scan_type: str | None = None,
        max_linear: float | None = None,
        max_angular: float | None = None,
        max_duration: float | None = None,
        publish_rate: float = 10.0,
        init_services: list[dict[str, Any]] | None = None,
    ) -> None:
        self.node_name = self._check_name("node_name", node_name)
        self.cmd_vel_topic = self._check_topic("cmd_vel_topic", cmd_vel_topic)
        self.odom_topic = self._check_topic("odom_topic", odom_topic) if odom_topic else None
        self.scan_topic = self._check_topic("scan_topic", scan_topic) if scan_topic else None
        self.transport = transport
        self.cmd_vel_type = cmd_vel_type if cmd_vel_type is not None else transport.twist_type
        self.odom_type = odom_type
        self.scan_type = scan_type
        context = type(self).__name__
        self.max_linear = None if max_linear is None else positive_finite("max_linear", max_linear, context)
        self.max_angular = None if max_angular is None else positive_finite("max_angular", max_angular, context)
        self.max_duration = None if max_duration is None else positive_finite("max_duration", max_duration, context)
        self.publish_rate = positive_finite("publish_rate", publish_rate, context)
        self.init_services = list(init_services or [])
        if self.init_services and not self.supports("service_call"):
            raise ValueError(
                f"init_services needs a transport that can call services, but "
                f"{type(transport).__name__} does not implement service_call"
            )
        for item in self.init_services:
            self._check_topic("init_services service", item.get("service", ""))
            if not item.get("type"):
                raise ValueError(
                    f"init_services entry for {item.get('service')!r} is missing its 'type' "
                    "(expected an interface type like pkg/srv/Name)"
                )
        self._enabled = False

    # -- validation ---------------------------------------------------------

    @classmethod
    def _check(cls, label: str, value: str, pattern: re.Pattern[str], hint: str) -> str:
        """Return ``value`` if ``pattern`` accepts it, else raise naming ``hint``.

        Args:
            label: Parameter name to quote in the refusal.
            value: Candidate name.
            pattern: Grammar the value must satisfy.
            hint: Sentence describing ``pattern``, appended to the refusal.
                Passed in rather than read off the class so the two seams -
                node names and topics - each carry the hint for their own
                grammar.

        Returns:
            ``value`` unchanged.

        Raises:
            ValueError: If ``value`` is empty or ``pattern`` refuses it.
        """
        if not value or not pattern.match(value):
            raise ValueError(f"invalid {label}: {value!r}{hint}")
        return value

    @classmethod
    def _check_name(cls, label: str, value: str) -> str:
        return cls._check(label, value, cls._NAME_RE, cls._NAME_HINT)

    @classmethod
    def _check_topic(cls, label: str, value: str) -> str:
        return cls._check(label, value, cls._TOPIC_RE, cls._TOPIC_HINT)

    # -- capabilities -------------------------------------------------------

    def supports(self, capability: str) -> bool:
        """Whether this robot's transport implements an optional capability.

        ``publish`` and ``echo`` are always present. ``service_call`` and
        ``action_send_goal`` are optional, so anything that depends on them
        must ask first rather than assume.
        """
        return callable(getattr(self.transport, capability, None))

    @staticmethod
    def _error(text: str) -> dict[str, Any]:
        return {"status": "error", "content": [{"text": text}]}

    # -- command shape (the kinematics seam) --------------------------------

    def _cmd_fields(self, linear: float, angular: float, lateral: float = 0.0) -> dict[str, Any]:
        """Message fields for a body-frame velocity command.

        The one override point for kinematics. The default is the identity
        mapping onto a ``geometry_msgs`` Twist, which is correct for
        differential-drive and skid-steer bases and for any platform fronted by
        a Twist-compatible controller. An Ackermann car overrides this with a
        bicycle model; a holonomic base overrides it to use ``lateral``.

        ``lateral`` is a reserved seam for holonomic/mecanum platforms and is
        always ``0.0`` today: :meth:`drive` deliberately does not expose it yet,
        because a base that accepted a lateral velocity and then dropped it on a
        non-holonomic platform would be a silent lie. It lives in the signature
        now so that adding the first holonomic platform is a subclass plus one
        ``drive`` argument, not a change to every override in the tree.
        """
        fields: dict[str, Any] = {"linear": {"x": float(linear)}, "angular": {"z": float(angular)}}
        if lateral:
            fields["linear"]["y"] = float(lateral)
        return fields

    def _publish_cmd(
        self, linear: float, angular: float, count: int, tool_context: ToolContext | None = None
    ) -> dict[str, Any]:
        return self.transport.publish(
            topic=self.cmd_vel_topic,
            type=self.cmd_vel_type,
            fields=self._cmd_fields(linear, angular),
            count=count,
            rate=self.publish_rate,
            tool_context=tool_context,
        )

    # -- enable handshake ---------------------------------------------------

    def enable(self, tool_context: ToolContext | None = None) -> dict[str, Any]:
        """Run the :attr:`init_services` handshake once; idempotent on success.

        Stops at the first failing call and returns its structured error
        **without latching**, so a later attempt retries the whole sequence -
        the right behavior for a service that is merely not up yet.

        Args:
            tool_context: Operator context forwarded to each service call. An
                arming service commands the vehicle, so it reaches the same gate
                a velocity publish does rather than being exempt for arriving
                over a different verb.
        """
        if self._enabled:
            return {"status": "success", "content": [{"text": f"{self.node_name}: already enabled"}]}
        # Narrowed, not ignored: __init__ refuses init_services on a transport
        # without service_call, so this cast can only be reached once that check
        # has passed.
        transport = cast(ServiceCapable, self.transport)
        for item in self.init_services:
            result = transport.service_call(
                service=item["service"],
                type=item["type"],
                fields=item.get("fields", {}),
                tool_context=tool_context,
            )
            if result.get("status") != "success":
                return result
        self._enabled = True
        return {
            "status": "success",
            "content": [{"text": f"{self.node_name}: enabled ({len(self.init_services)} init calls)"}],
        }

    # -- drive contract -----------------------------------------------------

    def drive(
        self,
        linear: float = 0.0,
        angular: float = 0.0,
        duration: float | None = None,
        count: int = 1,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Command a body-frame velocity.

        Args:
            linear: Forward linear velocity (m/s).
            angular: Yaw angular velocity (rad/s).
            duration: When given, hold the command for this many seconds by
                publishing ``round(duration * publish_rate)`` messages. Takes
                precedence over ``count``.
            count: Number of messages to publish when ``duration`` is omitted.
            tool_context: Operator context forwarded to the transport, whose
                tool may gate this surface. Injected by the agent runtime for
                the ``drive_<suffix>`` tool; a programmatic caller that passes
                none is refused by a gating transport unless the surface is
                pre-approved out of band.

        Returns:
            The transport's publish result, or a structured error when the
            request was refused - or when the command was published and the
            trailing stop that ends it was not, which names the still-live
            topic and the stop's own cause. The module docstring states the
            full safety contract; the order here is deliberate and pinned by
            tests - finite check, then ``duration``, then the handshake, then
            publish.
        """
        # One domain rule per parameter, taken from the shared guards rather than
        # restated here, so the accepted set cannot drift between this base and
        # the rest of the codebase - and so the first refusable parameter is the
        # one named, instead of a merged message the caller has to unpick.
        request_error = (
            finite_number_error(linear, "linear", "drive")
            or finite_number_error(angular, "angular", "drive")
            or (
                positive_finite_number_error(duration, "duration", "drive")
                if duration is not None
                # ``duration`` supersedes ``count``, so ``count`` is the effective
                # horizon only when no duration was given; refusing a ``count``
                # this call never reads would reject a command that is valid.
                # Left unchecked, ``count=0`` publishes nothing and reports
                # success - a drive the caller believes happened.
                else positive_whole_number_error(count, "count", "drive")
            )
        )
        if request_error:
            return self._error(request_error)
        if duration is not None and self.max_duration is not None and duration > self.max_duration:
            return self._error(
                f"drive: duration {duration}s exceeds max_duration {self.max_duration}s "
                "- issue shorter commands instead of one long hold"
            )
        # Only after the request is known good may we touch the robot: an
        # invalid drive must not be what puts a vehicle into manual mode.
        if not self._enabled and self.init_services:
            enabled = self.enable(tool_context=tool_context)
            if enabled.get("status") != "success":
                return enabled
        v = float(linear) if self.max_linear is None else max(-self.max_linear, min(self.max_linear, float(linear)))
        w = (
            float(angular)
            if self.max_angular is None
            else max(-self.max_angular, min(self.max_angular, float(angular)))
        )
        n = max(1, round(duration * self.publish_rate)) if duration is not None else count
        # The trailing stop runs from ``finally`` so it goes out even when the
        # main publish raised, and its verdict is kept rather than dropped -
        # see :func:`failed_halt_error`.
        halt: dict[str, Any] | None = None
        try:
            result = self._publish_cmd(v, w, count=n, tool_context=tool_context)
        finally:
            # A timed or repeated command owns its own stop. Skipped for a
            # command that was already zero - there is nothing to undo. Carries
            # the same context as the command it undoes: a trailing zero that
            # could not reach the gate would leave the robot latched at speed.
            if (duration is not None or n > 1) and (v or w):
                halt = self._publish_cmd(0.0, 0.0, count=1, tool_context=tool_context)
        latched = failed_halt_error(result, halt, topic=self.cmd_vel_topic, subject=LATCHED_VELOCITY)
        return self._error(latched) if latched else result

    def stop(self, tool_context: ToolContext | None = None) -> dict[str, Any]:
        """Publish a single zero-velocity command.

        Never gated on the :meth:`enable` handshake - an emergency stop must not
        need a working service graph. It is not exempt from the transport tool's
        own command gate, which is keyed on the surface rather than the payload:
        zero means "stationary" on a ``Twist`` but commands motion to the zero
        pose on a joint-command topic, so a payload-shaped carve-out could not
        be written correctly. The halt stays reachable through the same approval
        paths as any other command instead.

        Args:
            tool_context: Operator context forwarded to the transport.
        """
        return self._publish_cmd(0.0, 0.0, count=1, tool_context=tool_context)

    # -- sensing ------------------------------------------------------------

    def get_pose(self, timeout: float = 5.0) -> dict[str, Any]:
        """Read one sample from the robot's odometry/pose topic."""
        if not self.odom_topic:
            return self._error("get_pose: no odom_topic configured for this robot")
        return self.transport.echo(topic=self.odom_topic, type=self.odom_type, count=1, timeout=timeout)

    def get_scan(self, timeout: float = 5.0) -> dict[str, Any]:
        """Read one sample from the robot's laser-scan topic."""
        if not self.scan_topic:
            return self._error("get_scan: no scan_topic configured for this robot")
        return self.transport.echo(topic=self.scan_topic, type=self.scan_type, count=1, timeout=timeout)

    # -- agent tools --------------------------------------------------------

    @property
    def tool_suffix(self) -> str:
        """Instance-unique suffix so several robots can share one ``Agent``."""
        return self.node_name.strip("/").replace("/", "_")

    def _drive_description(self) -> str:
        """Agent-facing description of ``drive``, including the real limits.

        The latch semantics are disclosed here rather than hidden: an agent that
        does not know a bare ``drive`` keeps running cannot plan around it.
        """
        limits = []
        if self.max_linear is not None:
            limits.append(f"linear m/s up to {self.max_linear}")
        if self.max_angular is not None:
            limits.append(f"angular rad/s up to {self.max_angular}")
        bounds = f" ({', '.join(limits)})" if limits else ""
        return (
            f"Drive the {self.node_name} robot at a linear/angular velocity{bounds}, "
            "with an optional duration in seconds. A command with a duration stops "
            "automatically when it ends; without one, the command latches until stop."
        )

    def _extra_tools(self) -> list[AgentTool]:
        """Platform-specific tools appended after the standard four."""
        return []

    @property
    def tools(self) -> list[AgentTool]:
        """This robot's capabilities as named strands agent tools.

        Built from what is actually wired, so the agent is never handed a tool
        that can only return "not configured": ``get_pose`` appears only with an
        ``odom_topic``, ``get_scan`` only with a ``scan_topic``. ``drive`` and
        ``stop`` are always present together - anything that can move must be
        stoppable through the same surface.

        Every tool that carries a command is declared ``@tool(context=True)``
        and forwards the injected context to the transport, so a transport whose
        tool gates its command surface prompts the operator rather than failing
        closed on every call. The read-only tools take no context because a
        read is never gated. Declaring them here is what makes this true for
        every transport at once, instead of once per bridge.
        """
        suffix = self.tool_suffix

        @tool(name=f"drive_{suffix}", description=self._drive_description(), context=True)
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

        agent_tools: list[AgentTool] = [drive, stop]
        if self.odom_topic:
            agent_tools.append(get_pose)
        if self.scan_topic:
            agent_tools.append(get_scan)
        agent_tools.extend(self._extra_tools())
        return agent_tools

    def __repr__(self) -> str:
        try:
            return (
                f"{type(self).__name__}(node_name={self.node_name!r}, cmd_vel_topic={self.cmd_vel_topic!r}, "
                f"odom_topic={self.odom_topic!r}, scan_topic={self.scan_topic!r})"
            )
        except AttributeError:
            return partial_construction_repr(self)
