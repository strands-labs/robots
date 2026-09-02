"""The contract a ``mode="real"`` driver satisfies.

:func:`~strands_robots.robot.Robot` with ``mode="real"`` builds one object and
hands it to everything downstream: a Strands agent invokes it as a tool, the
Zenoh mesh publishes from it, and the teleop rail commands through it. Until a
second implementation existed that object was always
:class:`strands_robots.hardware_robot.Robot`, so the surface those consumers
rely on was recorded nowhere - a driver author had to read a 3000-line class to
learn which members are load-bearing.

:class:`HardwareDriver` writes that surface down. It is deliberately the
*measured* contract rather than an aspirational one, which makes it smaller than
a reader might expect:

* ``get_observation`` is **not** a member. A lerobot robot is a wrapper: it
  holds the device that owns the bus under ``robot``, and
  :func:`strands_robots.bus_access.read_observation` takes that inner device -
  so the top-level driver is not the thing asked for a frame.

  A native driver has no inner device, and for joint telemetry that is now
  resolved rather than assumed:
  :func:`strands_robots.bus_access.joint_read_source` prefers ``robot.robot``
  and falls back to the driver itself, so a driver that owns its bus publishes
  ``joints`` on the state topic by exposing either a ``bus`` with ``sync_read``
  or a ``get_observation``, plus ``is_connected`` to say it is live. Neither is
  required, which is why neither is a member: a driver with no motors to report
  is otherwise complete.
* The sensor attributes a mesh publishes (``_pose``, ``_imu``, ``_battery`` and
  their siblings) are **not** members either. Every one is read with a
  ``getattr(robot, name, None)`` default, so a driver with no IMU publishes no
  IMU topic and is otherwise complete - making them optional by construction. A
  Protocol cannot express "optional", and requiring them would refuse a
  perfectly good arm for lacking a lidar.

What remains is the surface a driver must have for an agent to call it and for
the mesh's command and task paths to work. The four tool members
(``tool_name``, ``tool_type``, ``tool_spec``, ``stream``) are also the abstract
surface of :class:`strands.tools.tools.AgentTool`, which is what makes an object
usable as a Strands tool at all.

Structural, not nominal: a driver satisfies this by having the members, with no
import of - or inheritance from - anything here. Inheriting ``AgentTool`` is
still the easy way to get the tool quarter right.

Constructor contract (a Protocol cannot express ``__init__``): the factory
builds a native driver as
``driver_cls(tool_name=<canonical name>, cameras=<cameras or None>,
data_config=<data_config or None>, **kwargs)``, so a driver must accept those
three keywords and tolerate the caller's extras. ``port=`` arrives in
``**kwargs`` and stays polymorphic - a serial path, an IP address or a URL,
interpreted by the driver that receives it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy


@runtime_checkable
class HardwareDriver(Protocol):
    """The surface :func:`~strands_robots.robot.Robot` ``mode="real"`` returns.

    See the module docstring for what is deliberately absent and why.
    """

    # --- Agent tool surface (also ``AgentTool``'s abstract members) --------- #

    @property
    def tool_name(self) -> str:
        """Name the agent invokes this robot by."""

    @property
    def tool_type(self) -> str:
        """Tool kind reported to the agent runtime."""

    @property
    def tool_spec(self) -> ToolSpec:
        """Schema describing the actions the agent may request."""

    def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Run one agent tool call, yielding events and finally the result.

        Args:
            tool_use: The agent's request, carrying the tool id and parameters.
            invocation_state: Caller-provided state passed to the agent.
            **kwargs: Additional keyword arguments, for forward compatibility.

        Yields:
            Tool events, the last of which is the tool result. Spelled out
            rather than borrowing ``strands.types.tools.ToolGenerator``, which
            is an alias for exactly this type: the reference implementation
            spells it out too, and naming the alias would add an SDK symbol this
            package does not otherwise depend on.
        """

    # --- Command path ------------------------------------------------------ #

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Command one action.

        Args:
            action: Joint targets, keyed the way this driver names its joints.
            robot_name: Which robot to command when the driver fronts several;
                ``None`` means the driver's own robot.

        Returns:
            A status envelope describing what was commanded.
        """

    # --- Task and policy path ---------------------------------------------- #

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Start a policy-driven task in the background.

        Args:
            instruction: Natural-language instruction for the policy.
            policy_port: Port the policy server listens on; ``None`` uses the
                provider's default.
            policy_host: Host the policy server runs on.
            policy_provider: Which policy provider to build.
            duration: Wall-clock budget for the task, in seconds.
            **policy_kwargs: Extra provider-specific policy options.

        Returns:
            A status envelope describing the task that started.
        """

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run an already-built policy against this robot.

        Args:
            policy_object: The policy to roll out.
            instruction: Natural-language instruction handed to the policy.
            duration: Wall-clock budget for the rollout, in seconds.
            n_steps: Step budget; when given it wins over ``duration``.

        Returns:
            A status envelope describing the rollout.
        """

    def get_task_status(self) -> dict[str, Any]:
        """Report the running task's state.

        Returns:
            A status envelope; the shape a caller polls between steps.
        """

    def stop_task(self) -> dict[str, Any]:
        """Stop the running task.

        Returns:
            A status envelope describing what was stopped.
        """

    # --- Lifecycle --------------------------------------------------------- #

    async def get_status(self) -> dict[str, Any]:
        """Report the driver's own health and connection state.

        Returns:
            A status envelope the mesh publishes as this peer's presence.
        """

    async def stop(self) -> None:
        """Stop motion and any background loop, leaving the robot connected.

        Annotated ``-> None``, so it carries no verdict: a caller that needs the
        halt outcome reads :meth:`stop_task`, which decides one. That is exactly
        what makes the log the *only* place a halt this hook could not complete
        can be recorded, so an implementation that delegates to a halt verb must
        read that verb's envelope and log a non-success, naming what may still be
        moving. Discarding it returns from shutdown reporting a robot as stopped
        on the one surface that carries no way to say otherwise.
        :func:`halt_failure_detail` reads the reason out of such an envelope.
        """

    def cleanup(self) -> None:
        """Release the device and every background resource held for it."""


#: The driver a robot gets when nothing says otherwise. Every robot in the
#: package registry is a lerobot robot today, so the default keeps them working
#: without a per-robot declaration.
DEFAULT_DRIVER = "lerobot"

#: Accepted ``driver=`` values. ``"auto"`` expresses no preference: it reads the
#: registry and falls back to :data:`DEFAULT_DRIVER`. Mirrors the
#: :data:`~strands_robots.registry.LIST_ROBOTS_MODES` pattern - a value outside
#: this tuple is refused by name rather than silently treated as the default,
#: because a typo that resolves to a working driver is a caller who never learns
#: the driver they asked for does not exist.
DRIVER_CHOICES = ("auto", DEFAULT_DRIVER, "strands")


#: Every member :class:`HardwareDriver` requires, derived from the Protocol
#: itself so the two can never disagree. A second hand-written list would be a
#: second source of truth, and the one that drifts is always the copy.
DRIVER_SURFACE: tuple[str, ...] = tuple(sorted(name for name in dir(HardwareDriver) if not name.startswith("_")))


def missing_driver_members(candidate: object) -> tuple[str, ...]:
    """Report which :data:`DRIVER_SURFACE` members ``candidate`` does not have.

    Answers for a *class* as well as an instance, which is what a caller
    holding a driver class before construction needs -
    :func:`issubclass` cannot: a Protocol declaring a ``@property`` has
    non-method members, and ``issubclass`` refuses those outright with
    ``TypeError``.

    Args:
        candidate: A driver class or a built driver instance.

    Returns:
        The missing member names in sorted order; empty when ``candidate``
        satisfies the whole surface.
    """
    return tuple(name for name in DRIVER_SURFACE if not hasattr(candidate, name))


def halt_failure_detail(envelope: dict[str, Any]) -> str | None:
    """Read why a halt did not complete, or ``None`` when it did.

    :meth:`HardwareDriver.stop` carries no verdict, so the envelope of the halt
    verb it delegates to is the only one there is and the log is the only place
    it survives. This renders that envelope as the detail such a log line
    quotes, in one place rather than once per driver, because the shipped halt
    verbs answer in two shapes: a refusal *text*, and a per-half *outcome* dict
    naming which half of a two-part halt failed.

    Args:
        envelope: The halt verb's own status envelope.

    Returns:
        The reason, or ``None`` when ``envelope`` reports success. A non-success
        envelope always yields a string - a failure whose content this cannot
        parse still reports as a failure, because returning ``None`` there would
        read as "the halt landed".
    """
    if envelope.get("status") == "success":
        return None
    blocks = [block for block in envelope.get("content") or [] if isinstance(block, dict)]
    if texts := [str(block["text"]).strip() for block in blocks if block.get("text")]:
        return " ".join(texts)
    if payloads := [block["json"] for block in blocks if "json" in block]:
        return "; ".join(
            ", ".join(f"{key}={value!r}" for key, value in sorted(payload.items()))
            if isinstance(payload, dict)
            else repr(payload)
            for payload in payloads
        )
    return "no detail reported"
