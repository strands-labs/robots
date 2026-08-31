"""Native Feetech STS/SCS-series driver satisfying :class:`HardwareDriver`.

This driver drives the arm. :mod:`~strands_robots.drivers.feetech.bus` opens
the SCS serial port, so ``send_action`` writes goal positions and the joints
reach the mesh - that is :issue:`360` scope 1, which the driver named as
deferred when it landed as a stub against the driver seam (:issue:`353` /
:pr:`2734`).

What works and what does not:

* ``send_action`` - writes the commanded joints in one SYNC_WRITE frame.
  Targets are **degrees** (``gripper`` is percent open); a key may be spelled
  ``shoulder_pan`` or ``shoulder_pan.pos``, matching lerobot's suffix.
* ``bus`` / ``is_connected`` - the pair
  :func:`strands_robots.bus_access.joint_read_source` resolves, so an SO-arm
  publishes ``joints`` on the mesh state topic without a wrapper. This is the
  documented seam for a driver that owns its motors; a ``state()`` method is
  not part of :data:`~strands_robots.drivers.base.DRIVER_SURFACE` and no
  shipped driver has one.
* ``stop`` - releases torque on every motor, reporting any that stayed driven.
* ``start_task`` / ``run_policy`` - still refused, and now for the *accurate*
  reason: a policy needs a control loop (rate, action horizon, a thread that
  can be stopped), which is its own slice. The refusal no longer blames the
  bus, because the bus is here.

None of this pretends. Every refusal returns an envelope of the same shape a
successful path returns, so the mesh and the agent need no code change on the
day the policy loop lands.

The class is registered for every Feetech robot the package registry knows
about - see :func:`~strands_robots.drivers._register_shipped_drivers` for the
list. Registering after import (``from strands_robots.drivers.feetech import
FeetechDriver`` then :func:`register_native_driver`) is also supported and
is how an out-of-tree driver package would extend the table.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

from strands_robots.bus_access import bus_lock
from strands_robots.drivers.feetech.bus import SO_ARM_MOTORS, FeetechBus
from strands_robots.utils import boolean_flag_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The robots this driver serves. Every entry corresponds to a canonical name
# in ``strands_robots/registry/robots.json``. The list is deliberately narrow:
# a Feetech driver could in principle serve every arm on that bus, but the
# ones :issue:`360` names are the ones the acceptance criteria measure, and
# registering for a robot we cannot verify is a promise this driver does not
# yet keep.
#
# Mirrors :data:`~strands_robots.drivers.dynamixel.driver.SUPPORTED_ROBOTS`
# in shape and spirit: canonical names, deliberately excluding any robot the
# scope note does not name.
# ---------------------------------------------------------------------------
# ``hope_jr`` and ``open_duck_mini`` share the SCS bus but NOT the six-servo
# SO-arm layout, and no joint map for either is established in this package. A
# driver built for one of them gets :data:`SO_ARM_MOTORS` by default, which
# names joints it does not have; pass ``motor_ids=`` (or a ``motors=`` map on
# the bus) until a verified map for those two lands. Said here rather than
# discovered on the wire.
SUPPORTED_ROBOTS: tuple[str, ...] = (
    "so100",
    "so101",
    "lekiwi",
    "hope_jr",
    "open_duck_mini",
)

_TOOL_TYPE = "robot"

# Refusal reason shared by the policy verbs. The literal string is checked in
# tests, so a change here is a change to the driver contract. It names the
# control loop and not the bus: blaming the bus for a missing policy loop sends
# a caller to read serial code that already works.
_NO_POLICY_LOOP = "not wired yet (the policy control loop)"


class FeetechDriver:
    """Native SCS-protocol driver for the arms in :data:`SUPPORTED_ROBOTS`.

    Constructor contract matches :class:`~strands_robots.drivers.base.HardwareDriver`
    - the factory builds every native driver as ``driver_cls(tool_name=...,
    cameras=..., data_config=..., **kwargs)`` and forwards the caller's extras
    in ``kwargs``. Feetech-specific keywords land in ``kwargs``:

    * ``port`` - a serial device path (``/dev/tty.usbserial-*``) for the SCS
      bus. Optional at construction; the bus opens it on connect.
    * ``baud_rate`` - integer, defaults to ``1_000_000``. The Feetech default
      for STS3215 arms; SCS-series can also run at 500_000 or below and a
      caller who knows better passes it here.
    * ``motor_ids`` - the servo IDs on the bus, in wire order. Optional at
      construction; the bus discovers them on connect.
    """

    tool_type = _TOOL_TYPE

    def __init__(
        self,
        tool_name: str,
        cameras: Any | None = None,
        data_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._tool_name = tool_name
        self._cameras = cameras
        self._data_config = data_config
        # A Feetech arm today is one U-shape bus. Aloha-style bimanual rigs
        # are Dynamixel not Feetech, so we accept a single ``port`` and refuse
        # ``ports`` outright rather than pretend to multi-bus a family that
        # does not need it. The keyword is still tolerated in kwargs so a
        # caller mis-passing ``ports=[...]`` gets a named refusal rather than
        # a silent ignore.
        port = kwargs.pop("port", None)
        ports = kwargs.pop("ports", None)
        if ports is not None:
            raise ValueError(
                f"FeetechDriver({tool_name!r}): pass port= for the Feetech bus; "
                f"multi-bus rigs are not part of {SUPPORTED_ROBOTS}",
            )
        self._port: str | None = port
        self._baud_rate: int = int(kwargs.pop("baud_rate", 1_000_000))
        self._motor_ids: tuple[int, ...] = tuple(kwargs.pop("motor_ids", ()))
        # ``motor_ids`` narrows the arm to a subset of SO_ARM_MOTORS. Honoured
        # rather than recorded: a keyword that changes nothing is worse than one
        # that is refused, because the caller believes the arm is configured.
        # An ID this driver has no joint name for is refused for the same
        # reason - we would otherwise command a motor we cannot name.
        motors = dict(SO_ARM_MOTORS)
        if self._motor_ids:
            known = {spec.motor_id: name for name, spec in SO_ARM_MOTORS.items()}
            if unknown := sorted(set(self._motor_ids) - set(known)):
                raise ValueError(
                    f"FeetechDriver({tool_name!r}): motor_ids {unknown} are not on an SO-arm; "
                    f"ids {sorted(known)} map to {[known[i] for i in sorted(known)]}",
                )
            motors = {known[i]: SO_ARM_MOTORS[known[i]] for i in self._motor_ids}
        self._bus = FeetechBus(port=self._port, baud_rate=self._baud_rate, motors=motors)
        self._connect_error: str | None = None
        # Extras from the caller are kept for a downstream driver package
        # to consume; refusing them here would refuse a valid future
        # extension.
        self._extras = kwargs

    # ------------------------------------------------------------------ #
    # Tool surface.                                                       #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """Name the agent invokes this robot by."""
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Schema describing the actions the agent may request.

        Every verb here reaches the bus. ``home`` is deliberately absent: a
        home pose is a per-arm calibration this driver does not own, and
        refusing a verb the schema declares is worse than not declaring it - an
        agent that plans against the schema will pick a verb it sees.
        """
        return {
            "name": self._tool_name,
            "description": (
                f"Feetech-native driver for {self._tool_name} (SCS protocol). Joint targets are degrees; "
                "gripper is percent open."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "sensors", "move_to", "set_torque", "stop"],
                            "description": (
                                "status: connection + motor list; sensors: read joint positions; "
                                "move_to: command joints (targets, degrees); "
                                "set_torque: energize or release (enabled); stop: release torque."
                            ),
                        },
                        "targets": {
                            "type": "object",
                            "description": "move_to only: joint name -> degrees (gripper: percent open).",
                            "additionalProperties": {"type": "number"},
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "set_torque only: true energizes, false releases.",
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    @property
    def declared_verbs(self) -> list[str]:
        """The action verbs this driver's schema declares, in schema order.

        Read back out of :attr:`tool_spec` rather than restated, so the verb
        list an agent is handed when it fires an unknown action is the one the
        schema really carries. A hand-copied list drifts the moment a verb is
        added or narrowed, and the agent then corrects itself towards a verb
        that does not exist.
        """
        action_schema = self.tool_spec["inputSchema"]["json"]["properties"]["action"]
        return [str(verb) for verb in action_schema["enum"]]

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Handle one agent invocation and yield exactly one tool result.

        Follows the shape :class:`DynamixelDriver` uses for its own deferred
        motion path so a caller writes the same error-checking code either
        way.
        """
        del kwargs  # forward-compat only
        del invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        action = (tool_use.get("input") or {}).get("action", "status")
        if action == "status":
            envelope = {
                "status": "success",
                "content": [{"json": await self.get_status()}],
            }
        elif action == "sensors":
            envelope = self._read_joints_envelope()
        elif action == "move_to":
            envelope = self.send_action((tool_use.get("input") or {}).get("targets") or {})
        elif action == "set_torque":
            enabled = (tool_use.get("input") or {}).get("enabled", True)
            # Checked before use, never coerced into one: every non-boolean an
            # agent actually emits for this field reads as the WRONG state.
            # "false", "no" and "0" are all truthy strings, so a caller asking
            # to release an arm would energize it instead while the envelope
            # reported torque_enabled=True as a success. A refusal naming the
            # field is recoverable; a silently inverted torque command on a
            # loaded arm is not.
            #
            # Through `boolean_flag_error` rather than `isinstance(x, bool)` so
            # the accepted domain matches every other posture flag in the
            # package: that helper also admits a numpy boolean, which an
            # `isinstance` check refuses. A policy or array path handing over
            # `np.bool_(False)` is a legitimate release request, and turning it
            # into a refusal would leave an operator unable to de-energize.
            # `bool()` below narrows the accepted value for the type checker;
            # it runs only after the check has already ruled the value in.
            if text := boolean_flag_error(enabled, "enabled", "set_torque"):
                envelope = _refuse(text)
            else:
                envelope = self._set_torque_envelope(bool(enabled))
        elif action == "stop":
            envelope = self._set_torque_envelope(False)
        else:
            envelope = _refuse(
                f"FeetechDriver: unknown action {action!r}; declared verbs are {self.declared_verbs}",
            )
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Motion, task and policy paths. All refuse in the same envelope.     #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Command joint positions over the SCS bus.

        Args:
            action: Joint name -> target. Degrees for every joint, percent open
                for ``gripper``. A ``.pos`` suffix is accepted and stripped, so
                a lerobot-shaped action dict works unchanged.
            robot_name: Unused; this driver fronts exactly one arm.

        Returns:
            A success envelope naming the joints commanded, or an error
            envelope. Never raises: a driver is invoked as an agent tool, and
            an exception past dispatch is not something the caller can handle.
        """
        del robot_name
        if not isinstance(action, dict) or not action:
            return _refuse("send_action: pass a non-empty mapping of joint targets")
        targets = {str(key).removesuffix(".pos"): value for key, value in action.items()}
        try:
            with bus_lock(self):
                self._connect_if_needed()
                self._bus.write_goal_positions(targets)
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            return _refuse(f"send_action: {e}")
        return {
            "status": "success",
            "content": [{"json": {"commanded": targets, "unit": "degrees (gripper: percent open)"}}],
        }

    def start_task(
        self,
        task: str,
        robot_name: str | None = None,
        policy: Policy | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Refuse: the bus is live, but no policy control loop drives it yet."""
        del task, robot_name, policy, kwargs
        return _refuse(f"start_task: {_NO_POLICY_LOOP}")

    def run_policy(
        self,
        policy: Policy,
        robot_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Refuse: the bus is live, but no policy control loop drives it yet."""
        del policy, robot_name, kwargs
        return _refuse(f"run_policy: {_NO_POLICY_LOOP}")

    def get_task_status(self) -> dict[str, Any]:
        """Return an empty-but-well-formed envelope.

        A caller polling task status sees nothing running rather than an
        error, because "nothing to run" is the honest answer during the stub
        phase.
        """
        return {
            "status": "success",
            "content": [{"json": {"in_flight": False, "reason": _NO_POLICY_LOOP}}],
        }

    def stop_task(self) -> dict[str, Any]:
        """No-op success: there is nothing to stop."""
        return {"status": "success", "content": [{"text": f"stop_task: {_NO_POLICY_LOOP}"}]}

    def cleanup(self) -> None:
        """Close the serial port.

        Idempotent, and safe on a driver that never connected: the bus tracks
        whether it holds an open handle. Torque is deliberately left as it is -
        releasing it here would drop an arm holding a payload when a caller
        merely tore down a process; ``stop`` is the verb that de-energizes.
        """
        with bus_lock(self):
            self._bus.disconnect()

    # ------------------------------------------------------------------ #
    # Lifecycle and status.                                               #
    # ------------------------------------------------------------------ #

    @property
    def bus(self) -> FeetechBus:
        """The motor bus, for :func:`strands_robots.bus_access.joint_read_source`.

        Exposing a ``bus`` with ``sync_read`` alongside :attr:`is_connected` is
        the documented way a native driver publishes ``joints`` on the mesh
        state topic - see :func:`strands_robots.bus_access.read_joints`, which
        prefers this over a full observation so a dead camera cannot hide the
        joint positions.

        Every path in this class that touches the bus holds
        :func:`~strands_robots.bus_access.bus_lock` on ``self`` - the same lock
        :func:`~strands_robots.bus_access.read_joints` takes for the read it
        does through this property. A driver-side write outside that lock would
        interleave with a mesh-side read on a half-duplex bus and corrupt both
        frames, which is the collision that module exists to prevent.
        """
        return self._bus

    @property
    def is_connected(self) -> bool:
        """Whether the serial port is open, so a consumer can tell live from stale."""
        return self._bus.is_connected

    def _connect_if_needed(self) -> None:
        """Open the bus on first use, recording the reason when it fails."""
        if self._bus.is_connected:
            return
        try:
            self._bus.connect()
        except (ValueError, OSError) as e:
            self._connect_error = str(e)
            raise
        self._connect_error = None

    def _read_joints_envelope(self) -> dict[str, Any]:
        """Read joint positions into a tool envelope."""
        try:
            with bus_lock(self):
                self._connect_if_needed()
                joints = self._bus.sync_read()
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            return _refuse(f"sensors: {e}")
        return {
            "status": "success",
            "content": [{"json": {"joint_state": joints, "unit": "degrees (gripper: percent open)"}}],
        }

    def _set_torque_envelope(self, enabled: bool) -> dict[str, Any]:
        """Energize or release the arm, reporting any motor that stayed driven."""
        try:
            with bus_lock(self):
                self._connect_if_needed()
                failed = self._bus.set_torque(enabled)
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            return _refuse(f"set_torque: {e}")
        if failed:
            # A partial release is a safety fact, not a success: say which
            # joints are still driven rather than reporting the arm released.
            return _refuse(f"set_torque({enabled}): these motors did not answer and may still be driven: {failed}")
        return {"status": "success", "content": [{"json": {"torque_enabled": enabled}}]}

    def connect_eagerly(self) -> str | None:
        """Open the serial bus now, or name why it could not open.

        Returns:
            ``None`` once the port is open, otherwise the failure reason. A
            named string rather than a raise, because a caller cannot tell a
            raise here from a real hardware fault mid-session.
        """
        try:
            # Under the lock like every other bus path: opening the port writes
            # to it (a torque-enable sweep on first contact), and this is the
            # one `_connect_if_needed` caller outside an already-locked block.
            # A lock only guarantees anything where EVERY caller takes it.
            with bus_lock(self):
                self._connect_if_needed()
        except (ValueError, OSError) as e:
            return str(e)
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report the driver's construction and configuration.

        Shape matches :meth:`DynamixelDriver.get_status` so both peers publish
        identically; fields absent on a Feetech bus (an FSM, a battery
        percentage) are simply not in the payload.
        """
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "tool_type": self.tool_type,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        "port": self._port,
                        "baud_rate": self._baud_rate,
                        "motors": {name: spec.motor_id for name, spec in self._bus.motors.items()},
                        "motor_ids": list(self._motor_ids),
                        "supported_robots": list(SUPPORTED_ROBOTS),
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """De-energize every motor.

        Logs the motors that did not answer rather than raising: ``stop`` is
        called from teardown paths that cannot handle an exception, and a
        silent partial release would report an arm safe while joints are still
        driven.
        """
        envelope = self._set_torque_envelope(False)
        if envelope["status"] == "error":
            logger.error("%s: %s", self._tool_name, envelope["content"][0]["text"])


# ---------------------------------------------------------------------------
# Envelope helpers. Kept private and one-liner-ish rather than reaching for a
# shared library, because the shape is small and the tests grade against the
# literal envelope. Duplicated with :mod:`strands_robots.drivers.dynamixel.driver`
# on purpose: two drivers with two two-line helpers is smaller than one driver
# and one shared module that binds their evolution together.
# ---------------------------------------------------------------------------
def _refuse(message: str) -> dict[str, Any]:
    """Return an error envelope with ``message``, matching the "not wired" contract."""
    return {"status": "error", "content": [{"text": message}]}
