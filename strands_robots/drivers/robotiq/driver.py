"""Native Modbus TCP driver for the Robotiq 2F-85, satisfying :class:`HardwareDriver`.

``Robot("robotiq_2f85", mode="real", port="192.168.1.11")`` builds one of these
and the fingers move. The driver owns its socket, so unlike a lerobot-shaped
robot there is no inner device: it answers the joint read itself (see
:func:`strands_robots.bus_access.joint_read_source`), which is what puts the
gripper's aperture on the mesh state topic.

Why a native driver: lerobot has no robot type for a Robotiq gripper - it is an
end effector, not an arm - and the gripper does not speak a servo bus. It speaks
Modbus, three registers each way, and the honest client speaks Modbus. The
codec is :mod:`strands_robots.drivers.robotiq.protocol`; only the socket and the
activation sequence live here.

The activation sequence is the one piece of behaviour a caller cannot skip. A
2F-85 powers up unactivated and *ignores every position command* until it has
been told to activate and has finished its open-close calibration stroke - it
does not report an error, it simply does not move. So :meth:`connect_eagerly`
activates and waits for ``gSTA == ACTIVE`` rather than returning as soon as the
socket opens, and :meth:`send_action` refuses while the gripper is not activated
instead of writing a frame that would be silently dropped.

What this driver does not do: run a policy. A 1-DOF end effector has no
rollout of its own - it is commanded as one dimension of the arm's action, by
whichever driver owns that arm - so :meth:`start_task` and :meth:`run_policy`
refuse and name the path that does work. That is a design fact about a gripper,
not a deferred feature.

``mode="real"`` on this robot resolves here by default: the registry entries
declare ``hardware.driver = "strands"``, because the alternative
(:data:`~strands_robots.drivers.base.DEFAULT_DRIVER`) cannot build a robot
lerobot has no type for.
"""

from __future__ import annotations

import enum
import logging
import socket
import struct
import threading
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from strands_robots.drivers.robotiq.protocol import (
    DEFAULT_TCP_PORT,
    DEFAULT_UNIT_ID,
    INPUT_BASE,
    MAX_COUNTS,
    MBAP_SIZE,
    OUTPUT_BASE,
    REGISTER_COUNT,
    STROKE_MM,
    ActivationStatus,
    FunctionCode,
    ObjectStatus,
    ProtocolError,
    aperture_mm_to_counts,
    closed_fraction_to_counts,
    command_registers,
    counts_to_aperture_mm,
    parse_response,
    parse_status,
    read_input_registers_frame,
    read_registers_payload,
    write_registers_frame,
)
from strands_robots.utils import (
    finite_number_error,
    positive_finite_number_error,
    tcp_port_error,
)

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)


SUPPORTED_ROBOTS: tuple[str, ...] = ("robotiq_2f85", "robotiq_2f85_v4")
"""Robots this driver registers for. Both registry entries are the same
gripper - ``robotiq_2f85_v4`` is the newer simulation model of the same
hardware - and the register map is identical, so one driver serves both."""

_NORMALISED_KEYS: tuple[str, ...] = ("gripper", "gripper.pos")
"""``send_action`` keys carrying a closed fraction, ``0.0`` open to ``1.0``
closed. Both spellings because a policy action dict names the joint
``gripper.pos`` while a hand-written call says ``gripper``, and a caller should
not have to discover which one this driver happens to read."""

_APERTURE_KEYS: tuple[str, ...] = ("position", "aperture_mm")
"""``send_action`` keys carrying a fingertip aperture in millimetres."""

_MODIFIER_KEYS: tuple[str, ...] = ("speed", "force")
"""``send_action`` keys scaling the motion, ``0.0``..``1.0`` of the gripper's
maximum. Absent means keep the driver's configured default."""

_NO_POLICY = (
    "a 2F-85 is a 1-DOF end effector with no rollout of its own - command it as "
    "one dimension of the arm's action through send_action({'gripper': ...}), or "
    "run the policy against the arm that carries it"
)


class _ModbusTcpClient:
    """One Modbus TCP connection: framed reads, serialised writes.

    Modbus TCP runs over a byte stream, so a reply must be read by its declared
    length rather than by whatever one ``recv`` returns - a short read that is
    parsed as a whole frame decodes a position the gripper never sent. The
    header is read first, its length field consulted, then exactly that many
    further bytes.

    Writes and their replies are serialised under one lock so two threads
    cannot interleave request and response on the same socket; the transaction
    id is checked on top of that, which catches a desynchronised stream rather
    than trusting the lock alone.
    """

    def __init__(self, host: str, port: int, unit_id: int, timeout: float) -> None:
        self._host = host
        self._port = port
        self._unit_id = unit_id
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._transaction = 0

    @property
    def alive(self) -> bool:
        """Whether the socket is open."""
        return self._sock is not None

    def connect(self) -> None:
        """Open the TCP connection. Raises on failure so the caller names it."""
        self._sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        self._sock.settimeout(self._timeout)

    def _next_transaction(self) -> int:
        self._transaction = (self._transaction + 1) % 0x10000
        return self._transaction

    def _read_exactly(self, sock: socket.socket, count: int) -> bytes:
        """Read exactly ``count`` bytes, or raise naming what arrived."""
        chunks: list[bytes] = []
        remaining = count
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ProtocolError(f"gripper closed the connection after {count - remaining} of {count} bytes")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _exchange(self, request: bytes) -> bytes:
        """Send one frame and read its whole reply."""
        sock = self._sock
        if sock is None:
            raise ProtocolError("gripper socket is not connected")
        with self._lock:
            sock.sendall(request)
            header = self._read_exactly(sock, MBAP_SIZE)
            # The MBAP length counts the unit id plus the PDU, and the unit id
            # is the last header byte already in hand.
            (length,) = struct.unpack(">H", header[4:6])
            body = self._read_exactly(sock, length - 1)
        return header + body

    def write_registers(self, address: int, values: tuple[int, ...]) -> None:
        """Write ``values`` at ``address`` and validate the acknowledgement.

        Raises:
            ProtocolError: If the gripper answers an exception or a frame that
                does not match the request.
            OSError: If the socket fails.
        """
        transaction = self._next_transaction()
        reply = self._exchange(write_registers_frame(transaction, self._unit_id, address, values))
        parse_response(reply, transaction, FunctionCode.WRITE_MULTIPLE_REGISTERS)

    def read_input_registers(self, address: int, count: int) -> bytes:
        """Read ``count`` registers at ``address`` and return their bytes.

        Raises:
            ProtocolError: If the reply is malformed or an exception response.
            OSError: If the socket fails.
        """
        transaction = self._next_transaction()
        reply = self._exchange(read_input_registers_frame(transaction, self._unit_id, address, count))
        return read_registers_payload(reply, transaction, count)

    def close(self) -> None:
        """Close the socket. Idempotent."""
        sock, self._sock = self._sock, None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already half-closed; a shutdown race here is not actionable
        try:
            sock.close()
        except OSError:
            pass  # closing a socket that is already gone is fine


class RobotiqDriver:
    """Native Modbus TCP driver for the grippers in :data:`SUPPORTED_ROBOTS`.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver` structurally;
    the surface check in
    :func:`~strands_robots.drivers.register_native_driver` pins the contract at
    registration.
    """

    tool_type = "robot"

    def __init__(
        self,
        tool_name: str = "robotiq_2f85",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        tcp_port: int = DEFAULT_TCP_PORT,
        unit_id: int = DEFAULT_UNIT_ID,
        timeout: float = 2.0,
        activation_timeout: float = 10.0,
        stroke_mm: float = STROKE_MM,
        speed: float = 1.0,
        force: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` opens the socket.

        Args:
            tool_name: Name the agent invokes the driver by, and the mesh peer id.
            cameras: Accepted for parity with the other drivers; a gripper
                addresses no camera of its own.
            data_config: Accepted for parity; unused.
            port: The gripper's host - an IP address or hostname. ``port`` is the
                factory's polymorphic device keyword (a serial path for a servo
                bus, a host here), which is why the TCP port is a separate
                ``tcp_port``.
            tcp_port: Modbus TCP port; :data:`DEFAULT_TCP_PORT` unless the
                gripper is behind a controller that remaps it.
            unit_id: Modbus slave id. :data:`DEFAULT_UNIT_ID` is what Robotiq
                ships; a gripper behind a Universal Robots controller often
                answers on ``0``.
            timeout: Socket and reply timeout in seconds.
            activation_timeout: How long :meth:`connect_eagerly` waits for the
                power-on calibration stroke to finish. The manual's sequence has
                no faster completion signal than ``gSTA``.
            stroke_mm: Aperture at position count ``0``. Defaults to the
                2F-85's; a 2F-140 is the same protocol with a wider stroke.
            speed: Default speed, ``0.0``..``1.0`` of maximum.
            force: Default force, ``0.0``..``1.0`` of maximum.
            **kwargs: Ignored; accepted so the factory can forward extras.

        Raises:
            ValueError: If a numeric knob is outside its domain. Raised here
                rather than returned from :meth:`connect_eagerly`, which is
                declared ``-> str | None``: a value the transport cannot use is
                not a connection this driver can degrade to reporting.
        """
        del cameras, data_config
        if kwargs:
            logger.debug("RobotiqDriver ignoring extra kwargs: %s", sorted(kwargs))

        # Each of these reaches a consumer that cannot report what it was
        # handed: the timeouts go to socket.settimeout and to a deadline
        # comparison, tcp_port and unit_id are packed into a frame, and
        # speed/force are scaled into a byte. A nan or a string surfaces from
        # inside the socket call or from struct.pack, naming neither this
        # driver nor the parameter.
        for value, name, check in (
            (timeout, "timeout", positive_finite_number_error),
            (activation_timeout, "activation_timeout", positive_finite_number_error),
            (stroke_mm, "stroke_mm", positive_finite_number_error),
        ):
            if reason := check(value, name, "RobotiqDriver"):
                raise ValueError(reason)
        if reason := tcp_port_error(tcp_port, "tcp_port", "RobotiqDriver"):
            raise ValueError(reason)
        for value, name in ((speed, "speed"), (force, "force")):
            if reason := finite_number_error(value, name, "RobotiqDriver"):
                raise ValueError(reason)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"RobotiqDriver: {name} must be in 0.0..1.0, got {value}")
        if not isinstance(unit_id, int) or isinstance(unit_id, bool) or not 0 <= unit_id <= MAX_COUNTS:
            raise ValueError(f"RobotiqDriver: unit_id must be an int in 0..{MAX_COUNTS}, got {unit_id!r}")

        self._tool_name = tool_name
        self._host = str(port) if port else "127.0.0.1"
        self._tcp_port = int(tcp_port)
        self._unit_id = int(unit_id)
        self._timeout = float(timeout)
        self._activation_timeout = float(activation_timeout)
        self._stroke_mm = float(stroke_mm)
        self._speed = closed_fraction_to_counts(speed)
        self._force = closed_fraction_to_counts(force)

        self._client: _ModbusTcpClient | None = None
        self._connected = False
        self._connect_error: str | None = None
        self._cache_lock = threading.Lock()
        self._last_status: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Agent tool surface.                                                #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """The name the Strands agent invokes this driver by."""
        return self._tool_name

    @property
    def is_connected(self) -> bool:
        """Whether the Modbus connection is live, for the mesh joint read."""
        return self._connected and self._client is not None and self._client.alive

    @property
    def tool_spec(self) -> ToolSpec:
        """The verbs an agent may ask of a gripper.

        Every declared verb works. ``open`` and ``close`` are named separately
        from a position command because they are what an agent actually plans
        with, and making it compute a count for "let go" is a worse surface.
        """
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Robotiq 2F-85 native driver over Modbus TCP. Opens and closes the gripper, "
                    "reports fingertip aperture, grasp detection and faults."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "open: release to full aperture; "
                                    "close: grasp at the configured force; "
                                    "sensors: aperture, grasp state and current; "
                                    "status: connection and activation state; "
                                    "stop: halt the fingers where they are"
                                ),
                                "enum": ["open", "close", "sensors", "status", "stop"],
                                "default": "sensors",
                            },
                        },
                        "required": ["action"],
                    }
                },
            },
        )

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Handle one agent invocation and yield exactly one tool result."""
        del kwargs, invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        action = (tool_use.get("input") or {}).get("action", "sensors")
        envelope: dict[str, Any]
        if action == "open":
            envelope = self.send_action({"gripper": 0.0})
        elif action == "close":
            envelope = self.send_action({"gripper": 1.0})
        elif action == "sensors":
            envelope = self.read_status()
        elif action == "status":
            envelope = await self.get_status()
        else:  # "stop"
            # Delegate rather than re-derive: stop_task decides whether the halt
            # reached the gripper, and an agent that read a restated success here
            # would be told the fingers stopped when the write failed.
            envelope = self.stop_task()
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Lifecycle.                                                          #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Open the socket and activate the gripper.

        Activation is part of connecting, not a separate call, because an
        unactivated 2F-85 accepts a position command and does not move. A
        driver that returned success on an open socket alone would report a
        healthy gripper that silently ignores every action.

        Returns:
            ``None`` when the gripper is connected and activated, or a reason
            naming what failed.
        """
        client = _ModbusTcpClient(self._host, self._tcp_port, self._unit_id, self._timeout)
        try:
            client.connect()
        except OSError as exc:
            self._connect_error = f"cannot reach the gripper at {self._host}:{self._tcp_port} - {exc}"
            client.close()
            return self._connect_error
        self._client = client
        try:
            reason = self._activate()
        except (OSError, ProtocolError) as exc:
            reason = f"activation failed: {exc}"
        if reason is not None:
            client.close()
            self._client = None
            self._connect_error = reason
            return reason
        self._connected = True
        self._connect_error = None
        return None

    def _activate(self) -> str | None:
        """Run the manual's activation sequence. Returns a reason or ``None``."""
        status = self._read_status_raw()
        if status["activation"] is ActivationStatus.ACTIVE and status["activated"]:
            return None
        # A reset first: the manual requires rACT be cleared before a fresh
        # activation, and a gripper holding a fault will not activate without it.
        self._write_command(activate=False, go_to=False)
        self._write_command(activate=True, go_to=False)
        deadline = time.monotonic() + self._activation_timeout
        while time.monotonic() < deadline:
            status = self._read_status_raw()
            if status["activation"] is ActivationStatus.ACTIVE:
                return None
            time.sleep(0.1)
        return (
            f"gripper did not finish activating within {self._activation_timeout}s "
            f"(gSTA={status['activation'].name}, gFLT={status['fault']!r})"
        )

    async def get_status(self) -> dict[str, Any]:
        """Report the driver's connection and the gripper's activation state."""
        payload: dict[str, Any] = {
            "tool_name": self._tool_name,
            "tool_type": self.tool_type,
            "connected": self.is_connected,
            "connect_error": self._connect_error,
            "host": self._host,
            "tcp_port": self._tcp_port,
            "unit_id": self._unit_id,
            "stroke_mm": self._stroke_mm,
            "supported_robots": list(SUPPORTED_ROBOTS),
            # Part of the tool_name/connected/battery_pct triple every shipped
            # driver reports, so a mesh consumer reads one shape for every peer.
            # Always None here: the gripper is powered from the arm's controller
            # and reports no state of charge, and inventing one would be worse
            # than an honest absence.
            "battery_pct": None,
        }
        with self._cache_lock:
            cached = self._last_status
        if cached is not None:
            payload["gripper"] = _readable(cached)
        return {"status": "success", "content": [{"json": payload}]}

    async def stop(self) -> None:
        """Halt the fingers where they are, leaving the gripper activated.

        Clearing ``rGTO`` stops motion without dropping the activation, so the
        next :meth:`send_action` moves immediately rather than paying for
        another calibration stroke. A stop on a disconnected gripper is a no-op
        rather than an error - there is nothing moving to halt.
        """
        if not self.is_connected:
            return
        try:
            self._write_command(activate=True, go_to=False)
        except (OSError, ProtocolError) as exc:
            logger.debug("stop could not reach the gripper: %s", exc)

    def cleanup(self) -> None:
        """Close the socket. Leaves the gripper activated and holding position."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._connected = False

    # ------------------------------------------------------------------ #
    # Command path.                                                       #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Command an aperture.

        Gates, in order: this driver fronts this robot; it is connected and
        activated; the action names exactly one aperture spelling; the values
        are finite. A refusal is an envelope, never an exception, so the mesh
        command path handles a bad action the same way it handles a dead socket.

        Args:
            action: One of :data:`_NORMALISED_KEYS` carrying a closed fraction
                (``0.0`` open, ``1.0`` closed), or one of
                :data:`_APERTURE_KEYS` carrying a fingertip aperture in
                millimetres. Optionally :data:`_MODIFIER_KEYS` to scale speed
                and force for this command.
            robot_name: Which robot to command; ``None`` or this driver's own
                name. A gripper fronts exactly one device.

        Returns:
            A success envelope naming the commanded count and aperture, or an
            error envelope naming the reason.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return _refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        if not self.is_connected or self._client is None:
            return _refuse("send_action: not connected - call connect_eagerly() first")

        normalised = [key for key in _NORMALISED_KEYS if key in action]
        aperture = [key for key in _APERTURE_KEYS if key in action]
        if len(normalised) + len(aperture) > 1:
            return _refuse(
                f"send_action: {sorted(normalised + aperture)} are two spellings of the same command - "
                f"pass a closed fraction ({' or '.join(_NORMALISED_KEYS)}) or millimetres "
                f"({' or '.join(_APERTURE_KEYS)}), not both"
            )
        if not normalised and not aperture:
            return _refuse(
                f"send_action: nothing to command - none of {sorted(action)} names an aperture; "
                f"expected one of {sorted(_NORMALISED_KEYS + _APERTURE_KEYS)}"
            )

        key = (normalised or aperture)[0]
        for name in (key, *(m for m in _MODIFIER_KEYS if m in action)):
            if reason := finite_number_error(action[name], name, "send_action"):
                return _refuse(reason)

        try:
            counts = (
                closed_fraction_to_counts(action[key])
                if normalised
                else aperture_mm_to_counts(action[key], self._stroke_mm)
            )
            speed = closed_fraction_to_counts(action["speed"]) if "speed" in action else self._speed
            force = closed_fraction_to_counts(action["force"]) if "force" in action else self._force
            self._write_command(activate=True, go_to=True, position=counts, speed=speed, force=force)
        except ProtocolError as exc:
            return _refuse(f"send_action: {exc}")
        except OSError as exc:
            return _refuse(f"send_action: writing to the gripper failed: {exc}")

        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "commanded_key": key,
                        "position": counts,
                        "aperture_mm": counts_to_aperture_mm(counts, self._stroke_mm),
                        "speed": speed,
                        "force": force,
                    }
                }
            ],
        }

    def read_status(self) -> dict[str, Any]:
        """Read the gripper's live status.

        Returns:
            A success envelope carrying the decoded status, or an error
            envelope naming why the read failed.
        """
        if not self.is_connected:
            return _refuse("read_status: not connected - call connect_eagerly() first")
        try:
            status = self._read_status_raw()
        except ProtocolError as exc:
            return _refuse(f"read_status: {exc}")
        except OSError as exc:
            return _refuse(f"read_status: reading the gripper failed: {exc}")
        return {"status": "success", "content": [{"json": _readable(status)}]}

    def get_observation(self) -> dict[str, float]:
        """Report the gripper's one joint, for the mesh state topic.

        The key is ``gripper.pos`` because that is the name a policy action
        dict uses for this degree of freedom, so an observation and the action
        that answers it agree.

        Returns:
            The closed fraction under ``gripper.pos``, or an empty mapping when
            the gripper cannot be read - "no joints" is not a failure.
        """
        if not self.is_connected:
            return {}
        try:
            status = self._read_status_raw()
        except (OSError, ProtocolError) as exc:
            logger.debug("joint read failed: %s", exc)
            return {}
        return {"gripper.pos": status["position"] / MAX_COUNTS}

    # ------------------------------------------------------------------ #
    # Task and policy paths - a gripper has no rollout, so these refuse.  #
    # ------------------------------------------------------------------ #

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Refuse: see :data:`_NO_POLICY`."""
        del instruction, policy_port, policy_host, policy_provider, duration, policy_kwargs
        return _refuse(f"start_task: {_NO_POLICY}")

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refuse: see :data:`_NO_POLICY`."""
        del policy_object, instruction, duration, n_steps
        return _refuse(f"run_policy: {_NO_POLICY}")

    def get_task_status(self) -> dict[str, Any]:
        """Report that no task runs, which is always true for a gripper."""
        return {"status": "success", "content": [{"json": {"in_flight": False, "reason": _NO_POLICY}}]}

    def stop_task(self) -> dict[str, Any]:
        """Halt the fingers. There is no task, but a caller asking to stop means it."""
        if not self.is_connected:
            return {"status": "success", "content": [{"text": "stop_task: not connected, nothing to stop"}]}
        try:
            self._write_command(activate=True, go_to=False)
        except (OSError, ProtocolError) as exc:
            return _refuse(f"stop_task: {exc}")
        return {"status": "success", "content": [{"text": f"stop_task: {self._tool_name} fingers halted"}]}

    # ------------------------------------------------------------------ #
    # Wire helpers.                                                       #
    # ------------------------------------------------------------------ #

    def _write_command(self, **fields: Any) -> None:
        """Write one output frame. Raises ``OSError``/``ProtocolError``."""
        if self._client is None:
            raise ProtocolError("gripper socket is not connected")
        self._client.write_registers(OUTPUT_BASE, command_registers(**fields))

    def _read_status_raw(self) -> dict[str, Any]:
        """Read and decode the input registers, caching the result."""
        if self._client is None:
            raise ProtocolError("gripper socket is not connected")
        payload = self._client.read_input_registers(INPUT_BASE, REGISTER_COUNT)
        status = parse_status(payload)
        status["aperture_mm"] = counts_to_aperture_mm(status["position"], self._stroke_mm)
        status["holding"] = status["object"] in (ObjectStatus.CONTACT_OPENING, ObjectStatus.CONTACT_CLOSING)
        with self._cache_lock:
            self._last_status = status
        return status


def _readable(status: dict[str, Any]) -> dict[str, Any]:
    """Render a decoded status with its enums as names, for an envelope.

    An ``IntEnum`` serialises as a bare integer, so a consumer reading
    ``object: 2`` off the wire has to own a copy of the map to know it means a
    grasp. Naming it in the payload keeps the reading self-describing.

    Args:
        status: A :func:`~strands_robots.drivers.robotiq.protocol.parse_status`
            result.

    Returns:
        A copy with the enum members replaced by their names.
    """
    rendered = dict(status)
    for field in ("activation", "object", "fault"):
        value = rendered.get(field)
        # An undocumented fault arrives as a plain int and is passed through, so
        # this cannot assume every field is an enum member.
        if isinstance(value, enum.Enum):
            rendered[field] = value.name
    return rendered


def _refuse(message: str) -> dict[str, Any]:
    """Return an error envelope carrying ``message``."""
    return {"status": "error", "content": [{"text": message}]}
