"""ReachyMiniDriver - Device Connect DeviceDriver for Pollen Reachy Mini robots.

Auto-detects hardware variant via the daemon's ``wireless_version`` flag:
- **Wireless** (has onboard Pi): uses Zenoh transport for real-time I/O.
- **Lite** (USB-only, no Pi): uses WebSocket to the daemon directly.

REST API calls go through reachy_transport.api() for daemon/move operations.
"""

import asyncio
import logging
import math
import re
from typing import Any

from device_connect_edge.drivers import DeviceDriver, emit, get_rpc_source_device, on, rpc
from device_connect_edge.types import DeviceIdentity, DeviceStatus

from strands_robots.device_connect._authz import authz_error, is_authorized_caller
from strands_robots.device_connect.reachy_transport import (
    WebSocketLink,
    ZenohLink,
    api,
    identity_pose,
    rpy_to_pose,
)
from strands_robots.mesh.security import ValidationError, validate_mesh_identifier
from strands_robots.tools.reachy import envelope_error
from strands_robots.utils import finite_number_error, tcp_port_error

logger = logging.getLogger(__name__)

# Security hardening: recorded-move names are interpolated into a REST URL
# path, so restrict them to a safe charset to prevent path traversal and
# query/parameter injection into the daemon API.
_MOVE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _key_prefix_error(value: Any, param: str, cls_name: str) -> str | None:
    """Structured rejection when a Zenoh key prefix cannot address one robot.

    The prefix is interpolated verbatim into the three key expressions the
    Wireless variant lives on -- ``<prefix>/joint_positions`` and
    ``<prefix>/imu_data`` in :meth:`ZenohLink.start`, and ``<prefix>/command``
    in :meth:`ZenohLink.send_cmd` -- and nothing downstream narrows it. Two
    unusable shapes reach the wire from there, and they fail in opposite
    directions:

    * Zenoh reads ``*`` and ``**`` as key-expression wildcards, and accepts
      them on a *publisher* key as readily as on a subscriber one. So a
      wildcard prefix widens ``<prefix>/command`` from one robot's inbox into a
      match-any key: a single :meth:`ReachyMiniDriver.look` is delivered to
      every Reachy Mini subscribed beneath the pattern, and the reciprocal
      subscriptions fold every robot's joint and IMU frames into the one
      driver's ``_latest_joints`` / ``_latest_imu``. Nothing reports this --
      the widened call succeeds.
    * The shapes Zenoh refuses outright (an empty segment, a stray ``?``) are
      not refused here either. They surface from inside the transport at
      :meth:`ZenohLink.start`, after :meth:`ReachyMiniDriver.connect` has
      already probed the daemon and logged a connection, rather than from the
      constructor call that named them.

    So the accepted domain is "a ``/``-joined sequence of mesh identifiers":
    each segment goes through the shared
    :func:`~strands_robots.mesh.security.validate_mesh_identifier`, whose own
    docstring gives the reason -- it exists because an unvalidated key-
    expression segment "silently widens a point-to-point subscription into a
    match-any one". A multi-segment prefix stays legitimate, so namespacing two
    Minis apart as ``reachy_mini/robot_a`` and ``reachy_mini/robot_b`` is
    unaffected; it is the wildcard that is refused, not the ``/``.

    Args:
        value: Candidate prefix, as the caller supplied it.
        param: Parameter name, used in the message.
        cls_name: Constructing class, used as the message prefix.

    Returns:
        The rejection message, or ``None`` when every segment can address a
        single robot.
    """
    label = f"{cls_name}.{param}"
    if not isinstance(value, str):
        return f"{label} must be a string (got {type(value).__name__})"
    for segment in value.split("/"):
        try:
            validate_mesh_identifier(segment, f"{label} segment {segment!r}")
        except ValidationError as exc:
            return str(exc)
    return None


#: Per-RPC map from a movement RPC's own parameter name to the envelope axis it
#: commands. This surface spells the head axes ``pitch`` / ``roll`` / ``yaw``
#: where :data:`~strands_robots.tools.reachy.MOTION_ENVELOPE_DEG` keys them
#: ``head_pitch`` / ``head_roll`` / ``head_yaw``, so the values have to be
#: re-keyed before the envelope can bound them - handing it this RPC's own
#: keyword dict would bound nothing, because it ignores a key it has no limit
#: for. ``look``'s millimetre offsets and ``antennas``' angles carry no entry
#: because the envelope declares no bound for them.
_ENVELOPE_AXIS_BY_PARAM: dict[str, dict[str, str]] = {
    "look": {"pitch": "head_pitch", "roll": "head_roll", "yaw": "head_yaw"},
    "body": {"yaw": "body_yaw"},
}


def _motion_domain_error(rpc_name: str, values: dict[str, Any]) -> dict[str, str] | None:
    """Structured rejection when a motion argument cannot reach the robot.

    Every value in *values* is a signed physical quantity -- an angle in degrees
    or an offset in millimetres -- that the movement RPCs carry verbatim into a
    command dict and hand to the active :class:`HardwareLink`. Both links put
    that dict on the wire as JSON, and Python's encoder emits ``nan`` / ``inf``
    as the bare tokens ``NaN`` / ``Infinity``, which RFC 8259 does not define.
    So a non-finite argument is not refused by the transport: it produces a
    frame the daemon must either reject whole -- losing the command with the
    call already reported as ``success`` -- or parse leniently, handing a
    non-finite target to the servos.

    Shared by all three movement RPCs so their accepted domain cannot diverge:
    :meth:`ReachyMiniDriver.look` alone carries six of these values, and before
    this helper the same unusable argument behaved differently depending on
    which one it landed in -- an ``inf`` angle raised ``ValueError: math domain
    error`` out of the RPC from ``math.cos``, while an ``inf`` offset was
    divided by 1000 and reported as a successful move.

    Two bounds, in this order. Finiteness and numeric-ness first, via the shared
    :func:`~strands_robots.utils.finite_number_error`; both signs and zero are
    legitimate there (a negative pitch looks down, zero re-centres). Then every
    parameter that names a bounded joint is held to the shared travel envelope,
    :func:`~strands_robots.tools.reachy.envelope_error`, through
    :data:`_ENVELOPE_AXIS_BY_PARAM`.

    That second bound used to be excused as the daemon's to enforce, on the
    grounds that it depends on hardware this library does not model. That was
    true when this helper was written and is no longer: the library models the
    envelope. :data:`~strands_robots.tools.reachy.MOTION_ENVELOPE_DEG` gives
    every bounded axis its travel in degrees, in a package whose own purpose
    statement is "what the *two* Reachy consumers must agree on and neither
    owns: the motion envelope", and which is importable with no Reachy and no
    daemon attached. The other of those two consumers,
    :meth:`~strands_robots.drivers.reachy.ReachyDriver.send_action`, already
    consults it. So a head pitch of 200 degrees on a +/-40 degree axis was
    refused by one driver and carried to the wire as 3.49 radians by the other,
    for the same physical robot, and this RPC reported ``success``.

    Finiteness is asked first for two reasons: an unusable value is then named
    by the caller's own parameter spelling rather than by the axis it maps to,
    and a travel comparison against ``nan`` is meaningless - ``abs(nan) <= 40``
    is ``False``, so an unordered value would be refused with a message about
    travel. That is the same ordering :func:`envelope_error` documents for its
    own checks.

    The head-body yaw coupling limit the envelope also carries is not reachable
    from here: it bounds ``head_yaw - body_yaw``, and this surface splits the
    pair across :meth:`ReachyMiniDriver.look` and :meth:`ReachyMiniDriver.body`,
    so neither RPC ever holds both. Per-axis travel is the half that transfers.

    That is a scope and not a delegation. The other half needs the head yaw the
    daemon is targeting while a lone ``body`` turn is carried out;
    :meth:`~strands_robots.drivers.reachy.ReachyDriver.send_action` has that,
    because it records the head pose it last sent, and so applies the coupling
    to a body-only turn as well as to a pair. This driver keeps no such record,
    so a ``body`` RPC here is per-axis only. Keeping no record is the invariant:
    were one added, the limit would become reachable and this exclusion would
    have to go with it.

    Args:
        rpc_name: The RPC that received the values, used as the message prefix.
        values: Parameter name to caller-supplied value, in signature order.

    Returns:
        The rejection dict for the first unusable value, or ``None`` when every
        value can be honored.
    """
    for param, value in values.items():
        if (message := finite_number_error(value, param, rpc_name)) is not None:
            logger.warning("Rejected Reachy Mini %s: %s", rpc_name, message)
            return {"status": "error", "reason": message}

    axis_by_param = _ENVELOPE_AXIS_BY_PARAM.get(rpc_name, {})
    bounded = {axis: values[param] for param, axis in axis_by_param.items() if param in values}
    if bounded and (message := envelope_error(bounded, rpc_name)) is not None:
        logger.warning("Rejected Reachy Mini %s: %s", rpc_name, message)
        return {"status": "error", "reason": message}
    return None


class ReachyMiniDriver(DeviceDriver):
    """Device Connect driver for Pollen Reachy Mini.

    Auto-detects Wireless (Zenoh) vs Lite (WebSocket) via the daemon's
    ``wireless_version`` flag. REST API calls work the same for both.
    """

    device_type = "reachy_mini"

    def __init__(
        self,
        host: str = "reachy-mini.local",
        prefix: str = "reachy_mini",
        api_port: int = 8000,
    ):
        """Configure the driver for a Reachy Mini reachable at ``host``.

        Args:
            host: Hostname or IP of the Reachy Mini daemon.
            prefix: Zenoh key prefix used by the Wireless variant. Must be a
                ``/``-joined sequence of mesh identifiers -- a Zenoh wildcard
                (``*`` / ``**``) would widen the command key to every Mini
                beneath the pattern. ``reachy_mini/robot_a`` is accepted;
                ``reachy_mini/*`` is not.
            api_port: TCP port the daemon serves its REST API and WebSocket on.
                Must name a port: an ``int`` in ``[1, 65535]``.

        Raises:
            ValueError: If ``api_port`` cannot address a TCP port, or if
                ``prefix`` cannot address a single robot's key expressions
                (see :func:`_key_prefix_error`).
        """
        # Refused here rather than at first use, and before any base-class state
        # is allocated. This port is interpolated verbatim into both the daemon
        # REST URL (``reachy_transport.api``) and the Lite WebSocket target, and
        # neither refuses it: ``api`` reports every failure as an ``{"error":
        # ...}`` result rather than raising, so an unusable port is reported as
        # an unreachable daemon - identically to a reachable port with the
        # daemon down. Naming the port is the only point a caller can act on.
        if (port_error := tcp_port_error(api_port, "api_port", type(self).__name__)) is not None:
            raise ValueError(port_error)
        # Refused alongside the port, and for the same reason: this value is
        # interpolated verbatim into the Wireless variant's three key
        # expressions, one of which actuates the robot. A wildcard is a valid
        # Zenoh key, so it is not refused downstream at all - it just widens
        # the command key to every Mini beneath the pattern.
        if (prefix_error := _key_prefix_error(prefix, "prefix", type(self).__name__)) is not None:
            raise ValueError(prefix_error)
        super().__init__()
        self._host = host
        self._prefix = prefix
        self._api_port = api_port
        self._latest_joints: dict[str, Any] | None = None
        self._latest_imu: dict[str, Any] | None = None
        self._hw: WebSocketLink | ZenohLink | None = None

    @property
    def identity(self) -> DeviceIdentity:
        """Static Device Connect identity for the Reachy Mini head.

        Returns a :class:`~device_connect_edge.types.DeviceIdentity` reporting
        ``device_type="reachy_mini"``, the ``Pollen Robotics`` manufacturer, and
        the configured host in the model string.
        """
        return DeviceIdentity(
            device_type="reachy_mini",
            manufacturer="Pollen Robotics",
            model=f"Reachy Mini @ {self._host}",
            description="Pollen Reachy Mini expressive robot head with antennas",
        )

    @property
    def status(self) -> DeviceStatus:
        """Availability of the Reachy Mini; always reports ``"idle"``.

        The head has no long-running task state, so it advertises itself as
        available for commands at all times.
        """
        return DeviceStatus(availability="idle")

    async def connect(self) -> None:
        """Connect to the Reachy Mini, auto-detecting Wireless vs Lite."""
        try:
            status = await asyncio.to_thread(api, self._host, self._api_port, "/api/daemon/status")
            is_lite = not status.get("wireless_version", True)
        except Exception:
            is_lite = False

        if is_lite:
            self._hw = WebSocketLink(self._host, self._api_port)
            logger.info("Connected to Reachy Mini Lite at %s (WebSocket)", self._host)
        else:
            self._hw = ZenohLink(self.transport, self._prefix)
            logger.info("Connected to Reachy Mini at %s (Zenoh)", self._host)

        await self._hw.start(
            on_joints=lambda d: setattr(self, "_latest_joints", d),
            on_imu=lambda d: setattr(self, "_latest_imu", d),
        )

    async def disconnect(self) -> None:
        """Tear down the hardware link and drop the handle to it.

        The handle is dropped before the stop is awaited because
        :meth:`_send_cmd` reads it as its "is the link connected?" test, and a
        link left in ``_hw`` keeps that guard unreachable. A movement RPC
        issued after a disconnect is then not refused: on the Wireless variant
        :meth:`ZenohLink.send_cmd` publishes to ``<prefix>/command`` and the
        RPC reports success, actuating the head after the driver was told to
        let go of it; on the Lite variant it reaches a closed socket instead.

        Clearing first also holds when the stop itself fails - the link is
        being torn down either way, so the driver must stop treating it as
        connected.
        """
        hw, self._hw = self._hw, None
        if hw:
            await hw.stop()

    # ── Helpers ────────────────────────────────────────────────

    async def _send_cmd(self, cmd: dict[str, Any]) -> None:
        """Send a real-time command via the active hardware link."""
        if self._hw is None:
            raise RuntimeError("Reachy Mini hardware link not connected")
        await self._hw.send_cmd(cmd)

    # ── Movement RPCs (Zenoh via transport) ────────────────────

    @rpc()
    async def look(
        self,
        pitch: float = 0,
        roll: float = 0,
        yaw: float = 0,
        x: float = 0,
        y: float = 0,
        z: float = 0,
    ) -> dict[str, Any]:
        """Set head pose instantly.

        Args:
            pitch: Pitch angle in degrees
            roll: Roll angle in degrees
            yaw: Yaw angle in degrees
            x: X offset in mm
            y: Y offset in mm
            z: Z offset in mm. Every value must be a finite number of
                either sign, and ``pitch`` / ``roll`` / ``yaw`` must be inside
                the shared travel envelope. The millimetre offsets carry no
                envelope bound and are the daemon's to enforce.

        Returns:
            ``{"status": "success", ...}``, or a ``{"status": "error",
            "reason": ...}`` dict naming the first argument that cannot be
            carried to the robot.
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "look")
        if (
            rejection := _motion_domain_error(
                "look", {"pitch": pitch, "roll": roll, "yaw": yaw, "x": x, "y": y, "z": z}
            )
        ) is not None:
            return rejection
        await self._send_cmd({"head_pose": rpy_to_pose(pitch, roll, yaw, x, y, z)})
        return {"status": "success", "pitch": pitch, "roll": roll, "yaw": yaw}

    @rpc()
    async def antennas(self, left: float = 0, right: float = 0) -> dict[str, Any]:
        """Set antenna angles.

        Args:
            left: Left antenna angle in degrees
            right: Right antenna angle in degrees. Both must be finite
                numbers of either sign.

        Returns:
            ``{"status": "success", ...}``, or a ``{"status": "error",
            "reason": ...}`` dict naming the argument that was refused.
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "antennas")
        if (rejection := _motion_domain_error("antennas", {"left": left, "right": right})) is not None:
            return rejection
        await self._send_cmd({"antennas_joint_positions": [math.radians(left), math.radians(right)]})
        return {"status": "success", "left": left, "right": right}

    @rpc()
    async def body(self, yaw: float = 0) -> dict[str, Any]:
        """Set body yaw angle.

        Args:
            yaw: Body yaw angle in degrees. Must be a finite number of
                either sign and inside the shared travel envelope.

        Returns:
            ``{"status": "success", ...}``, or a ``{"status": "error",
            "reason": ...}`` dict naming the argument that was refused.
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "body")
        if (rejection := _motion_domain_error("body", {"yaw": yaw})) is not None:
            return rejection
        await self._send_cmd({"body_yaw": math.radians(yaw)})
        return {"status": "success", "yaw": yaw}

    # ── Sensor RPCs (cached from transport subscription) ───────

    @rpc()
    async def getJoints(self) -> dict[str, Any]:
        """Get current joint positions (head + antennas)."""
        d = self._latest_joints
        if d is not None:
            head = d.get("head_joint_positions", [])
            ant = d.get("antennas_joint_positions", [])
            return {
                "status": "success",
                "head": [math.degrees(j) for j in head],
                "antennas": [math.degrees(j) for j in ant],
            }
        return {"status": "error", "reason": "no joint data"}

    @rpc()
    async def getImu(self) -> dict[str, Any]:
        """Get IMU data (accelerometer, gyroscope, quaternion, temperature)."""
        d = self._latest_imu
        if d is not None:
            return {
                "status": "success",
                "accelerometer": d.get("accelerometer"),
                "gyroscope": d.get("gyroscope"),
                "quaternion": d.get("quaternion"),
                "temperature": d.get("temperature"),
            }
        return {"status": "error", "reason": "no IMU data"}

    # ── Motor RPCs (Zenoh via transport) ───────────────────────

    @rpc()
    async def enableMotors(self, motor_ids: str = "") -> dict[str, Any]:
        """Enable motors (torque on).

        Args:
            motor_ids: Comma-separated motor IDs (empty = all)
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "enableMotors")
        ids = [s.strip() for s in motor_ids.split(",") if s.strip()] or None
        await self._send_cmd({"torque": True, "ids": ids})
        return {"status": "success", "enabled": motor_ids or "all"}

    async def _disable_motors_impl(self, motor_ids: str = "") -> dict[str, Any]:
        """Disable motors / torque off (un-gated core; callers enforce authz).

        The emergency-stop handler invokes this directly so the ``@rpc()``
        rpc-scope gate -- which sees a ``None`` caller in event-handler context
        and would fail-closed under ``DEVICE_CONNECT_RPC_ALLOW`` -- cannot
        silently deny the torque-off and leave motors live. ``_send_cmd``
        raises when the hardware link is down, so a failed torque-off surfaces
        to the caller rather than reporting a false ack.

        Args:
            motor_ids: Comma-separated motor IDs (empty = all).
        """
        ids = [s.strip() for s in motor_ids.split(",") if s.strip()] or None
        await self._send_cmd({"torque": False, "ids": ids})
        return {"status": "success", "disabled": motor_ids or "all"}

    @rpc()
    async def disableMotors(self, motor_ids: str = "") -> dict[str, Any]:
        """Disable motors (torque off).

        Args:
            motor_ids: Comma-separated motor IDs (empty = all)
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "disableMotors")
        return await self._disable_motors_impl(motor_ids)

    # ── Move RPCs (REST) ──────────────────────────────────────

    @rpc()
    async def playMove(self, move_name: str, library: str = "emotions") -> dict[str, Any]:
        """Play a recorded move from the HuggingFace library.

        Args:
            move_name: Name of the move to play
            library: Move library (emotions or dance)
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "playMove")
        if not _MOVE_NAME_RE.fullmatch(move_name or ""):
            return {"status": "error", "reason": f"invalid move_name: {move_name!r}"}
        ds = f"pollen-robotics/reachy-mini-{'emotions' if library == 'emotions' else 'dances'}-library"
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            f"/api/move/play/recorded-move-dataset/{ds}/{move_name}",
            "POST",
        )
        return {"status": "success", "move": move_name, "result": result}

    @rpc()
    async def listMoves(self, library: str = "emotions") -> dict[str, Any]:
        """List available recorded moves.

        Args:
            library: Move library (emotions or dance)
        """
        ds = f"pollen-robotics/reachy-mini-{'emotions' if library == 'emotions' else 'dances'}-library"
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            f"/api/move/recorded-move-datasets/list/{ds}",
        )
        return {"status": "success", "moves": result}

    # ── Expression RPCs (Zenoh animations via transport) ───────

    @rpc()
    async def nod(self) -> dict[str, Any]:
        """Nod the head (yes gesture)."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "nod")
        for _ in range(3):
            await self._send_cmd({"head_pose": rpy_to_pose(15, 0, 0)})
            await asyncio.sleep(0.25)
            await self._send_cmd({"head_pose": rpy_to_pose(-10, 0, 0)})
            await asyncio.sleep(0.25)
        await self._send_cmd({"head_pose": identity_pose()})
        return {"status": "success", "expression": "nod"}

    @rpc()
    async def shake(self) -> dict[str, Any]:
        """Shake the head (no gesture)."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "shake")
        for _ in range(3):
            await self._send_cmd({"head_pose": rpy_to_pose(0, 0, 25)})
            await asyncio.sleep(0.2)
            await self._send_cmd({"head_pose": rpy_to_pose(0, 0, -25)})
            await asyncio.sleep(0.2)
        await self._send_cmd({"head_pose": identity_pose()})
        return {"status": "success", "expression": "shake"}

    @rpc()
    async def happy(self) -> dict[str, Any]:
        """Happy antenna wiggle expression."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "happy")
        for _ in range(4):
            await self._send_cmd({"antennas_joint_positions": [math.radians(60), math.radians(-60)]})
            await asyncio.sleep(0.2)
            await self._send_cmd({"antennas_joint_positions": [math.radians(-60), math.radians(60)]})
            await asyncio.sleep(0.2)
        await self._send_cmd({"antennas_joint_positions": [0, 0]})
        return {"status": "success", "expression": "happy"}

    # ── Lifecycle RPCs (REST) ─────────────────────────────────

    @rpc()
    async def wakeUp(self) -> dict[str, Any]:
        """Wake up the robot (enable motors + play wake animation)."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "wakeUp")
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            "/api/move/play/wake_up",
            "POST",
        )
        return {"status": "success", "result": result}

    @rpc()
    async def sleep(self) -> dict[str, Any]:
        """Put robot to sleep (play sleep animation + disable motors)."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "sleep")
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            "/api/move/play/goto_sleep",
            "POST",
        )
        return {"status": "success", "result": result}

    async def _stop_motion_impl(self) -> dict[str, Any]:
        """Stop all current motion (un-gated core; callers enforce authz).

        The emergency-stop handler invokes this directly so the ``@rpc()``
        rpc-scope gate (``None`` caller in event context) cannot fail-closed
        and drop the stop.

        Surfaces daemon transport failure: :func:`reachy_transport.api`
        returns ``{"error": ...}`` on any HTTP/connection failure WITHOUT
        raising, so a stop issued against a down daemon would otherwise report
        ``status="success"``. Raise instead so the caller sees a real failure.
        """
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            "/api/move/stop",
            "POST",
        )
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"stopMotion transport failure: {result['error']}")
        return {"status": "success", "result": result}

    @rpc()
    async def stopMotion(self) -> dict[str, Any]:
        """Stop all current motion."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "stopMotion")
        return await self._stop_motion_impl()

    @rpc()
    async def getDaemonStatus(self) -> dict[str, Any]:
        """Report the daemon's status payload under this driver's own verdict.

        The payload's keys are merged into the envelope so a caller reads
        ``motors_on`` / ``freq`` at the top level, and ``status`` is re-asserted
        afterwards. Merged last, a daemon reply carrying a ``status`` field of
        its own replaced this envelope's verdict: a healthy call answered
        ``status="idle"``, and a daemon reporting its own fault answered
        ``status="error"`` for an RPC that reached it and succeeded. The
        presence path resolves the same collision the same way, by spreading
        the foreign mapping first so the locally decided keys win - see
        :meth:`strands_robots.mesh.sensors.SensorLoopsMixin._stamp_local_keys`.

        A daemon that was not reached is reported rather than merged.
        :func:`~strands_robots.device_connect.reachy_transport.api` answers
        every HTTP and connection failure with ``{"error": ...}`` instead of
        raising, so the unreachable daemon this RPC exists to detect came back
        as ``status="success"`` with the reason merged in beside it. The native
        driver reads this same endpoint and refuses that shape - see
        :meth:`strands_robots.drivers.reachy.ReachyDriver.connect_eagerly`.
        The error envelope is used rather than the ``RuntimeError``
        :meth:`_stop_motion_impl` raises, because that one guards a stop, where
        a caller acting on a false success stops nothing; this one answers a
        question, and its callers already branch on ``status``.

        A body that decodes to something other than a JSON object is reported
        for the same reason: spreading it raises ``TypeError`` out of a method
        whose whole contract is the envelope.

        Returns:
            ``{**payload, "status": "success"}`` when the daemon answered with
            a JSON object, otherwise ``{"status": "error", "reason": ...}``
            naming the daemon that was not reached or the body that could not
            be merged.
        """
        result = await asyncio.to_thread(
            api,
            self._host,
            self._api_port,
            "/api/daemon/status",
        )
        if not isinstance(result, dict):
            return {
                "status": "error",
                "reason": f"daemon status is not a JSON object: {type(result).__name__}",
            }
        if (error := result.get("error")) is not None:
            return {
                "status": "error",
                "reason": f"daemon unreachable ({self._host}:{self._api_port}): {error}",
            }
        return {**result, "status": "success"}

    # ── Events ────────────────────────────────────────────────

    @emit()
    async def emergencyStop(self, reason: str = "") -> None:
        """Emitted when this device triggers an emergency stop.

        Args:
            reason: Why the emergency stop was triggered
        """
        pass

    @on(event_name="emergencyStop")
    async def onEmergencyStop(self, device_id: str, event_name: str, payload: dict[str, Any]) -> None:
        """React to emergencyStop from an authorized safety controller.

        Security hardening: only act on emergency-stop events whose source is
        in the emergency-stop allowlist, so a spoofed event from an arbitrary
        device cannot interrupt operations.
        """
        if not is_authorized_caller(device_id, scope="estop"):
            logger.warning("Ignoring emergencyStop from unauthorized source %s", device_id)
            return
        logger.warning("Emergency stop received from %s - disabling motors", device_id)
        # Call the UN-GATED impls directly. The ``@rpc()``-decorated
        # ``stopMotion`` / ``disableMotors`` re-check ``get_rpc_source_device()``,
        # which is ``None`` in an event-handler context; with
        # ``DEVICE_CONNECT_RPC_ALLOW`` set that gate fail-closes and the returned
        # ``authz_error`` dicts were discarded, so the motors stayed live. This
        # handler is already authorized above on ``scope="estop"``, so bypassing
        # the rpc-scope gate here is correct.
        #
        # Attempt BOTH stop actions even if one fails (a safety handler must not
        # skip torque-off because stopMotion's REST call errored) and surface
        # any failure loudly instead of masking it behind a false ack. Torque
        # off runs first so the definitive motor kill lands even if the REST
        # stop hangs or errors.
        failures: list[str] = []
        try:
            await self._disable_motors_impl()
        # Recovery path: catch broadly. Hardware links raise transport-specific
        # exceptions outside (RuntimeError, OSError) -- e.g. the Lite variant's
        # WebSocketLink raises websockets.exceptions.ConnectionClosed
        # (WebSocketException -> Exception) and the Zenoh variant raises its own
        # publish errors. A safety handler must attempt BOTH stops regardless of
        # the failing link type, so record and continue rather than crash out.
        except Exception as exc:  # noqa: BLE001 - attempt-both e-stop recovery
            failures.append(f"disableMotors: {exc}")
        try:
            await self._stop_motion_impl()
        except Exception as exc:  # noqa: BLE001 - attempt-both e-stop recovery
            failures.append(f"stopMotion: {exc}")
        if failures:
            logger.critical(
                "Emergency stop from %s did NOT fully complete: %s",
                device_id,
                "; ".join(failures),
            )
