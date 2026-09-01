"""Native CRTP driver for the Bitcraze Crazyflie 2.x nano-quadcopter.

``Robot("crazyflie", mode="real", port="radio://0/80/2M/E7E7E7E7E7")`` builds one
of these. The instance satisfies
:class:`~strands_robots.drivers.base.HardwareDriver`, so
:func:`~strands_robots.robot.Robot` returns it and the mesh, the teleop rail and
the agent tool surface consume it exactly like the lerobot driver they replace
for this robot. There is nothing to replace in practice: lerobot has no robot
type for a Crazyflie, so before this driver ``mode="real"`` raised
``ValueError: Unsupported robot type: 'crazyflie'`` while the registry happily
served the MuJoCo asset for ``mode="sim"``.

Why a native driver: a 27-gram quadcopter is not a servo bus. It has no joints
to read or write, it is addressed over a Crazyradio dongle speaking CRTP rather
than over a serial servo protocol, and its command surface is a *setpoint
stream* rather than a position write. Three properties of that stream are the
whole reason this file exists, and each one is a way to crash the aircraft if a
caller has to guess:

**1. Angular velocity is radians here and degrees on the wire.** Every twist in
this package is SI - ``wz`` in rad/s, the same convention the mesh, the G1
driver and the simulation twists use. ``cflib``'s ``Commander`` takes ``yawrate``
in **degrees per second**. Handing 1.0 rad/s straight to ``cflib`` commands
1 deg/s: a yaw the operator reads as "not responding" at 1/57th of the requested
rate. :func:`twist_to_setpoint` is the single place that conversion happens and
:data:`RADIANS_TO_DEGREES` is the only factor in the file.

**2. A setpoint is not a command, it is a subscription.** The firmware
supervisor cuts thrust when the setpoint stream goes quiet, so a single
``send_hover_setpoint`` call does not produce sustained motion - it produces a
twitch. This driver therefore owns a background repeater
(:attr:`CrazyflieDriver.setpoint_hz`, default :data:`DEFAULT_SETPOINT_HZ`) that
re-sends the last accepted setpoint until a new one replaces it or the stream is
stopped. ``send_action`` returns once the setpoint is *latched*, not once the
motion is finished.

**3. Stopping and landing are different verbs, and confusing them breaks the
aircraft.** ``Commander.send_stop_setpoint`` cuts the motors: an airborne
Crazyflie falls. Descending under control is ``HighLevelCommander.land``. So
:meth:`~CrazyflieDriver.stop` - the contract's "stop motion, stay connected" -
**lands**, and cutting the motors is a separate, explicitly named
:meth:`~CrazyflieDriver.emergency_stop`. Handing the low-level stream back to
the high-level commander first is mandatory and easy to miss: ``cflib`` ships
``Commander.send_notify_setpoint_stop`` for exactly that, and a high-level
command issued without it is ignored while the low-level stream still owns the
setpoint priority. Both high-level verbs - :meth:`~CrazyflieDriver.takeoff` and
:meth:`~CrazyflieDriver.land` - perform that handover in order.

What the driver actually does:

* Opens the radio link in :meth:`~CrazyflieDriver.connect_eagerly` and **waits
  for it to come up**. ``cflib``'s ``open_link`` is asynchronous and swallows
  every error, so an absent dongle otherwise reports success and every later
  packet is silently discarded; the driver waits for ``connected`` or
  ``connection_failed`` (bounded by :data:`CONNECT_TIMEOUT_S`) and returns the
  reason. Only then does it arm the platform (firmware 2023.02 and later refuse
  to spin the motors until ``Platform.send_arming_request`` is called) and start
  one telemetry log block. Nothing is armed and no propeller turns until the
  link is up.
* Caches what the log block delivers. ``_pose`` is the onboard state estimate
  (``stateEstimate.x/y/z`` plus ``stabilizer.roll/pitch/yaw``), ``_imu`` is the
  attitude triple, ``_battery`` is ``pm.vbat`` in volts alongside the decoded
  ``pm.state``. The mesh reads all three with ``getattr(robot, name, None)``, so
  a Crazyflie that has not connected publishes no sensor topic and is otherwise
  complete.
* Refuses a setpoint outside :func:`twist_envelope` by name, before it reaches
  the radio.

Deliberately absent, so a reader is not left guessing:

* **No** ``_joints``. A quadcopter has four propellers and no joint the
  package's joint contract can name; publishing ``{}`` would put an empty
  reading on the mesh where "this robot has no joints" is the fact.
* **No** ``_lidar_*`` and no ranger deck. The Multi-ranger and Flow decks are
  real hardware this driver has not been measured against, and their log
  variables are absent on a bare Crazyflie - a log block naming them fails to
  add rather than degrading, taking the pose and battery reads down with it.
* ``start_task`` and ``run_policy`` refuse rather than standing in for work in
  progress. There is no trained flight policy in this package and no action
  space registered for an aerial robot, so a rollout would be a loop sending
  whatever a manipulation policy emitted to a flying machine.
* No position ``go_to``. ``HighLevelCommander.go_to`` needs a trustworthy
  absolute position estimate, which a bare Crazyflie without a Flow deck or a
  Loco/Lighthouse system does not have; its estimator drifts freely in x and y.
  Offering the verb would invite a caller to fly to a coordinate the aircraft
  cannot find.

Nothing here imports ``cflib`` at module load: every radio touch is inside a
method body reached through :func:`_resolve_cflib`, so the module imports on a
machine with no Crazyradio, on CI, and in every unit test against a fake link.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, Any, cast

from strands_robots.mesh.pacing import Ticker
from strands_robots.utils import (
    finite_number_error,
    positive_count_error,
    positive_finite_number_error,
)

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The robots this driver serves. One entry: the ``crazyflie`` registry name
#: (aliases ``cf2`` / ``bitcraze_crazyflie`` resolve to it, and
#: :func:`~strands_robots.drivers.registry.register_native_driver` resolves them,
#: so listing them here would be a second copy of the alias table).
#:
#: ``skydio_x2`` is in the registry under the same ``aerial`` category and is
#: deliberately **not** here: it is a different airframe behind a different SDK,
#: and advertising a robot this driver cannot fly is a promise it does not keep.
SUPPORTED_ROBOTS: tuple[str, ...] = ("crazyflie",)

#: The radio URI a Crazyflie ships with: channel 80, 2 Mbit, address
#: ``E7E7E7E7E7``. ``port=`` overrides it and stays polymorphic - a
#: ``radio://`` URI for a Crazyradio, or ``usb://0`` for the USB cable.
DEFAULT_URI: str = "radio://0/80/2M/E7E7E7E7E7"

#: Setpoint repeat rate, in Hz. The firmware supervisor cuts thrust when the
#: setpoint stream goes quiet, so a latched setpoint has to be re-sent; 20 Hz is
#: five times ``cflib``'s own examples' 10 Hz loop and well under the radio's
#: capacity, leaving headroom for the telemetry log block sharing the link.
DEFAULT_SETPOINT_HZ: int = 20

#: The one unit conversion in this module. ``wz`` is rad/s everywhere in this
#: package; ``cflib``'s ``Commander`` takes ``yawrate`` in deg/s.
RADIANS_TO_DEGREES: float = 180.0 / math.pi

#: How long :meth:`CrazyflieDriver.connect_eagerly` waits for the radio link
#: before reporting a reason, in seconds. ``Crazyflie.open_link`` is
#: asynchronous: it returns as soon as the request is queued and delivers the
#: outcome later on the link thread, so the driver has to wait for that outcome
#: or it cannot tell a flying aircraft from an absent dongle. Ten seconds covers
#: the radio handshake plus the log and parameter TOC downloads with a cold
#: cache. Bounded, unlike ``cflib``'s own ``SyncCrazyflie.open_link``, because an
#: agent blocked forever on a switched-off aircraft reports nothing at all.
CONNECT_TIMEOUT_S: float = 10.0

# --------------------------------------------------------------------------- #
# Flight envelope.                                                            #
#                                                                             #
# These are this driver's caps, not the SDK's: ``cflib`` imposes no ceiling on #
# a setpoint and the firmware will faithfully attempt whatever arrives. The    #
# scale is taken from Bitcraze's own ``MotionCommander`` defaults (0.2 m/s     #
# linear, 72 deg/s yaw), widened to leave a usable margin above them, and      #
# bounded for an indoor room rather than an open field. A caller who needs     #
# more is flying a mission this driver has not been measured against.          #
# --------------------------------------------------------------------------- #

#: Horizontal speed ceiling, m/s, applied to ``vx`` and ``vy`` independently.
MAX_HORIZONTAL_SPEED: float = 1.0

#: Vertical speed ceiling, m/s, for the ``vz`` of a world-velocity setpoint.
MAX_VERTICAL_SPEED: float = 0.5

#: Yaw-rate ceiling, rad/s (about 143 deg/s on the wire).
MAX_YAW_RATE: float = 2.5

#: Lowest commandable hover height, m. Below roughly a rotor diameter the
#: downwash reflects off the floor and the aircraft becomes unstable in ground
#: effect, so a hover setpoint under this is refused rather than attempted.
MIN_HEIGHT: float = 0.05

#: Highest commandable hover height, m - a low indoor ceiling.
MAX_HEIGHT: float = 2.0

#: Battery voltage, in volts, at which Bitcraze's power manager reports low
#: power on the 1S cell. Reported alongside ``_battery`` rather than acted on:
#: cutting a caller's flight off mid-command is a policy decision this driver
#: does not own, but a caller polling status needs the threshold to compare to.
LOW_BATTERY_VOLTS: float = 3.2

#: ``pm.state`` codes decoded to names, so a status payload reads as a state
#: rather than as an integer a caller has to look up. Matches the firmware's
#: ``pmStates`` enum order.
POWER_STATES: tuple[str, ...] = ("battery", "charging", "charged", "low_power", "shutdown")

#: The log variables one telemetry block subscribes to, as ``(name, ctype)``.
#: All are core variables present on a bare Crazyflie with no expansion deck -
#: a block naming an absent variable fails to add entirely, which would take
#: the pose and battery reads down with whichever deck-specific variable was
#: optimistically included.
LOG_VARIABLES: tuple[tuple[str, str], ...] = (
    ("stateEstimate.x", "float"),
    ("stateEstimate.y", "float"),
    ("stateEstimate.z", "float"),
    ("stabilizer.roll", "float"),
    ("stabilizer.pitch", "float"),
    ("stabilizer.yaw", "float"),
    ("pm.vbat", "float"),
    ("pm.state", "uint8_t"),
)

#: Name of the telemetry log block, and its period.
_LOG_NAME = "strands_state"
_LOG_PERIOD_MS = 100

#: Action keys :func:`action_to_setpoint` understands, for the refusal message
#: when an action names none of them.
ACTION_KEYS: tuple[str, ...] = ("vx", "vy", "vz", "wz", "z")

#: ``cflib`` ``Commander`` method names this driver calls. Named constants
#: because :func:`action_to_setpoint` returns one of them and the tests assert
#: on the returned name - the wire contract, not an implementation detail.
HOVER_SETPOINT = "send_hover_setpoint"
VELOCITY_SETPOINT = "send_velocity_world_setpoint"


def _refuse(reason: str) -> dict[str, Any]:
    """The driver's error envelope, one shape for every refusal path."""
    return {"status": "error", "content": [{"text": reason}]}


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    """The driver's success envelope, one shape for every accepted path."""
    return {"status": "success", "content": [{"json": payload}]}


def twist_envelope() -> dict[str, float]:
    """Report the flight envelope a setpoint is held to.

    The discovery surface behind :func:`twist_error`'s refusals: a caller whose
    setpoint was refused needs the bound it exceeded, and an agent planning a
    flight needs the bounds before it plans. Derived from the module constants,
    so the reported envelope and the enforced one cannot drift.

    Returns:
        Bound name -> value, in SI units (m/s, rad/s, m).
    """
    return {
        "max_horizontal_speed": MAX_HORIZONTAL_SPEED,
        "max_vertical_speed": MAX_VERTICAL_SPEED,
        "max_yaw_rate": MAX_YAW_RATE,
        "min_height": MIN_HEIGHT,
        "max_height": MAX_HEIGHT,
    }


def twist_error(
    vx: Any,
    vy: Any,
    wz: Any,
    vz: Any = 0.0,
    height: Any = None,
    context: str = "set_twist",
) -> str | None:
    """Report why this twist is outside the flight envelope, or ``None``.

    Refuses rather than clamping. A clamp on an aircraft is the worse failure:
    an operator who asked for 5 m/s and silently got 1 m/s plans the next
    command around a speed the vehicle never flew, and the discrepancy only
    surfaces as a position error metres later. Naming the bound at the door
    leaves the caller's model of the aircraft correct.

    Args:
        vx: Forward velocity, m/s in the body frame.
        vy: Left velocity, m/s in the body frame.
        wz: Yaw rate, **rad/s** (converted to deg/s on the wire).
        vz: Upward velocity, m/s. Only meaningful for a world-velocity
            setpoint; a hover setpoint holds ``height`` instead.
        height: Hover height, m, or ``None`` for a world-velocity setpoint.
        context: Calling surface to quote in the reason.

    Returns:
        A reason naming the offending parameter and the bound it broke, or
        ``None`` when every component is inside the envelope.
    """
    for name, value, limit in (
        ("vx", vx, MAX_HORIZONTAL_SPEED),
        ("vy", vy, MAX_HORIZONTAL_SPEED),
        ("vz", vz, MAX_VERTICAL_SPEED),
        ("wz", wz, MAX_YAW_RATE),
    ):
        if (reason := finite_number_error(value, name, context)) is not None:
            return reason
        if abs(float(value)) > limit:
            unit = "rad/s" if name == "wz" else "m/s"
            return (
                f"{context}: {name}={float(value)} {unit} exceeds the flight envelope "
                f"(|{name}| <= {limit} {unit}). See twist_envelope() for every bound."
            )

    if height is not None:
        if (reason := positive_finite_number_error(height, "z", context)) is not None:
            return reason
        if not MIN_HEIGHT <= float(height) <= MAX_HEIGHT:
            return (
                f"{context}: z={float(height)} m is outside the hover envelope "
                f"({MIN_HEIGHT} <= z <= {MAX_HEIGHT} m). Below {MIN_HEIGHT} m the aircraft is "
                "in ground effect. See twist_envelope() for every bound."
            )
    return None


def twist_to_setpoint(
    vx: float,
    vy: float,
    wz: float,
    height: float | None = None,
    vz: float = 0.0,
) -> tuple[str, tuple[float, float, float, float]]:
    """Turn an SI twist into the ``cflib`` commander call that carries it.

    Pure, so a test asserts the exact wire arguments without a radio. This is
    the only place ``wz`` becomes degrees, and the only place the choice between
    the two setpoint kinds is made:

    * ``height`` given -> ``send_hover_setpoint(vx, vy, yawrate, zdistance)``.
      The onboard controller holds that altitude, so a dropped packet costs
      horizontal drift and not height. The safe indoor default.
    * ``height`` omitted -> ``send_velocity_world_setpoint(vx, vy, vz, yawrate)``.
      Altitude becomes the integral of a commanded rate, which is only
      trustworthy with a working z estimate, and a caller who wants to climb has
      no other way to say so.

    Note the argument *order* differs between the two: the hover setpoint's yaw
    rate is third and its height fourth, while the world-velocity setpoint's
    yaw rate is fourth. Building the tuple here rather than at each call site is
    what stops a height being commanded as a yaw rate.

    Args:
        vx: Forward velocity, m/s.
        vy: Left velocity, m/s.
        wz: Yaw rate, rad/s.
        height: Hover height, m, or ``None`` for a world-velocity setpoint.
        vz: Upward velocity, m/s; ignored when ``height`` is given, because a
            hover setpoint's altitude is the height rather than a rate.

    Returns:
        ``(method_name, args)`` where ``method_name`` is
        :data:`HOVER_SETPOINT` or :data:`VELOCITY_SETPOINT` - the literal
        ``cflib`` ``Commander`` method - and ``args`` is its positional
        arguments with the yaw rate already in deg/s.
    """
    yawrate_deg = float(wz) * RADIANS_TO_DEGREES
    if height is not None:
        return HOVER_SETPOINT, (float(vx), float(vy), yawrate_deg, float(height))
    return VELOCITY_SETPOINT, (float(vx), float(vy), float(vz), yawrate_deg)


def action_to_setpoint(action: Mapping[str, Any]) -> tuple[str, tuple[float, float, float, float]] | str:
    """Translate an action dict into one commander call, or a refusal reason.

    Absent keys are zero, which is the resting twist rather than a guess: an
    action naming only ``wz`` yaws in place. An action naming none of
    :data:`ACTION_KEYS` is refused, because the alternative - sending an
    all-zero setpoint - would latch the aircraft into a hover the caller never
    asked for and report success for it.

    Args:
        action: Joint targets keyed as this driver names them; see
            :data:`ACTION_KEYS`.

    Returns:
        ``(method_name, args)`` from :func:`twist_to_setpoint`, or a reason
        string when the action is empty of known keys or outside the envelope.
    """
    if not any(key in action for key in ACTION_KEYS):
        return (
            f"send_action: no flight command in {sorted(action)}; expected at least one of "
            f"{list(ACTION_KEYS)} (vx/vy/vz in m/s, wz in rad/s, z the hover height in m)"
        )
    height = action.get("z")
    if (
        reason := twist_error(
            action.get("vx", 0.0),
            action.get("vy", 0.0),
            action.get("wz", 0.0),
            action.get("vz", 0.0),
            height,
            context="send_action",
        )
    ) is not None:
        return reason
    return twist_to_setpoint(
        float(action.get("vx", 0.0)),
        float(action.get("vy", 0.0)),
        float(action.get("wz", 0.0)),
        None if height is None else float(height),
        float(action.get("vz", 0.0)),
    )


def parse_log_data(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalise one telemetry callback payload into the driver's cached shapes.

    Pure, so a test parses a real log frame without a radio.

    Args:
        data: The ``data`` mapping a ``cflib`` ``LogConfig`` callback receives,
            keyed by the variable names in :data:`LOG_VARIABLES`.

    Returns:
        ``{"pose": ..., "imu": ..., "battery": ...}``. ``pose`` carries the
        estimated position and the attitude; ``imu`` carries the attitude alone
        (the Crazyflie fuses its IMU on board and publishes the result, not the
        raw rates); ``battery`` carries ``volts``, the decoded ``state``, and
        the ``low_volts`` threshold to compare against. A variable missing from
        the payload lands as ``None`` rather than as a zero, because a zero
        altitude and an unreported altitude are different facts.
    """
    read = data.get
    attitude = {
        "roll": read("stabilizer.roll"),
        "pitch": read("stabilizer.pitch"),
        "yaw": read("stabilizer.yaw"),
    }
    state_code = read("pm.state")
    return {
        "pose": {
            "x": read("stateEstimate.x"),
            "y": read("stateEstimate.y"),
            "z": read("stateEstimate.z"),
            **attitude,
        },
        "imu": dict(attitude),
        "battery": {
            "volts": read("pm.vbat"),
            "state": POWER_STATES[state_code]
            if isinstance(state_code, int) and state_code < len(POWER_STATES)
            else None,
            "low_volts": LOW_BATTERY_VOLTS,
        },
    }


def _resolve_cflib() -> Any:
    """Return the ``cflib`` pieces this driver needs, or a reason naming what failed.

    The same shape as :func:`strands_robots.drivers.reachy._resolve_transport`:
    the seam's other drivers resolve a lazy SDK import through a helper that
    hands back a reason string, and every refusal boundary turns that string
    into a named failure. Doing the same here keeps the no-raise contract intact
    when ``cflib`` is absent - :meth:`CrazyflieDriver.connect_eagerly` reports a
    reason and leaves the driver disconnected but usable, rather than raising
    ``ModuleNotFoundError`` through the agent tool surface.

    Returns:
        An object with ``crtp``, ``Crazyflie`` and ``LogConfig`` attributes, or
        a reason string naming the module and the remedy.
    """
    try:
        import importlib

        crtp = importlib.import_module("cflib.crtp")
        crazyflie_mod = importlib.import_module("cflib.crazyflie")
        log_mod = importlib.import_module("cflib.crazyflie.log")
    except ImportError as exc:
        return (
            f"cannot import cflib ({exc}); the Crazyflie driver needs the Bitcraze "
            "client library. Install it with: pip install 'strands-robots[crazyflie]'"
        )
    return type(
        "_CflibPieces",
        (),
        {"crtp": crtp, "Crazyflie": crazyflie_mod.Crazyflie, "LogConfig": log_mod.LogConfig},
    )


class _LinkOutcome:
    """Whichever of ``connected`` / ``connection_failed`` arrives first.

    ``Crazyflie.open_link`` never raises. It wraps its whole body in
    ``except Exception`` and routes every failure - no Crazyradio, a malformed
    URI, a link driver that will not load - to the ``connection_failed``
    callback, and it reports success by calling ``connected`` once the log and
    parameter TOCs are down. Both arrive on ``cflib``'s link thread *after*
    ``open_link`` has already returned, so a caller that reads the return value
    learns nothing. This latches the first outcome so
    :meth:`CrazyflieDriver.connect_eagerly` can wait on it.
    """

    def __init__(self) -> None:
        self.settled = threading.Event()
        self.failure: str | None = None

    def on_connected(self, link_uri: str) -> None:
        """``cf.connected``: the link is up and both TOCs are downloaded."""
        del link_uri
        self.settled.set()

    def on_failed(self, link_uri: str, message: str) -> None:
        """``cf.connection_failed``: the link never came up, and why.

        Only the first line is kept. ``cflib`` builds this message with
        ``traceback.format_exc()`` appended, so a missing dongle arrives as a
        20-line traceback whose first line - ``Couldn't load link driver: Cannot
        find a Crazyradio Dongle`` - is the whole actionable content. The rest is
        an internal stack that would be pasted verbatim into an agent-facing
        error envelope, so it goes to the log instead of the refusal.
        """
        del link_uri
        text = str(message)
        self.failure = text.strip().splitlines()[0] if text.strip() else text
        if text.strip() != self.failure:
            logger.debug("Crazyflie connection_failed detail: %s", text)
        self.settled.set()


class CrazyflieDriver:
    """Native CRTP driver for the Bitcraze Crazyflie 2.x.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver` structurally,
    so no import from that module is needed; the surface check
    :func:`~strands_robots.drivers.register_native_driver` runs at registration
    pins the contract.
    """

    def __init__(
        self,
        tool_name: str = "crazyflie",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        setpoint_hz: int = DEFAULT_SETPOINT_HZ,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` opens the radio.

        Args:
            tool_name: Name the agent invokes the driver by, and the mesh peer id.
            cameras: Accepted for parity with the other drivers and unused. The
                AI-deck's camera is reached over its own WiFi streamer rather
                than over CRTP, so it is not a camera this driver opens.
            data_config: Accepted for parity; unused.
            port: The link URI. Defaults to :data:`DEFAULT_URI`. A
                ``radio://<dongle>/<channel>/<rate>/<address>`` URI for a
                Crazyradio, or ``usb://0`` over the cable.
            setpoint_hz: Rate at which the latched setpoint is re-sent, in Hz.
                A positive integer: it divides into the repeater's sleep, and
                the firmware supervisor cuts thrust if the stream goes quiet.
            **kwargs: Ignored; accepted so the factory can forward extras.

        Raises:
            ValueError: If ``setpoint_hz`` is not a positive integer. Raised
                here rather than returned from :meth:`connect_eagerly`, which is
                declared ``-> str | None``: a rate the repeater cannot use is
                not a connection this driver can degrade to reporting. A
                ``nan`` or a zero would make ``1 / setpoint_hz`` a crash or an
                infinity inside a background thread, where nothing reports it.
        """
        del cameras, data_config
        if kwargs:
            logger.debug("CrazyflieDriver ignoring extra kwargs: %s", sorted(kwargs))
        if (reason := positive_count_error(setpoint_hz, "setpoint_hz", "CrazyflieDriver")) is not None:
            raise ValueError(reason)

        self._tool_name = tool_name
        self._uri = str(port) if port else DEFAULT_URI
        self._setpoint_hz = int(setpoint_hz)

        self._cache_lock = threading.Lock()
        self._pose: dict[str, Any] | None = None
        self._imu: dict[str, Any] | None = None
        self._battery: dict[str, Any] | None = None

        self._cf: Any | None = None
        self._log_config: Any | None = None
        self._connected = False
        self._connect_error: str | None = None
        self._armed = False

        # The latched setpoint the repeater re-sends, and the thread that does
        # it. ``None`` means no stream is running, which is distinct from a
        # zero setpoint (a commanded hover in place).
        self._setpoint: tuple[str, tuple[float, float, float, float]] | None = None
        self._repeater: threading.Thread | None = None
        self._repeater_stop = threading.Event()

    # ------------------------------------------------------------------ #
    # Agent tool surface.                                                #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """The name the Strands agent invokes this driver by."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Always ``"robot"`` - mirrors the other drivers."""
        return "robot"

    @property
    def is_connected(self) -> bool:
        """Whether the radio link is open, for the mesh presence read."""
        return self._connected and self._cf is not None

    @property
    def setpoint_hz(self) -> int:
        """Rate at which the latched setpoint is re-sent, in Hz."""
        return self._setpoint_hz

    @property
    def tool_spec(self) -> ToolSpec:
        """A minimal agent-facing spec: read sensors, report status, land."""
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Bitcraze Crazyflie native driver over CRTP. Reads the onboard state estimate, "
                    "attitude and battery, reports link status, and lands the aircraft. Flight "
                    "commands (vx/vy/vz in m/s, wz in rad/s, z the hover height) go through "
                    "send_action; takeoff and land are their own verbs."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "sensors: latest cached pose/attitude/battery; "
                                    "status: link, arming and flight envelope; "
                                    "land: descend under control and stop the setpoint stream"
                                ),
                                "enum": ["sensors", "status", "land"],
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
        """Handle one agent invocation and yield exactly one tool result.

        ``land`` is the agent-facing stop verb rather than ``stop``, because on
        this airframe the two words mean different things and the schema should
        say which one the agent gets: a controlled descent, never a motor cut.
        """
        del kwargs, invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        action = (tool_use.get("input") or {}).get("action", "sensors")
        if action == "sensors":
            envelope: dict[str, Any] = _ok(
                {
                    "pose": self._snapshot("_pose"),
                    "imu": self._snapshot("_imu"),
                    "battery": self._snapshot("_battery"),
                }
            )
        elif action == "status":
            envelope = await self.get_status()
        else:  # "land"
            # The sibling's envelope verbatim. Building one here could only
            # restate the intent ("asked it to land"), while the descent itself
            # can be refused - a disconnected link, an unusable duration - and
            # the agent needs that verdict rather than an acknowledgement.
            envelope = self.stop_task()
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Flight commands.                                                   #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Latch one flight setpoint and keep it alive.

        Returns once the setpoint is latched and the repeater is running - not
        once the motion has finished. A Crazyflie has no notion of "arrived":
        the setpoint is a rate the aircraft holds until something replaces it,
        so a caller that wants a bounded move sends a setpoint, waits, and sends
        the resting one.

        Args:
            action: Flight command keyed by :data:`ACTION_KEYS`.
            robot_name: Accepted for the contract and ignored - this driver
                fronts exactly one aircraft.

        Returns:
            A success envelope carrying the commander method and the wire
            arguments actually sent, or an error envelope naming the refusal.
        """
        del robot_name
        if not self.is_connected:
            return _refuse(f"send_action: {self._tool_name} is not connected ({self._connect_error or 'no link'})")
        translated = action_to_setpoint(action)
        if isinstance(translated, str):
            return _refuse(translated)
        if not self._armed:
            return _refuse(
                f"send_action: {self._tool_name} is not armed; the firmware refuses to spin the "
                "motors until arming succeeds. Reconnect to retry the arming request."
            )
        self._latch(translated)
        method, args = translated
        return _ok({"commanded": method, "args": list(args), "setpoint_hz": self._setpoint_hz})

    def set_twist(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, z: float | None = None) -> dict[str, Any]:
        """Command a body-frame twist, the mobile-robot verb spelled for the air.

        The named counterpart of :meth:`send_action` for the twist a mobile base
        exposes: the same three components, plus the one an aircraft needs that
        a ground robot does not. ``z`` given holds that altitude while the twist
        runs (a hover setpoint); ``z`` omitted leaves altitude to the
        world-velocity setpoint's own rate, which is zero here.

        Args:
            vx: Forward velocity, m/s.
            vy: Left velocity, m/s.
            wz: Yaw rate, **rad/s**. Converted to the deg/s the wire wants.
            z: Hover height in m, or ``None`` for a world-velocity setpoint.

        Returns:
            The envelope :meth:`send_action` returns.
        """
        action: dict[str, Any] = {"vx": vx, "vy": vy, "wz": wz}
        if z is not None:
            action["z"] = z
        return self.send_action(action)

    def takeoff(self, height: float = 0.5, duration: float = 2.0) -> dict[str, Any]:
        """Climb to ``height`` under the high-level commander, after the handover.

        The high-level commander runs the trajectory on board, so the climb
        survives a dropped packet in a way a streamed setpoint does not - which
        is what makes it the right verb for the one manoeuvre where losing the
        stream means falling out of a climb.

        It reaches the firmware through the same priority rule as
        :meth:`land`, so it performs the same handover first, in the same order:
        halt the setpoint repeater, ``Commander.send_notify_setpoint_stop()``,
        then the high-level command. While the low-level stream is running it
        owns the setpoint priority and a high-level command underneath it is
        ignored - and because the repeater re-sends at
        :attr:`setpoint_hz` the priority never decays on its own. Without the
        handover, a ``takeoff`` commanded during a hover returns success while
        the aircraft holds its current height, and the caller then plans every
        later command around an altitude the vehicle never climbed to.

        Args:
            height: Absolute target height, m; held to the hover envelope.
            duration: Time to take over the climb, s.

        Returns:
            A success envelope naming the height and duration, or an error
            envelope naming the refusal.
        """
        if (reason := twist_error(0.0, 0.0, 0.0, 0.0, height, context="takeoff")) is not None:
            return _refuse(reason)
        if (reason := positive_finite_number_error(duration, "duration", "takeoff")) is not None:
            return _refuse(reason)
        if not self.is_connected:
            return _refuse(f"takeoff: {self._tool_name} is not connected ({self._connect_error or 'no link'})")
        if not self._armed:
            return _refuse(f"takeoff: {self._tool_name} is not armed; reconnect to retry the arming request.")
        self._halt_repeater()
        self._commander().send_notify_setpoint_stop()
        self._high_level().takeoff(float(height), float(duration))
        return _ok({"commanded": "takeoff", "height": float(height), "duration": float(duration)})

    def land(self, duration: float = 2.0) -> dict[str, Any]:
        """Descend to the ground under control, in the order the SDK requires.

        Three steps, and the order is the point:

        1. Stop the setpoint repeater, so nothing re-latches a twist mid-descent.
        2. ``Commander.send_notify_setpoint_stop()`` - hand the setpoint
           priority back to the high-level commander. A ``land`` issued while
           the low-level stream still owns priority is ignored, and the aircraft
           keeps flying the last twist.
        3. ``HighLevelCommander.land(0.0, duration)``.

        This is what :meth:`stop` calls, and it is **not**
        :meth:`emergency_stop`: this descends, that cuts the motors.

        Args:
            duration: Time to take over the descent, s.

        Returns:
            A success envelope, or an error envelope naming the refusal.
        """
        if (reason := positive_finite_number_error(duration, "duration", "land")) is not None:
            return _refuse(reason)
        if not self.is_connected:
            return _refuse(f"land: {self._tool_name} is not connected ({self._connect_error or 'no link'})")
        self._halt_repeater()
        commander = self._commander()
        commander.send_notify_setpoint_stop()
        self._high_level().land(0.0, float(duration))
        return _ok({"commanded": "land", "duration": float(duration)})

    def emergency_stop(self) -> dict[str, Any]:
        """Cut the motors. **An airborne aircraft falls.**

        ``Commander.send_stop_setpoint`` sets every motor to zero immediately.
        That is the correct response to a propeller strike or a fly-away and the
        wrong response to "stop moving", which is why it is a separate verb from
        :meth:`stop` and why neither :meth:`stop` nor the agent tool schema can
        reach it. A caller has to name it.

        Returns:
            A success envelope, or an error envelope naming the refusal.
        """
        if not self.is_connected:
            return _refuse(f"emergency_stop: {self._tool_name} is not connected")
        self._halt_repeater()
        self._commander().send_stop_setpoint()
        return _ok({"commanded": "send_stop_setpoint", "note": "motors cut; an airborne aircraft falls"})

    # ------------------------------------------------------------------ #
    # Task and policy paths.                                             #
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
        """Refuse: no flight policy is registered for this airframe.

        Not deferred work with a shape to fill in later. A manipulation policy
        emits joint targets; this aircraft has no joints and four propellers,
        so there is nothing to map an emitted action onto. Naming that is more
        useful than a rollout loop that sends a gripper command to a quadcopter.
        """
        del instruction, policy_port, policy_host, policy_provider, duration, policy_kwargs
        return _refuse(
            "start_task: the Crazyflie has no policy action space in this package (no joints, four "
            "propellers, and no aerial policy provider). Fly it with send_action / set_twist / "
            "takeoff / land."
        )

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refuse a rollout, for the same reason as :meth:`start_task`."""
        del policy_object, instruction, duration, n_steps
        return _refuse(
            "run_policy: the Crazyflie has no policy action space in this package. "
            "Fly it with send_action / set_twist / takeoff / land."
        )

    def get_task_status(self) -> dict[str, Any]:
        """Report the latched setpoint, which is the only thing in flight.

        A caller polling task status on this driver is asking what the aircraft
        is currently doing, and the honest answer is the setpoint the repeater
        is holding - not the "no task" a refusal-only driver would report while
        the propellers are turning.

        Returns:
            A success envelope carrying whether a setpoint stream is running
            and, if so, the commander call it repeats.
        """
        with self._cache_lock:
            setpoint = self._setpoint
        return _ok(
            {
                "streaming": setpoint is not None,
                "commanded": None if setpoint is None else setpoint[0],
                "args": None if setpoint is None else list(setpoint[1]),
                "setpoint_hz": self._setpoint_hz,
            }
        )

    def stop_task(self) -> dict[str, Any]:
        """Stop the setpoint stream and land, then report what was stopped.

        Reads as "stop what start_task started", and since the only thing in
        flight is a setpoint stream, stopping it while leaving the aircraft
        airborne with no stream would be the fall this driver exists to avoid.
        So it lands - the same descent :meth:`stop` performs.
        """
        was_streaming = self._setpoint is not None
        envelope = self.land()
        if envelope["status"] != "success":
            return _refuse(f"stop_task: the descent was refused - {envelope['content'][0]['text']}")
        return _ok({"stopped": "setpoint_stream" if was_streaming else None, "commanded": "land"})

    # ------------------------------------------------------------------ #
    # Lifecycle and status.                                              #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Open the radio link, arm the platform and start telemetry.

        Returns a reason instead of raising, so a missing ``cflib``, an absent
        Crazyradio or an aircraft that is switched off leaves the driver
        constructed and reporting why rather than taking down the caller.

        Waiting for the link is the step a reader is most likely to be surprised
        by, and it is why this method is longer than a call to ``open_link``.
        ``Crazyflie.open_link`` is **asynchronous and never raises**: with no
        Crazyradio plugged in it returns normally, leaves ``cf.link`` at ``None``
        and calls ``connection_failed`` on its link thread. Treating that return
        as success would arm and fly an aircraft that is not there -
        ``Crazyflie.send_packet`` is a silent no-op while ``link`` is ``None``,
        so the arming request, every setpoint and every ``takeoff`` would be
        dropped on the floor while this driver answered ``success``. So the
        driver waits for ``connected`` or ``connection_failed``, bounded by
        :data:`CONNECT_TIMEOUT_S`, and returns the failure message as the reason.
        Waiting is also what makes telemetry work: ``connected`` fires only after
        the log TOC is downloaded, and ``log.add_config`` raises ``KeyError`` for
        every variable until it is. ``cflib``'s own
        ``SyncCrazyflie.open_link`` is the reference for this wait.

        The arming request is the other step worth reading twice: firmware
        2023.02 and later will not spin the motors until
        ``Platform.send_arming_request(True)`` succeeds, and a driver that
        skipped it would connect cleanly, accept every setpoint, and produce no
        motion at all. Older ``cflib`` releases have no ``platform`` attribute;
        that is reported as the reason rather than ignored, because an unarmed
        aircraft accepting flight commands is the failure mode the check exists
        to prevent. It runs after the link is up, so it cannot be the packet
        that races the connection setup.

        Returns:
            ``None`` on success, or a reason naming what failed.
        """
        if self.is_connected:
            return None
        pieces = _resolve_cflib()
        if isinstance(pieces, str):
            self._connect_error = pieces
            return pieces

        outcome = _LinkOutcome()
        try:
            pieces.crtp.init_drivers()
            cf = pieces.Crazyflie()
            cf.connected.add_callback(outcome.on_connected)
            cf.connection_failed.add_callback(outcome.on_failed)
            cf.open_link(self._uri)
        except (OSError, RuntimeError, AttributeError) as exc:
            reason = f"cannot open the Crazyflie link at {self._uri!r}: {exc}"
            self._connect_error = reason
            return reason

        if (link_failure := self._await_link(cf, outcome)) is not None:
            self._connect_error = link_failure
            return link_failure

        self._cf = cf
        self._connected = True
        self._connect_error = None

        if (arm_failure := self._arm(cf)) is not None:
            self._connect_error = arm_failure
            return arm_failure
        if (telemetry_failure := self._start_telemetry(cf, pieces.LogConfig)) is not None:
            # Telemetry is not load-bearing for flight: an aircraft with no log
            # block still flies, and refusing the whole connection over a log
            # variable would ground a usable vehicle. Reported, not fatal.
            logger.warning("Crazyflie telemetry unavailable: %s", telemetry_failure)
        return None

    def _await_link(self, cf: Any, outcome: _LinkOutcome) -> str | None:
        """Block until the link settles, releasing it on anything but success.

        Args:
            cf: The ``Crazyflie`` whose ``open_link`` is in flight.
            outcome: The latch the link thread reports into.

        Returns:
            ``None`` once the link is up, or a reason naming the refusal or the
            timeout. The link is closed before a reason is returned, so the
            dongle is free for the next attempt.
        """
        if not outcome.settled.wait(CONNECT_TIMEOUT_S):
            reason = (
                f"the Crazyflie at {self._uri!r} did not answer within {CONNECT_TIMEOUT_S:g}s; "
                "check the aircraft is switched on, charged and on the URI's radio channel"
            )
        elif outcome.failure is not None:
            reason = f"cannot open the Crazyflie link at {self._uri!r}: {outcome.failure}"
        else:
            return None
        try:
            cf.close_link()
        except (OSError, RuntimeError, AttributeError) as exc:
            # The reason being returned already names the real failure; a close
            # that also fails must not replace it with a tidier-looking one.
            logger.debug("closing the unopened Crazyflie link raised: %s", exc)
        return reason

    def _arm(self, cf: Any) -> str | None:
        """Send the arming request, or report why the motors will stay still."""
        platform = getattr(cf, "platform", None)
        if platform is None or not hasattr(platform, "send_arming_request"):
            return (
                "this cflib build has no Platform.send_arming_request, so the aircraft cannot be "
                "armed; firmware 2023.02 and later will not spin the motors. Upgrade cflib."
            )
        try:
            platform.send_arming_request(True)
        except (OSError, RuntimeError) as exc:
            return f"the arming request failed ({exc}); the firmware will not spin the motors"
        self._armed = True
        return None

    def _start_telemetry(self, cf: Any, log_config_cls: Any) -> str | None:
        """Add and start the one telemetry log block, or report why not."""
        try:
            block = log_config_cls(name=_LOG_NAME, period_in_ms=_LOG_PERIOD_MS)
            for name, ctype in LOG_VARIABLES:
                block.add_variable(name, ctype)
            cf.log.add_config(block)
            block.data_received_cb.add_callback(self._on_log_data)
            block.start()
        except (AttributeError, KeyError, RuntimeError) as exc:
            return f"cannot start the {_LOG_NAME!r} log block: {exc}"
        self._log_config = block
        return None

    def _on_log_data(self, timestamp: int, data: Mapping[str, Any], logconf: Any) -> None:
        """Cache one telemetry frame. Runs on ``cflib``'s link thread."""
        del timestamp, logconf
        parsed = parse_log_data(data)
        with self._cache_lock:
            self._pose = parsed["pose"]
            self._imu = parsed["imu"]
            self._battery = parsed["battery"]

    async def get_status(self) -> dict[str, Any]:
        """Report the link, the arming state, the flight envelope and telemetry.

        Returns:
            A success envelope; the shape the mesh publishes as this peer's
            presence and the agent's ``status`` verb returns.
        """
        return _ok(
            {
                "tool_name": self._tool_name,
                "tool_type": self.tool_type,
                "uri": self._uri,
                "connected": self.is_connected,
                "armed": self._armed,
                "connect_error": self._connect_error,
                "streaming": self._setpoint is not None,
                "setpoint_hz": self._setpoint_hz,
                "envelope": twist_envelope(),
                "pose": self._snapshot("_pose"),
                # Part of the triple every native driver's status carries
                # (``tool_name`` / ``connected`` / ``battery_pct``), and
                # structurally ``None`` on this airframe rather than merely
                # unread: the Crazyflie's power manager reports a 1S cell
                # VOLTAGE and a coarse state, not a percentage, and a LiPo
                # discharge curve is nowhere near linear - so deriving a percent
                # from ``pm.vbat`` would put a number on the mesh that nothing
                # measured. The measured reading is in ``battery`` beside it.
                "battery_pct": None,
                "battery": self._snapshot("_battery"),
                "supported_robots": list(SUPPORTED_ROBOTS),
            }
        )

    async def stop(self) -> None:
        """Bring the aircraft down and stop the setpoint stream, staying connected.

        The contract's wording is "stop motion", and on an airframe that cannot
        hold still without a setpoint stream the only motion-free state is on
        the ground. So this **lands**; it never cuts the motors. Cutting them is
        :meth:`emergency_stop`, which a caller has to name.
        """
        if self.is_connected:
            self.land()
        else:
            self._halt_repeater()

    def cleanup(self) -> None:
        """Land, stop the log block and close the radio link.

        Ordered so nothing is released while it is still being used: the
        descent first (it needs the link), then the telemetry block, then the
        link itself. Every step tolerates a half-built driver, because cleanup
        is what runs after a failed connect.
        """
        if self.is_connected:
            self.land()
        self._halt_repeater()
        block = self._log_config
        if block is not None:
            try:
                block.stop()
            except (AttributeError, RuntimeError) as exc:
                logger.debug("Crazyflie log block did not stop cleanly: %s", exc)
            self._log_config = None
        cf = self._cf
        if cf is not None:
            try:
                cf.close_link()
            except (AttributeError, OSError, RuntimeError) as exc:
                logger.debug("Crazyflie link did not close cleanly: %s", exc)
        self._cf = None
        self._connected = False
        self._armed = False

    # ------------------------------------------------------------------ #
    # Setpoint stream.                                                   #
    # ------------------------------------------------------------------ #

    def _latch(self, setpoint: tuple[str, tuple[float, float, float, float]]) -> None:
        """Record the setpoint and make sure the repeater is running."""
        with self._cache_lock:
            self._setpoint = setpoint
        self._send_setpoint(setpoint)
        if self._repeater is None or not self._repeater.is_alive():
            self._repeater_stop.clear()
            self._repeater = threading.Thread(
                target=self._repeat_loop,
                name=f"{self._tool_name}-setpoints",
                daemon=True,
            )
            self._repeater.start()

    def _repeat_loop(self) -> None:
        """Re-send the latched setpoint until asked to stop.

        Daemon thread. The firmware supervisor cuts thrust when the setpoint
        stream goes quiet, so this loop is what turns a single ``send_action``
        into sustained motion.

        Paced by :class:`~strands_robots.mesh.pacing.Ticker` rather than by
        ``self._repeater_stop.wait(period)``, for the reason that module records:
        a ``wait(period)`` is a delay where a rate needs a deadline, so the time
        the radio spends on a CRTP write is *added* to the period instead of
        subtracted from it, and the stream runs at ``1 / (period + write)``. This
        is the one loop in the driver where that matters most - its whole job is
        to keep the supervisor's watchdog fed, so a stream that quietly paces
        slow is a thrust cut with no error anywhere.

        A send that raises stops the loop rather than spinning on a dead link:
        the next ``send_action`` reports the refusal through an envelope a caller
        can read, which a background thread cannot.
        """
        with Ticker(1.0 / self._setpoint_hz, self._repeater_stop) as ticker:
            while not self._repeater_stop.is_set():
                if ticker.wait():  # True == stopped, same sense as Event.wait
                    return
                with self._cache_lock:
                    setpoint = self._setpoint
                if setpoint is None:
                    return
                try:
                    self._send_setpoint(setpoint)
                except (AttributeError, OSError, RuntimeError) as exc:
                    logger.warning("Crazyflie setpoint stream stopped: %s", exc)
                    return

    def _send_setpoint(self, setpoint: tuple[str, tuple[float, float, float, float]]) -> None:
        """Send one setpoint to the commander."""
        method, args = setpoint
        getattr(self._commander(), method)(*args)

    def _halt_repeater(self) -> None:
        """Stop the repeater thread and clear the latched setpoint.

        Setting the event is what ends the loop; the ticker breaks its wait into
        short slices so that is seen within a slice rather than within a period.
        """
        self._repeater_stop.set()
        repeater = self._repeater
        if repeater is not None and repeater.is_alive():
            repeater.join(timeout=1.0 + 1.0 / self._setpoint_hz)
        self._repeater = None
        with self._cache_lock:
            self._setpoint = None

    def _commander(self) -> Any:
        """The low-level commander of the open link."""
        return self._cf.commander  # type: ignore[union-attr]

    def _high_level(self) -> Any:
        """The high-level (onboard-trajectory) commander of the open link."""
        return self._cf.high_level_commander  # type: ignore[union-attr]

    def _snapshot(self, attr: str) -> Any:
        """Read one cached sensor attribute under the lock, copied.

        The link thread replaces these wholesale, so a reader that returned the
        live object could be handed a dict mid-replacement.
        """
        with self._cache_lock:
            value = getattr(self, attr)
        return dict(value) if isinstance(value, dict) else value
