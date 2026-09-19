"""EarthRover Mini Plus native driver.

The Earth Rover (FrodoBots) is a mobile outdoor base reached over HTTP: the
vendor's `earth-rovers-sdk <https://github.com/frodobots-org/earth-rovers-sdk>`_
runs on the host (default ``http://localhost:8000``), proxies commands to the
rover over WebRTC/RTM, and exposes four endpoints this driver speaks:

* ``POST /control`` - one twist frame, ``{"command": {"linear", "angular",
  "lamp"}}``, each axis normalised to ``[-1, 1]``.
* ``GET /data`` - the telemetry snapshot: battery, GPS, orientation, IMU,
  wheel RPMs, signal level, lamp state.
* ``GET /v2/front`` / ``GET /v2/rear`` - one camera frame, base64 in the
  ``{camera}_frame`` field.
* ``POST /speak`` - text out of the rover's speaker.

The whole of that is the *agent's* surface too: :attr:`EarthRoverDriver.tool_spec`
declares one verb per capability (``sensors``, ``status``, ``camera``, ``move``,
``lamp``, ``speak``, ``stop``) and :meth:`EarthRoverDriver.stream` dispatches
each to the method that owns its judgement. A ``Robot("earthrover",
mode="real", driver="strands")`` handle therefore goes straight into
``Agent(tools=[rover])`` and the model can drive the rover it can see.

``requests`` is the transport, declared by the ``[earthrover]`` extra
(``pip install 'strands-robots[earthrover]'``, a member of ``[all]``). It is
imported lazily so the module loads and registers without it; a real connection
needs it and a running SDK, and :meth:`EarthRoverDriver.connect_eagerly` reports
the absent extra rather than raising.

Safety note: a rover is VELOCITY-commanded - unlike an arm, it does not hold
still when you stop talking to it, and whether the firmware times a twist out
on its own is not documented by the vendor. Until that answer exists,
:meth:`EarthRoverDriver.cleanup` sends a best-effort zero twist before closing
the session, so a clean teardown is always a STOP - the same reasoning as
feetech's torque-off loop, and errors are swallowed for the same reason: a
dead link must not block the close, and a rover behind a dead link cannot
hear a stop anyway.

Turn-direction note: the SDK/hardware already matches the ``+angular = left``
convention, so no inversion is applied by default. A rover observed turning
the wrong way is corrected with ``turn_sign=-1`` at construction rather than
an environment variable, so the correction is visible at the call site that
needed it.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import TYPE_CHECKING, Any, cast

from strands_robots.drivers.base import halt_failure_detail, telemetry_float, undeclared_verb_error
from strands_robots.utils import boolean_flag_error, finite_number_error, positive_finite_number_error

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping

    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The robots this driver registers for (read by ``_SHIPPED_DRIVERS``).
SUPPORTED_ROBOTS: tuple[str, ...] = ("earthrover",)

#: The whole control surface of a differential-drive base with a headlamp.
DRIVE_CHANNELS: tuple[str, ...] = ("linear", "angular", "lamp")

#: Magnitude bound on ``linear`` and ``angular``. The SDK's ``/control``
#: endpoint takes each axis normalised, so this is the whole envelope: ``1.0``
#: is already full speed and there is no faster value to ask for.
DRIVE_AXIS_LIMIT: float = 1.0

#: The camera views the SDK serves under ``/v2/<view>``.
CAMERA_VIEWS: tuple[str, ...] = ("front", "rear")

#: The longest twist one :meth:`EarthRoverDriver.move` call may hold before
#: the forced stop. An agent tool call is a conversation turn, not a control
#: loop: a longer run belongs to repeated calls, where telemetry is read
#: between legs rather than committing a velocity-commanded base to half a
#: minute blind.
MAX_MOVE_DURATION_S: float = 30.0

#: How often a timed :meth:`EarthRoverDriver.move` re-sends its twist while it
#: holds. The SDK's dead-man watchdog (``CONTROL_WATCHDOG_S``, default 3 s)
#: stops the rover once no fresh command has been confirmed for that long, so a
#: twist sent once and left alone drives for 3 s however long the hold is; the
#: SDK README asks moving clients to stream at about 10 Hz.
MOVE_REFRESH_PERIOD_S: float = 0.1

#: The latitude the SDK reports when the rover has no GPS fix. It is a real
#: number on the wire, so a summary that printed it would read as a position
#: off the coast of Antarctica rather than as no fix.
NO_FIX_LATITUDE: float = 1000.0

#: Where the vendor's SDK listens when started as documented: its Dockerfile
#: binds ``0.0.0.0:8000`` and every README example is ``localhost:8000``.
DEFAULT_SDK_URL = "http://localhost:8000"

#: The schemes this driver speaks. Compared case-insensitively, because a URI
#: scheme is case-insensitive (RFC 3986 section 3.1) and ``requests`` honours
#: that: ``HTTP://host:8001/data`` is fetched exactly like ``http://``.
ACCEPTED_SCHEMES: tuple[str, ...] = ("http", "https")


def _declared_scheme(value: str) -> str | None:
    """The scheme ``value`` declares, lowercased, or ``None`` when it declares none.

    The single owner of "does this value already carry a scheme?", so
    :func:`base_url_error` and the constructor's normalisation cannot disagree
    about where the scheme ends. They did while this was a literal
    ``startswith(("http://", "https://"))`` in both places: the check was
    case-sensitive, so ``HTTP://localhost:8001`` declared no scheme as far as
    the normaliser was concerned and was prefixed into
    ``http://HTTP://localhost:8001``.
    """
    scheme, separator, _ = value.partition("://")
    return scheme.lower() if separator else None


def _refuse(reason: str) -> dict[str, Any]:
    """One refusal envelope, so every refusal has the same shape."""
    return {"status": "error", "content": [{"text": reason}]}


def drive_axis_error(value: object, param: str, context: str) -> str | None:
    """Report why ``value`` is not a commandable drive axis, or ``None``.

    Refuses rather than clamping, which is the disposition the module docstring's
    safety note argues for: a rover is **velocity**-commanded, so a twist it was
    never asked for keeps being executed until the next command arrives. Clamping
    maps every out-of-range magnitude onto full speed, so the most common way to
    get one of these numbers wrong - writing it on the wrong scale - produces the
    fastest motion the base has, indefinitely, and reports success. A caller on a
    nought-to-a-hundred percent model then cannot tell its slowest crawl from its
    top speed: ``1`` and ``100`` are the same wire command once both saturate.

    This is :func:`~strands_robots.drivers.crazyflie.twist_error`'s reasoning at
    a lower speed, and the same rule
    :mod:`~strands_robots.drivers.feetech.bus` states for a joint target. It is
    the opposite disposition to
    :func:`~strands_robots.drivers.robotiq.protocol.aperture_mm_to_counts`, which
    clamps and says why: that axis is a bounded *position*, so the endpoint of
    the stroke really is what "200 mm on an 85 mm gripper" meant. A velocity has
    no such endpoint to land on.

    Args:
        value: The axis value as supplied.
        param: Which axis, to quote in the reason.
        context: Calling surface to quote in the reason.

    Returns:
        A reason naming the axis and the bound it broke, or ``None`` when the
        value is a finite magnitude inside the envelope.
    """
    if (reason := finite_number_error(value, param, context)) is not None:
        return reason
    magnitude = float(cast("float", value))
    if abs(magnitude) > DRIVE_AXIS_LIMIT:
        return (
            f"{context}: {param}={magnitude} is outside the normalised drive envelope "
            f"[-{DRIVE_AXIS_LIMIT}, {DRIVE_AXIS_LIMIT}] - the SDK's /control endpoint takes "
            f"a fraction of full speed, so {DRIVE_AXIS_LIMIT} is already the fastest value "
            "there is. A value on a percent or SI scale is the usual cause; divide it down "
            "rather than letting it saturate, because a rover holds the twist it was given."
        )
    return None


def base_url_error(value: object, param: str, context: str) -> str | None:
    """Report why ``value`` is not an SDK base URL, or ``None`` if it is one.

    ``port=`` is polymorphic across drivers - a serial path on the arms, an IP
    on the DDS robots, a URL here - so the wrong *shape* is refused at the
    chokepoint with a sentence naming the shape that belongs elsewhere, not by
    ``requests`` failing with "No host supplied" one call later.

    Beyond the shape, the base URL has to address *the host the caller wrote*.
    Every endpoint below is built from it, so a value whose authority names one
    host and dials another sends ``POST /control`` - a drive command - to a
    rover the caller never named, and ``connect_eagerly`` reports success when
    anything answers there. Two spellings do that, and neither is refused by
    the transport, which reports only the host it ended up with:

    * **Userinfo.** Everything before an ``@`` in the authority is credentials,
      so ``bot.local@10.0.0.9:8001`` dials ``10.0.0.9`` while the address still
      reads as ``bot.local`` - including in ``get_status``, which reports the
      base URL back verbatim.
    * **A foreign scheme.** ``ws://10.0.0.9:8001`` has no ``http`` prefix to
      recognise, so it is prefixed into ``http://ws://10.0.0.9:8001``, whose
      authority is ``ws:``. The request goes to the host ``ws`` on port 80 and
      the port the caller wrote is discarded - the same way the host half of a
      dialled address discards a validated port in
      :func:`~strands_robots.utils.dial_host_error`.

    That shared domain is deliberately *not* used here. It grades a bare host
    destined for interpolation into ``ws://{host}:{port}``, where an IPv6
    literal needs its brackets; the host inside a base URL has already been
    parsed, so ``http://[::1]:8001`` presents as ``::1`` and would be refused
    as "not a bare hostname or IP" - a legitimate base URL rejected.

    Everything else unusable here is left to the transport, which already names
    it: ``http:`` and ``//host`` raise ``InvalidURL: No host supplied``, and an
    out-of-range port or an embedded space raises ``InvalidURL: Failed to
    parse``. Both are ``ValueError`` subclasses, so ``connect_eagerly`` converts
    them into its own reason.

    Args:
        value: The candidate base URL.
        param: Parameter name to quote in the reason.
        context: Calling surface to quote in the reason.

    Returns:
        A reason, or ``None`` when ``value`` is a usable http(s) base or a
        bare ``host:port`` that can be prefixed into one.
    """
    if not isinstance(value, str) or not value.strip():
        return f"{context}: {param} must be the SDK base URL like {DEFAULT_SDK_URL!r}, got {value!r}"
    if value.startswith("/"):
        return (
            f"{context}: {param} is an HTTP base like {DEFAULT_SDK_URL!r}, got a filesystem "
            f"path {value!r} (that shape belongs to the serial arms or microduck's robotd socket)"
        )
    scheme = _declared_scheme(value)
    if scheme is not None and scheme not in ACCEPTED_SCHEMES:
        return (
            f"{context}: {param} must be an {' or '.join(ACCEPTED_SCHEMES)} base URL like "
            f"{DEFAULT_SDK_URL!r}, got the {scheme!r} address {value!r}. The SDK is plain HTTP, "
            f"and this is not refused by the transport: {scheme!r} becomes the host, so the "
            f"request goes to {scheme!r} on port 80 and the port above is discarded"
        )
    authority = (value.partition("://")[2] if scheme else value).partition("/")[0]
    if "@" in authority:
        named, _, dialled = authority.rpartition("@")
        return (
            f"{context}: {param} must name the SDK host directly, got {value!r}. Everything "
            f"before the '@' is userinfo, so {named!r} is not the host: every request - including "
            f"POST /control, a drive command - goes to {dialled!r} while the address still reads "
            f"as {named!r}"
        )
    return None


def telemetry_summary(data: Mapping[str, Any]) -> str:
    """One line naming what the rover just said, for a reader who wants no JSON.

    The ``sensors`` verb answers with this *and* the whole snapshot, so a
    caller selects a rendering by reading the block it wants rather than by
    passing a format flag.

    Coordinates go through :func:`~strands_robots.drivers.base.telemetry_float`
    rather than straight into the format string, because an agent verb must not
    raise past its dispatcher: a latitude the SDK reported as a string formats
    with ``ValueError`` under ``:.6f``, which would lose the battery and signal
    readings in the same snapshot. A coordinate that is no reading is reported
    as no fix, which is what it means - including a non-finite one, which
    :func:`~strands_robots.drivers.base.telemetry_float` passes through because a
    ``NaN`` on a *published* telemetry field is a reading the consumer must see
    for what it is, while ``f"{float('nan'):.6f}"`` renders here as the position
    ``nan`` rather than as no position at all.

    The headlamp is a flag, so it is read as one rather than for truthiness -
    the same disposition :meth:`EarthRoverDriver.send_action` applies to the
    ``lamp`` it writes. The SDK carries the field as the ``1``/``0`` that
    command puts on the wire, so those two integers and the two booleans are
    the readings; anything else is no reading. Read for truth, a snapshot whose
    firmware no longer carries ``lamp`` reported the headlamp *off*, and one
    that spelled it ``"off"`` reported it *on* - a lamp state this function
    decided rather than one the rover said, on the one field an operator
    driving at night reads to know whether the light is burning.

    Args:
        data: A ``/data`` snapshot, as :meth:`EarthRoverDriver.read_state`
            returns it.

    Returns:
        The summary line. Every field the snapshot does not carry reads
        ``?`` rather than being invented.
    """
    latitude = telemetry_float(data.get("latitude"))
    longitude = telemetry_float(data.get("longitude"))
    plotted = all(value is not None and math.isfinite(value) for value in (latitude, longitude))
    has_fix = bool(data.get("gps_signal")) and plotted and latitude != NO_FIX_LATITUDE
    gps = f"{latitude:.6f}, {longitude:.6f}" if has_fix else "no fix"
    lamp = data.get("lamp")
    lamp_state = "on" if lamp is True or lamp == 1 else "off" if lamp is False or lamp == 0 else "?"
    return (
        f"battery {data.get('battery', '?')}% | signal {data.get('signal_level', '?')}/4 | "
        f"heading {data.get('orientation', '?')} deg | speed {data.get('speed', '?')} | "
        f"lamp {lamp_state} | GPS {gps}"
    )


def detect_image_format(data: bytes) -> str:
    """Name the image format from magic bytes; the SDK may emit png/jpeg/webp."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "jpeg"


class EarthRoverDriver:
    """Drive an EarthRover Mini+ through the vendor's local SDK.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver`
    structurally - a Protocol - so no import from
    :mod:`strands_robots.drivers.base` is needed; the surface check
    :func:`~strands_robots.drivers.register_native_driver` runs at
    registration time is what pins the contract.
    """

    def __init__(
        self,
        tool_name: str = "earthrover",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        timeout_s: float = 10.0,
        turn_sign: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` does the network work.

        Args:
            tool_name: Name the agent invokes the driver by, and the mesh peer
                id when the driver is wrapped by
                :class:`~strands_robots.mesh.Mesh`.
            cameras: Accepted for parity with the lerobot driver; unused,
                because the rover's cameras are served by the SDK's ``/v2``
                endpoints, not v4l2.
            data_config: Accepted for parity; unused.
            port: The SDK base URL (``http://host:8001``) or a bare
                ``host:port`` to prefix. ``None`` selects
                :data:`DEFAULT_SDK_URL`. Kept polymorphic per the factory
                contract.
            timeout_s: Per-request timeout, seconds.
            turn_sign: ``1.0`` or ``-1.0`` - multiplied into every commanded
                ``angular``, for a rover whose physical turn direction is
                observed reversed. See the module docstring.
            **kwargs: Ignored; accepted so the factory can forward extras.

        Raises:
            ValueError: If ``port`` is not URL-shaped, ``timeout_s`` is not a
                positive finite number, or ``turn_sign`` is not ``±1.0``.
        """
        del cameras, data_config  # accepted for parity; unused here
        if kwargs:
            logger.debug("EarthRoverDriver ignoring extra kwargs: %s", sorted(kwargs))
        base = port or DEFAULT_SDK_URL
        if reason := base_url_error(base, "port", type(self).__name__):
            raise ValueError(reason)
        if _declared_scheme(base) is None:
            base = "http://" + base
        if reason := positive_finite_number_error(timeout_s, "timeout_s", type(self).__name__):
            raise ValueError(reason)
        if turn_sign not in (1.0, -1.0):
            raise ValueError(
                f"{type(self).__name__}: turn_sign flips the commanded turn direction, "
                f"so it must be 1.0 or -1.0, got {turn_sign!r}"
            )

        self._tool_name = tool_name
        self._base = base.rstrip("/")
        self._timeout = float(timeout_s)
        self._turn_sign = float(turn_sign)

        self._session: Any | None = None
        self._connected = False
        self._connect_error: str | None = None

        self._cache_lock = threading.Lock()
        self._last_data: dict[str, Any] | None = None
        self._last_command: dict[str, float] | None = None

    # ------------------------------------------------------------------ #
    # Agent tool surface.                                                #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """The name the Strands agent invokes this driver by."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Always ``"robot"`` - mirrors every other driver."""
        return "robot"

    @property
    def is_connected(self) -> bool:
        """Whether the HTTP session is open and the SDK answered.

        Derived from the leaf (the session) as well as the flag, so a torn-down
        driver cannot report live: :meth:`cleanup` drops the session, and a
        driver with no session is not connected no matter what a flag says.
        """
        return self._connected and self._session is not None

    @property
    def tool_spec(self) -> ToolSpec:
        """Everything an agent may ask of the rover: three reads, three writes, a halt.

        This object *is* the agent's tool - ``Robot("earthrover", mode="real",
        driver="strands")`` returns it and a caller passes it straight to
        ``Agent(tools=[rover])`` - so the verbs an agent has are the ones this
        enum declares and no others. Every capability the SDK exposes is
        therefore declared here rather than left to a wrapper: a rover an agent
        can read telemetry from but not drive, light, look through or speak
        with is not the robot the vendor shipped.

        Declaring the write verbs follows
        :class:`~strands_robots.drivers.feetech.driver.FeetechDriver`, which
        declares ``move_to`` and ``set_torque``, and
        :class:`~strands_robots.drivers.robotiq.driver.RobotiqDriver`, which
        declares ``open`` and ``close``. What keeps that safe is not withholding
        the verb but the driver's own judgement on the write path: an axis
        outside the normalised envelope is refused by name rather than clamped
        onto full speed (:func:`drive_axis_error`), ``lamp`` is read as a
        boolean rather than for truthiness, and a held twist is bounded by
        :data:`MAX_MOVE_DURATION_S` with the trailing stop reported.
        """
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "EarthRover Mini+ native driver, over the vendor's local SDK. Drives the "
                    "base with a normalised twist, switches the headlamp, grabs a front or rear "
                    "camera frame, speaks, reads telemetry (battery, GPS, orientation, IMU, "
                    "wheel RPMs) and halts. It is a velocity-commanded base: it keeps the last "
                    "twist until another arrives, so a leg either carries duration_s or ends "
                    "with stop."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "sensors: the latest telemetry snapshot, summarised; "
                                    "status: connection and the last commanded twist; "
                                    "camera: one frame from the front or rear view; "
                                    "move: drive one twist (linear, angular), optionally held for duration_s; "
                                    "lamp: switch the headlamp (on) - the rover stops, because the "
                                    "lamp rides the twist frame; "
                                    "speak: say text through the rover's speaker; "
                                    "stop: command a zero twist"
                                ),
                                "enum": ["sensors", "status", "camera", "move", "lamp", "speak", "stop"],
                                "default": "sensors",
                            },
                            "linear": {
                                "type": "number",
                                "description": (
                                    f"move only: forward speed, -{DRIVE_AXIS_LIMIT} to {DRIVE_AXIS_LIMIT} as a "
                                    "fraction of full speed; negative is reverse. A magnitude past that is "
                                    "refused by name, never clamped."
                                ),
                            },
                            "angular": {
                                "type": "number",
                                "description": (
                                    f"move only: turn rate, -{DRIVE_AXIS_LIMIT} to {DRIVE_AXIS_LIMIT}; positive "
                                    "is left. Refused past that, as linear is."
                                ),
                            },
                            "duration_s": {
                                "type": "number",
                                "description": (
                                    f"move only: hold the twist this long, at most {MAX_MOVE_DURATION_S}s, then "
                                    "stop; the answer reports both halves. Omitted, the twist is sent and the "
                                    "rover keeps rolling until the next command."
                                ),
                            },
                            "on": {
                                "type": "boolean",
                                "description": "lamp only: true switches the headlamp on, false off.",
                            },
                            "camera": {
                                "type": "string",
                                "description": "camera only: which view to grab.",
                                "enum": list(CAMERA_VIEWS),
                                "default": CAMERA_VIEWS[0],
                            },
                            "text": {
                                "type": "string",
                                "description": "speak only: what to say; must be non-empty.",
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

        Each branch delegates to the method that already owns the judgement, so
        the agent path and a Python caller are refused by the same sentence.
        """
        del kwargs, invocation_state  # forward-compat only
        tool_use_id = tool_use.get("toolUseId", "")
        request = tool_use.get("input") or {}
        action = request.get("action", "sensors")
        envelope: dict[str, Any]
        if action == "sensors":
            envelope = self._state_envelope()
        elif action == "status":
            envelope = await self.get_status()
        elif action == "camera":
            envelope = self._frame_envelope(request.get("camera", CAMERA_VIEWS[0]))
        elif action == "move":
            envelope = self.move(
                request.get("linear", 0.0),
                request.get("angular", 0.0),
                duration_s=request.get("duration_s"),
            )
        elif action == "lamp":
            envelope = self.set_lamp(request.get("on", True))
        elif action == "speak":
            envelope = self.speak(request.get("text", ""))
        elif action == "stop":
            envelope = self.stop_task()
        else:
            envelope = undeclared_verb_error(self, action)
        yield {"toolUseId": tool_use_id, **envelope}

    def _state_envelope(self) -> dict[str, Any]:
        """The ``sensors`` verb: the freshest snapshot, summarised, or a refusal.

        An empty snapshot is refused rather than published as an empty object.
        ``{}`` is what :meth:`read_state` returns before the rover has ever
        answered, and answering ``status="success"`` with it reads as a rover
        that reported nothing rather than one that was never heard from - so a
        caller retries the drive instead of starting the SDK.

        Returns:
            The summary line and the whole snapshot, or a refusal naming the
            remedy.
        """
        data = self.read_state()
        if not data:
            return _refuse(
                f"sensors: no telemetry yet - nothing has answered GET {self._base}/data. "
                "Is the earth-rovers-sdk running, with the rover connected to it?"
            )
        return {"status": "success", "content": [{"text": telemetry_summary(data)}, {"json": data}]}

    def _frame_envelope(self, camera: Any) -> dict[str, Any]:
        """The ``camera`` verb: one frame as a block the model can see.

        :meth:`capture_frame` answers with the frame base64-encoded, because
        the mesh publishes that envelope as JSON. A model cannot see a base64
        string, so the agent surface decodes it into an image block - the one
        place the two consumers of a frame differ.

        Args:
            camera: Which view, quoted back verbatim by
                :meth:`capture_frame` when it is not one of
                :data:`CAMERA_VIEWS`.

        Returns:
            A success envelope carrying the view name and the image, or
            :meth:`capture_frame`'s refusal unreshaped.
        """
        result = self.capture_frame(camera)
        if result["status"] != "success":
            return result
        import base64  # noqa: PLC0415 - stdlib, used only on this path

        payload = result["content"][0]["json"]
        return {
            "status": "success",
            "content": [
                {"text": f"[{payload['camera']}]"},
                {"image": {"format": payload["format"], "source": {"bytes": base64.b64decode(payload["b64"])}}},
            ],
        }

    # ------------------------------------------------------------------ #
    # Lifecycle.                                                         #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Open the session and prove the SDK answers ``/data``.

        Returns ``None`` on success. Off hardware - no ``requests``, no SDK
        process, or an SDK whose rover is not connected - returns a reason and
        leaves the driver usable: every read returns its empty cache and every
        write refuses "not connected". Idempotent.
        """
        if self.is_connected:
            return None
        try:
            import requests  # noqa: PLC0415 - lazy: the module must load without it
        except ImportError as exc:
            self._connect_error = (
                f"cannot import requests ({exc}); the EarthRover native driver speaks HTTP to the "
                "earth-rovers-sdk. Install it with: pip install 'strands-robots[earthrover]'"
            )
            return self._connect_error

        session = requests.Session()
        try:
            resp = session.get(f"{self._base}/data", timeout=self._timeout)
            data = resp.json() if resp.status_code == 200 else None
        except (OSError, ValueError) as exc:
            session.close()
            self._connect_error = (
                f"the earth-rovers-sdk did not answer GET {self._base}/data: {exc}. "
                "Is the SDK running? See https://github.com/frodobots-org/earth-rovers-sdk"
            )
            return self._connect_error
        if not isinstance(data, dict):
            session.close()
            self._connect_error = (
                f"GET {self._base}/data answered HTTP {resp.status_code} without a telemetry "
                "object - the SDK is up but the rover is not connected to it"
            )
            return self._connect_error

        with self._cache_lock:
            self._last_data = data
        self._session = session
        self._connected = True
        self._connect_error = None
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report reachability and what the rover last said and was told."""
        with self._cache_lock:
            data = dict(self._last_data or {})
            command = dict(self._last_command or {})
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        "sdk_url": self._base,
                        "battery_pct": data.get("battery"),
                        "signal_level": data.get("signal_level"),
                        "orientation": data.get("orientation"),
                        "latitude": data.get("latitude"),
                        "longitude": data.get("longitude"),
                        "lamp": data.get("lamp"),
                        "last_command": command or None,
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Command a zero twist, leaving the rover connected. Never raises.

        Annotated ``-> None`` by the driver protocol, so it carries no verdict:
        a caller that needs the halt outcome reads :meth:`stop_task`, which
        decides one. A zero twist that did not reach the SDK is logged, because
        a velocity-commanded base holds its last command until another one
        arrives - so an unsent halt leaves the rover driving.
        """
        if (detail := halt_failure_detail(self.stop_task())) is not None:
            logger.error(
                "%s.stop(): the zero twist did not reach the rover, which may still be "
                "driving at the commanded velocity: %s",
                self._tool_name,
                detail,
            )

    def cleanup(self) -> None:
        """Stop the wheels, then release the session. Idempotent.

        Closing an HTTP session does not stop a velocity-commanded base, so a
        best-effort zero twist goes out first - see the module docstring for
        why its errors are swallowed.
        """
        if self._session is not None:
            try:
                self._session.post(
                    f"{self._base}/control",
                    json={"command": {"linear": 0.0, "angular": 0.0}},
                    timeout=self._timeout,
                )
            except Exception:  # noqa: BLE001 - best effort by design
                logger.debug("%s: the parting zero twist did not send", self._tool_name, exc_info=True)
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                logger.debug("%s: session close failed during cleanup", self._tool_name, exc_info=True)
        self._session = None
        self._connected = False

    # ------------------------------------------------------------------ #
    # Write path.                                                        #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Command one twist frame.

        Args:
            action: Values for :data:`DRIVE_CHANNELS` - ``linear`` and
                ``angular`` inside ``[-1, 1]``, plus an optional ``lamp``
                boolean. An axis outside the envelope is refused by
                :func:`drive_axis_error` rather than clamped onto full speed,
                and ``lamp`` is read as a boolean rather than for truthiness, so
                ``"off"`` cannot switch the headlamp on. An absent axis is
                commanded ``0.0``, because a twist is a complete statement of
                intent: "turn" also means "stop driving forward".
            robot_name: Accepted for parity; this driver fronts one rover.

        Returns:
            A success envelope naming the commanded twist as sent (the caller's
            axes, with ``turn_sign`` applied), or a refusal naming what was
            wrong. Nothing reaches the SDK on a refusal.
        """
        session = self._session
        if session is None or not self._connected:
            suffix = f" ({self._connect_error})" if self._connect_error else ""
            return _refuse(f"send_action: not connected - call connect_eagerly() first{suffix}")
        bad = sorted(set(action) - set(DRIVE_CHANNELS))
        if bad:
            return _refuse(f"send_action: unknown drive channel(s) {bad}; valid: {list(DRIVE_CHANNELS)}")
        for axis in ("linear", "angular"):
            if axis in action and (reason := drive_axis_error(action[axis], axis, "send_action")):
                return _refuse(reason)
        if "lamp" in action and (reason := boolean_flag_error(action["lamp"], "lamp", "send_action")):
            return _refuse(reason)

        command: dict[str, float] = {
            "linear": float(action.get("linear", 0.0)),
            "angular": self._turn_sign * float(action.get("angular", 0.0)),
        }
        if "lamp" in action:
            command["lamp"] = 1 if action["lamp"] else 0
        try:
            resp = session.post(f"{self._base}/control", json={"command": command}, timeout=self._timeout)
        except OSError as exc:
            return _refuse(f"send_action: POST {self._base}/control did not reach the SDK: {exc}")
        if resp.status_code != 200:
            return _refuse(f"send_action: /control answered HTTP {resp.status_code}: {resp.text[:200]}")

        with self._cache_lock:
            self._last_command = command
        return {
            "status": "success",
            "content": [
                {"json": {"driver": "earthrover", "robot": robot_name or self._tool_name, "commanded": command}}
            ],
        }

    def move(self, linear: float = 0.0, angular: float = 0.0, duration_s: float | None = None) -> dict[str, Any]:
        """Command a twist by axis, optionally held for a bounded time then stopped.

        With ``duration_s`` unset this is sugar over :meth:`send_action`: one
        twist goes out and the rover keeps rolling until the next command or
        until the SDK's dead-man watchdog stops it, which is the SDK's own
        contract. With it set the twist is re-sent every
        :data:`MOVE_REFRESH_PERIOD_S` for that long, so the watchdog never
        fires mid-hold, and a zero twist follows; the answer reports **both**
        halves - a
        move whose trailing stop did not reach the SDK is not a completed move,
        because the rover is still rolling, so that outcome is an error naming
        the stop.

        Args:
            linear: Forward speed, inside ``[-1, 1]``; outside it is refused.
            angular: Turn rate, inside ``[-1, 1]``, positive left; outside it is
                refused.
            duration_s: How long to hold the twist before the forced stop, in
                seconds - at most :data:`MAX_MOVE_DURATION_S`. ``None`` sends
                the twist and returns immediately.

        Returns:
            :meth:`send_action`'s envelope for an untimed twist; for a timed
            move, an envelope reporting the twist, the time held and whether
            the trailing stop landed.
        """
        if duration_s is not None:
            if reason := positive_finite_number_error(duration_s, "duration_s", "move"):
                return _refuse(reason)
            if float(duration_s) > MAX_MOVE_DURATION_S:
                return _refuse(
                    f"move: duration_s is at most {MAX_MOVE_DURATION_S}s, got {duration_s}. A longer "
                    "run belongs to repeated calls, where telemetry is read between legs."
                )
        moved = self.send_action({"linear": linear, "angular": angular})
        if duration_s is None or moved["status"] != "success":
            return moved
        deadline = time.monotonic() + float(duration_s)
        while (remaining := deadline - time.monotonic()) > 0:
            time.sleep(min(MOVE_REFRESH_PERIOD_S, remaining))
            if remaining <= MOVE_REFRESH_PERIOD_S:
                break
            if self.send_action({"linear": linear, "angular": angular})["status"] != "success":
                break
        stopped = self.stop_task()
        outcome = {
            "commanded": moved["content"][0]["json"].get("commanded"),
            "held_s": float(duration_s),
            "stopped": stopped["status"] == "success",
        }
        if stopped["status"] != "success":
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "move: the twist was sent but the trailing stop did not reach the SDK - "
                            "the rover may still be rolling"
                        )
                    },
                    {"json": outcome},
                ],
            }
        return {"status": "success", "content": [{"json": outcome}]}

    def set_lamp(self, on: bool) -> dict[str, Any]:
        """Switch the headlamp - and stop, because the lamp rides the twist frame.

        The SDK carries ``lamp`` inside the one ``/control`` command, so a lamp
        write *is* a twist write: this sends ``{linear: 0, angular: 0, lamp}``.
        A rover that must keep moving with the lamp on is driven with
        :meth:`move` afterwards.

        Args:
            on: ``True`` for lamp on, ``False`` for off. Read as a boolean by
                :meth:`send_action` rather than for truthiness, so ``"off"``
                cannot switch the headlamp on.

        Returns:
            :meth:`send_action`'s envelope for the zero-twist-plus-lamp command.
        """
        return self.send_action({"linear": 0.0, "angular": 0.0, "lamp": on})

    def speak(self, text: str) -> dict[str, Any]:
        """Say ``text`` through the rover's speaker.

        Args:
            text: What to say.

        Returns:
            A success envelope, or a refusal naming what was wrong.
        """
        if not isinstance(text, str) or not text.strip():
            return _refuse(f"speak: text must be a non-empty string, got {text!r}")
        session = self._session
        if session is None or not self._connected:
            return _refuse("speak: not connected - call connect_eagerly() first")
        try:
            resp = session.post(f"{self._base}/speak", json={"text": text}, timeout=self._timeout)
        except OSError as exc:
            return _refuse(f"speak: POST {self._base}/speak did not reach the SDK: {exc}")
        if resp.status_code != 200:
            return _refuse(f"speak: /speak answered HTTP {resp.status_code}: {resp.text[:200]}")
        return {"status": "success", "content": [{"json": {"spoke": text}}]}

    # ------------------------------------------------------------------ #
    # Task paths.                                                        #
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
        """Refuse: no policy provider is wired to the rover yet."""
        del instruction, policy_port, policy_host, policy_provider, duration, policy_kwargs
        return _refuse(
            "start_task: no policy provider is wired to the earthrover yet. A caller with a "
            "built policy drives it by calling send_action on their own timer"
        )

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refuse a host-driven rollout; this driver ships the transport only."""
        del policy_object, instruction, duration, n_steps
        return _refuse(
            "run_policy: this driver sends one twist per call and owns no control loop. "
            'Call send_action on your own timer, or use mode="sim" for a host-driven rollout'
        )

    def get_task_status(self) -> dict[str, Any]:
        """Report the last commanded twist, the only task state this driver holds."""
        with self._cache_lock:
            command = dict(self._last_command or {})
        return {
            "status": "success",
            "content": [{"json": {"running": False, "last_command": command or None}}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Command a zero twist - the rover's halt.

        Returns:
            :meth:`send_action`'s envelope for the zero twist, so the caller
            learns whether the stop actually reached the SDK rather than being
            told a flag was cleared.
        """
        if not self.is_connected:
            return _refuse("stop_task: not connected")
        return self.send_action({"linear": 0.0, "angular": 0.0})

    # ------------------------------------------------------------------ #
    # Read path.                                                         #
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict[str, float]:
        """Joint positions by name - a wheeled base has none.

        Returns:
            ``{}`` always: the rover reports pose and battery through
            :meth:`read_state`, and publishing wheel RPMs as "joints" would
            put velocities where every consumer expects positions.
        """
        return {}

    def read_state(self) -> dict[str, Any]:
        """The freshest ``/data`` snapshot the driver can get.

        Polls the SDK when connected and falls back to the cached snapshot
        when the poll fails, so a caller always sees the last truth the rover
        told rather than an exception - the mesh publishes from here.

        Returns:
            The telemetry dict, or ``{}`` before the first successful read.
        """
        session = self._session
        if session is not None and self._connected:
            try:
                resp = session.get(f"{self._base}/data", timeout=self._timeout)
                data = resp.json() if resp.status_code == 200 else None
            except (OSError, ValueError) as exc:
                logger.debug("%s: /data poll failed: %s", self._tool_name, exc)
                data = None
            if isinstance(data, dict):
                with self._cache_lock:
                    self._last_data = data
        with self._cache_lock:
            return dict(self._last_data or {})

    def capture_frame(self, camera: str = "front") -> dict[str, Any]:
        """Grab one camera frame from the SDK.

        Args:
            camera: One of :data:`CAMERA_VIEWS`.

        Returns:
            A success envelope carrying ``{"camera", "format", "b64"}``, or a
            refusal naming what was wrong - including an SDK that answered
            without a frame, which is a rover with its video session down.
        """
        if camera not in CAMERA_VIEWS:
            return _refuse(f"capture_frame: camera must be one of {list(CAMERA_VIEWS)}, got {camera!r}")
        session = self._session
        if session is None or not self._connected:
            return _refuse("capture_frame: not connected - call connect_eagerly() first")
        try:
            resp = session.get(f"{self._base}/v2/{camera}", timeout=self._timeout)
        except OSError as exc:
            return _refuse(f"capture_frame: GET {self._base}/v2/{camera} did not reach the SDK: {exc}")
        if resp.status_code != 200:
            return _refuse(f"capture_frame: /v2/{camera} answered HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            b64 = (resp.json() or {}).get(f"{camera}_frame")
        except ValueError:
            b64 = None
        if not b64:
            return _refuse(
                f"capture_frame: the SDK answered /v2/{camera} without a {camera}_frame - "
                "the rover's video session is not up"
            )
        import base64  # noqa: PLC0415 - stdlib, used only on this path

        try:
            raw = base64.b64decode(b64)
        except (ValueError, TypeError) as exc:
            return _refuse(f"capture_frame: /v2/{camera} frame is not base64: {exc}")
        return {
            "status": "success",
            "content": [{"json": {"camera": camera, "format": detect_image_format(raw), "b64": b64}}],
        }
