"""EarthRover Mini Plus native driver.

The Earth Rover (FrodoBots) is a mobile outdoor base reached over HTTP: the
vendor's `earth-rovers-sdk <https://github.com/frodobots-org/earth-rovers-sdk>`_
runs on the host (default ``http://localhost:8001``), proxies commands to the
rover over WebRTC/RTM, and exposes four endpoints this driver speaks:

* ``POST /control`` - one twist frame, ``{"command": {"linear", "angular",
  "lamp"}}``, each axis normalised to ``[-1, 1]``.
* ``GET /data`` - the telemetry snapshot: battery, GPS, orientation, IMU,
  wheel RPMs, signal level, lamp state.
* ``GET /v2/front`` / ``GET /v2/rear`` - one camera frame, base64 in the
  ``{camera}_frame`` field.
* ``POST /speak`` - text out of the rover's speaker.

``requests`` is imported lazily so the module loads without it; a real
connection needs it and a running SDK.

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
import threading
from typing import TYPE_CHECKING, Any, cast

from strands_robots.drivers.base import halt_failure_detail
from strands_robots.utils import finite_number_error, positive_finite_number_error

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The robots this driver registers for (read by ``_SHIPPED_DRIVERS``).
SUPPORTED_ROBOTS: tuple[str, ...] = ("earthrover",)

#: The whole control surface of a differential-drive base with a headlamp.
DRIVE_CHANNELS: tuple[str, ...] = ("linear", "angular", "lamp")

#: The camera views the SDK serves under ``/v2/<view>``.
CAMERA_VIEWS: tuple[str, ...] = ("front", "rear")

#: Where the vendor's SDK listens when started as documented.
DEFAULT_SDK_URL = "http://localhost:8001"


def _refuse(reason: str) -> dict[str, Any]:
    """One refusal envelope, so every refusal has the same shape."""
    return {"status": "error", "content": [{"text": reason}]}


def base_url_error(value: object, param: str, context: str) -> str | None:
    """Report why ``value`` is not an SDK base URL, or ``None`` if it is one.

    ``port=`` is polymorphic across drivers - a serial path on the arms, an IP
    on the DDS robots, a URL here - so the wrong *shape* is refused at the
    chokepoint with a sentence naming the shape that belongs elsewhere, not by
    ``requests`` failing with "No host supplied" one call later.

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
    return None


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
        if not base.startswith(("http://", "https://")):
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
        """The universal read-only trio plus a controlled stop.

        Motion verbs are deliberately absent from the *agent* surface, matching
        every shipped driver: the write path is :meth:`send_action`, opened by
        a caller who has read its contract, not by a model choosing an enum.
        """
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "EarthRover Mini+ native driver: reads the SDK's /data telemetry "
                    "(battery, GPS, orientation, IMU, wheel RPMs); twist writes go "
                    "through send_action, a halt through stop."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "sensors: return the latest telemetry snapshot; "
                                    "status: report connection and the last commanded twist; "
                                    "stop: command a zero twist"
                                ),
                                "enum": ["sensors", "status", "stop"],
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
        del kwargs, invocation_state  # forward-compat only
        tool_use_id = tool_use.get("toolUseId", "")
        action = (tool_use.get("input") or {}).get("action", "sensors")
        if action == "sensors":
            envelope = {"status": "success", "content": [{"json": self.read_state()}]}
        elif action == "status":
            envelope = await self.get_status()
        else:  # "stop"
            envelope = self.stop_task()
        yield {"toolUseId": tool_use_id, **envelope}

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
            self._connect_error = f"requests is not installed: {exc}. Install it with: pip install requests"
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
                ``angular`` in ``[-1, 1]`` (clamped), plus an optional ``lamp``
                truthy flag. An absent axis is commanded ``0.0``, because a
                twist is a complete statement of intent: "turn" also means
                "stop driving forward".
            robot_name: Accepted for parity; this driver fronts one rover.

        Returns:
            A success envelope naming the commanded twist as sent (after
            clamping and ``turn_sign``), or a refusal naming what was wrong.
        """
        session = self._session
        if session is None or not self._connected:
            suffix = f" ({self._connect_error})" if self._connect_error else ""
            return _refuse(f"send_action: not connected - call connect_eagerly() first{suffix}")
        bad = sorted(set(action) - set(DRIVE_CHANNELS))
        if bad:
            return _refuse(f"send_action: unknown drive channel(s) {bad}; valid: {list(DRIVE_CHANNELS)}")
        for axis in ("linear", "angular"):
            if axis in action and (reason := finite_number_error(action[axis], axis, "send_action")):
                return _refuse(reason)

        command: dict[str, float] = {
            "linear": max(-1.0, min(1.0, float(action.get("linear", 0.0)))),
            "angular": self._turn_sign * max(-1.0, min(1.0, float(action.get("angular", 0.0)))),
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

    def move(self, linear: float = 0.0, angular: float = 0.0) -> dict[str, Any]:
        """Command a twist by axis - sugar over :meth:`send_action`.

        Args:
            linear: Forward speed, ``[-1, 1]``.
            angular: Turn rate, ``[-1, 1]``, positive left.

        Returns:
            :meth:`send_action`'s envelope.
        """
        return self.send_action({"linear": linear, "angular": angular})

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
