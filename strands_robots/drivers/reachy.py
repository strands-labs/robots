"""Native daemon driver for the Pollen Robotics Reachy Mini.

``Robot("reachy_mini", mode="real", port="reachy-a.local:8000")`` builds one of
these. The instance satisfies
:class:`~strands_robots.drivers.base.HardwareDriver`, so
:func:`~strands_robots.robot.Robot` returns it and the mesh, teleop rail and
agent tool surface consume it exactly like the lerobot driver they replace for
this robot. There is nothing to replace in practice: the Reachy Mini has no
lerobot robot type, so before this driver ``mode="real"`` raised
``ValueError: Unsupported robot type: 'reachy_mini'``.

Why a native driver rather than a lerobot entry: the Mini is an expressive desk
robot - a 6-DOF Stewart head on a rotating body, two antennas, a speaker and a
recorded-emotion library - with no arms and no gait. lerobot's robot classes
model a serial servo bus or a network arm; the Mini is addressed through the
Reachy daemon's REST API plus a real-time link, and its state of interest is
head orientation rather than a joint-space arm pose.

What the driver actually does:

* Probes ``GET /api/daemon/status`` in :meth:`~ReachyDriver.connect_eagerly` -
  the reachability check, and the same call that reports which hardware variant
  answered. A **Lite** (no onboard computer) is driven over a WebSocket to the
  daemon; a **Wireless** (onboard CM4) over Zenoh. Both links come from
  :mod:`strands_robots.device_connect.reachy_transport`, which the Device
  Connect driver already ships - this module reuses them rather than growing a
  second daemon client.
* Runs that link on one background asyncio loop and caches what it delivers.
  ``_imu`` is the head IMU verbatim; ``_pose`` is the head orientation, taken
  from the IMU quaternion because the Mini's IMU is *in the head*; ``_battery``
  is read from the daemon status payload when it carries one. The mesh reads all
  three with ``getattr(robot, name, None)``, so a Mini that has not connected
  publishes no sensor topic and is otherwise complete.
* Refuses a motion write outside the envelope, naming the limit, via the shared
  :func:`~strands_robots.tools.reachy.envelope_error`.

Deliberately absent, so a reader is not left guessing:

* **No** ``_lidar_*``. The Mini has no lidar.
* No forward kinematics of the Stewart platform. The link reports six leg
  positions; turning those into a head pose needs a model of the platform this
  repo does not have, so the legs are cached as legs (``_joints``) and the head
  *orientation* comes from the IMU rather than being derived from them.
  Inventing the kinematics would put a number on the mesh that nothing
  measured.
* ``start_task`` and ``run_policy`` refuse outright rather than standing in for
  work in progress. A recorded emotion is not a policy rollout: with no arms and
  no gait there is no action space for a policy to be trained against, so the
  refusal names the recorded-move path instead of implying a rollout is coming.

Nothing here imports a transport at module load: every daemon touch is inside a
method body, so the module imports on Thor, on CI and in every unit test with a
mocked daemon.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from strands_robots.tools.reachy import envelope_error
from strands_robots.utils import finite_number_error, tcp_port_error

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The daemon's own default port. ``port="host"`` with no ``:port`` suffix means
#: this one, which is what every Reachy install uses unless it was reconfigured.
DEFAULT_API_PORT: int = 8000

#: REST paths this driver calls. Grouped so a reader sees the whole daemon
#: surface the driver depends on in one place, and so a test can assert against
#: the same constants the driver sends.
_PATH_STATUS = "/api/daemon/status"
_PATH_STOP = "/api/move/stop"

#: The module every daemon touch here goes through. It is a leaf that imports
#: nothing but the standard library, and its parent package
#: :mod:`strands_robots.device_connect` resolves ``device_connect_edge`` and the
#: three Device Connect drivers lazily, so importing the leaf executes no
#: third-party import. Nothing an extra installs can therefore decide whether
#: this import succeeds: on a stock ``pip install strands-robots`` it does, and a
#: failure that still reaches :func:`_resolve_transport` is a broken install of a
#: module the core distribution ships rather than a missing optional dependency.
_TRANSPORT_MODULE = "strands_robots.device_connect.reachy_transport"


def _resolve_transport() -> Any:
    """Return the Reachy transport module, or a reason naming what failed.

    The same shape as
    :func:`~strands_robots.drivers.g1._resolve_message_class`: the seam's other
    driver resolves its lazy SDK import through a helper that hands back a
    reason string, and every refusal boundary turns that string into a named
    failure. Doing the same here keeps the driver's no-raise contract intact
    when the transport module cannot be imported - ``connect_eagerly`` reports a
    reason and leaves the driver disconnected but usable, rather than raising
    ``ModuleNotFoundError`` through the agent tool surface.

    The reason reports the module and the underlying ``ImportError`` and stops
    there. It prescribes no install remedy because there is none this branch
    could establish: the transport leaf imports nothing outside the standard
    library, so no ``pip install`` supplies a module whose absence would reach
    here. That is the position the shared optional-dependency helper
    :func:`~strands_robots.utils.require_optional` already refuses to print a
    pip line in, because such a line "would hand the caller an instruction that
    reports success without supplying the module". Every cause that can still
    reach this branch - a shadowing module, a partial wheel, a corrupt install -
    is described by the ``ImportError`` itself.

    Returns:
        The imported module, or a reason string naming the module and the cause.
    """
    try:
        import importlib

        return importlib.import_module(_TRANSPORT_MODULE)
    except ImportError as exc:
        return f"cannot import {_TRANSPORT_MODULE}: {exc}"


#: Keys the daemon status payload might carry a battery percentage under. The
#: daemon documents status as "daemon status, motor state, and control
#: frequency" and this repo cannot confirm a battery field without hardware, so
#: the read is defensive: a payload that carries one populates ``_battery``, a
#: payload that does not leaves it ``None`` and the mesh publishes no battery
#: topic. Guessing a dedicated ``/api/battery`` endpoint would be a request no
#: measurement supports.
_BATTERY_KEYS: tuple[str, ...] = ("battery_level", "battery_pct", "battery", "soc")


class ReachyDriver:
    """Native driver for the Pollen Reachy Mini.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver` structurally
    - a Protocol - so no import from :mod:`strands_robots.drivers.base` is
    needed. The surface check that
    :func:`~strands_robots.drivers.register_native_driver` runs at registration
    time is what pins the contract for this class.
    """

    def __init__(
        self,
        tool_name: str = "reachy_mini",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        api_port: int = DEFAULT_API_PORT,
        zenoh_prefix: str | None = None,
        transport: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` talks to the daemon.

        The three leading arguments are the ones every native driver takes - see
        :mod:`strands_robots.drivers.base`'s constructor contract - so the
        factory can build any driver the same way.

        Args:
            tool_name: Name the agent invokes the driver by, and the mesh peer
                id when the driver is wrapped by
                :class:`~strands_robots.mesh.Mesh`. Two Minis on one mesh differ
                by this and nothing else, which is what makes a two-instance
                bring-up work without either knowing about the other.
            cameras: Accepted for parity with the lerobot driver. The Mini's
                camera is reached through the daemon rather than v4l2, so this
                driver does not open it.
            data_config: Accepted for parity; unused.
            port: The daemon host, optionally with a port -
                ``"reachy-a.local"`` or ``"reachy-a.local:8000"``. ``port`` is
                polymorphic across drivers by contract; here it names a host,
                because that is what addresses a Mini. ``None`` means
                ``localhost``, which is where a Lite's daemon runs.
            api_port: Daemon port to use when ``port`` carries no ``:port``
                suffix. An explicit suffix in ``port`` wins.
            zenoh_prefix: Zenoh key prefix for a Wireless Mini. Defaults to
                ``tool_name``, so two Minis do not share a key space.
            transport: Zenoh transport for a Wireless Mini, passed through to
                :class:`~strands_robots.device_connect.reachy_transport.ZenohLink`.
                ``None`` is valid: a Lite does not need one, and a Wireless
                without one reports a named connect failure rather than raising.
            **kwargs: Ignored; accepted so the factory can forward extras
                without the driver knowing what they are.

        Raises:
            ValueError: If ``api_port`` or a ``:port`` suffix in ``port`` is not
                a usable TCP port. A bad port cannot address a daemon, and
                refusing here means the mistake surfaces at construction rather
                than as an unreachable host minutes later.
        """
        del cameras, data_config  # accepted for parity; unused here
        if kwargs:
            logger.debug("ReachyDriver ignoring extra kwargs: %s", sorted(kwargs))

        self._tool_name = tool_name
        self._host, self._api_port = _split_host_port(port, api_port)
        self._zenoh_prefix = zenoh_prefix or tool_name
        self._transport = transport

        # Sensor caches. Every one is optional per the mesh contract, so a
        # driver that has not connected is not broken. Written by link
        # callbacks on the loop thread; read by mesh loops on theirs.
        self._cache_lock = threading.Lock()
        self._imu: dict[str, Any] | None = None
        self._pose: dict[str, Any] | None = None
        self._battery: dict[str, Any] | None = None
        self._joints: dict[str, Any] | None = None

        # Connection state. ``None`` link on a machine that never connected is
        # a valid state for tests and for a peer built ahead of a bring-up.
        self._link: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._connected: bool = False
        self._connect_error: str | None = None
        self._variant: str | None = None
        self._stopped: bool = False

    # ------------------------------------------------------------------ #
    # Agent tool surface (matches AgentTool's abstract members).         #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """The name the Strands agent invokes this driver by."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Always ``"robot"`` - mirrors the lerobot driver."""
        return "robot"

    @property
    def tool_spec(self) -> ToolSpec:
        """A minimal agent-facing spec.

        The expressive verb set (``look``, ``antennas``, ``express``, ``say``)
        arrives as ``reachy_*`` agent tools built on the same daemon and the
        same shared envelope; they are a separate change. Here we ship the
        universal ``status``/``stop`` verbs and a ``sensors`` read-out, so an
        agent can introspect a Mini the day the driver merges.
        """
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Pollen Reachy Mini native driver: reads the Reachy daemon for head "
                    "IMU, head orientation and battery, and stops motion on request. "
                    "Expressive motion verbs arrive with the reachy_* tool bundle."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "sensors: return the latest cached IMU/pose/battery/joints; "
                                    "status: report daemon reachability and hardware variant; "
                                    "stop: ask the daemon to stop any motion in progress"
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
        """Handle one agent invocation and yield exactly one tool result.

        Args:
            tool_use: The agent's request, carrying the tool id and parameters.
            invocation_state: Caller-provided state; unused here.
            **kwargs: Forward compatibility only.

        Yields:
            One tool result envelope, carrying the requested read-out.
        """
        del kwargs, invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        action = (tool_use.get("input") or {}).get("action", "sensors")
        if action == "sensors":
            envelope: dict[str, Any] = {
                "status": "success",
                "content": [
                    {
                        "json": {
                            "imu": self._snapshot("_imu"),
                            "pose": self._snapshot("_pose"),
                            "battery": self._snapshot("_battery"),
                            "joints": self._snapshot("_joints"),
                        }
                    }
                ],
            }
        elif action == "status":
            envelope = {"status": "success", "content": [{"json": await self.get_status()}]}
        else:  # "stop"
            # Report the halt outcome rather than assert one.  ``stop`` is the
            # protocol's shutdown hook and returns ``None``: a daemon that
            # refuses the stop is logged and swallowed, so an envelope built
            # beside it can only restate the intent - and its text named a
            # daemon that had just declined.  ``stop_task`` posts the same
            # ``/api/move/stop`` and already decides the verdict, so the verb
            # returns that envelope rather than re-deriving one.
            envelope = self.stop_task()
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Lifecycle and status.                                              #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Probe the daemon, then start the real-time link. Idempotent.

        The factory only constructs the driver; whoever performs the bring-up
        calls this, so a real bring-up fails here rather than on the first mesh
        poll. Off hardware - the Thor case - the REST probe cannot reach a
        daemon and the returned reason names the address that did not answer.

        A second call on a connected driver is a no-op success rather than a
        second link: rebuilding would drop the only reference to the running
        link and leave its reader task subscribed.

        Returns:
            ``None`` on success and on a call against an already-connected
            driver. A named reason on failure - the driver is left disconnected
            but usable, so a mesh peer for a Mini that is switched off can still
            be constructed for later use.
        """
        if self._connected:
            logger.debug("%s already connected; connect_eagerly() is a no-op", self._tool_name)
            return None

        # Resolved before the probe so a missing extra is reported as a missing
        # extra. Routed through the probe instead, it would come back wrapped as
        # "daemon unreachable (host:port)" and send an operator to check a
        # network that was never touched.
        transport = _resolve_transport()
        if isinstance(transport, str):
            self._connect_error = transport
            return transport

        status = self._daemon_get(_PATH_STATUS)
        if (error := status.get("error")) is not None:
            reason = f"daemon unreachable ({self._host}:{self._api_port}): {error}"
            self._connect_error = reason
            return reason

        # The daemon reports the variant; a payload without the flag is treated
        # as a Wireless because that is the shipped default, matching the
        # Device Connect driver's reading of the same field.
        is_lite = not status.get("wireless_version", True)
        self._variant = "lite" if is_lite else "wireless"
        self._absorb_status(status)

        link = self._build_link(is_lite=is_lite)
        if isinstance(link, str):
            self._connect_error = link
            return link

        error_text = self._start_link(link)
        if error_text is not None:
            self._connect_error = error_text
            return error_text

        self._link = link
        self._connected = True
        self._connect_error = None
        self._stopped = False
        return None

    def _build_link(self, *, is_lite: bool) -> Any:
        """Return the link for this hardware variant, or a reason string.

        Kept as a method, like
        :meth:`~strands_robots.drivers.g1.G1Driver._subscription_plan`, so a
        test can substitute a link without a daemon and without patching an
        import.

        Args:
            is_lite: Whether the daemon reported a Lite (no onboard computer).

        Returns:
            A ``HardwareLink``, or a reason naming what the variant needs and
            did not get.
        """
        transport = _resolve_transport()
        if isinstance(transport, str):
            return transport

        if is_lite:
            return transport.WebSocketLink(self._host, self._api_port)
        if self._transport is None:
            return (
                f"daemon at {self._host}:{self._api_port} reports a Wireless Mini, which is driven over "
                "Zenoh - pass transport= to reach it"
            )
        return transport.ZenohLink(self._transport, self._zenoh_prefix)

    def _start_link(self, link: Any) -> str | None:
        """Start ``link`` on a dedicated background asyncio loop.

        The links are async and the mesh, the agent tool path and
        :meth:`connect_eagerly` are all synchronous callers, so the driver owns
        one loop on one thread for the link's lifetime. A daemon thread so a
        process that forgets :meth:`cleanup` still exits.

        Args:
            link: The link to start.

        Returns:
            ``None`` on success, or a reason naming the failure.
        """
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            name=f"{self._tool_name}-reachy-link",
            daemon=True,
        )
        thread.start()
        try:
            future = asyncio.run_coroutine_threadsafe(
                link.start(on_joints=self._on_joints, on_imu=self._on_imu),
                loop,
            )
            future.result(timeout=10)
        except Exception as exc:  # noqa: BLE001 - any link failure is a connect failure
            loop.call_soon_threadsafe(loop.stop)
            return f"link to {self._host}:{self._api_port} failed to start: {exc}"
        self._loop = loop
        self._loop_thread = thread
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report reachability, hardware variant and the latest battery read.

        The shape matches the lerobot driver's ``get_status`` envelope so the
        mesh publishes both peers identically.

        Returns:
            A success envelope carrying the driver's connection state.
        """
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self._connected,
                        "connect_error": self._connect_error,
                        "host": self._host,
                        "api_port": self._api_port,
                        "variant": self._variant,
                        "motion_stopped": self._stopped,
                        "battery_pct": (self._battery or {}).get("pct"),
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Ask the daemon to stop any motion in progress.

        Unlike a robot with no motion path, the Mini has a real stop:
        ``POST /api/move/stop`` halts a recorded move mid-play. The link stays
        up, so sensors keep arriving - a stopped Mini is still observable, which
        is what an operator wants after halting it.
        """
        result = self._daemon_post(_PATH_STOP)
        if (error := result.get("error")) is not None:
            logger.warning("%s.stop(): daemon refused the stop: %s", self._tool_name, error)
            return
        self._stopped = True

    def cleanup(self) -> None:
        """Stop the link and its loop. Idempotent."""
        if self._link is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._link.stop(), self._loop).result(timeout=5)
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                logger.debug("%s: link stop failed during cleanup: %s", self._tool_name, exc)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._link = None
        self._loop = None
        self._loop_thread = None
        self._connected = False

    # ------------------------------------------------------------------ #
    # Command path.                                                      #
    # ------------------------------------------------------------------ #

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Command head pose, body yaw and antennas, refusing what cannot be met.

        Three gates, in this order:

        1. The driver is connected. A write to a link that was never started has
           nowhere to go.
        2. Every numeric value is finite, and every bounded axis is inside the
           envelope - both from the shared
           :func:`~strands_robots.tools.reachy.envelope_error`, so this driver
           and the ``reachy_*`` tools cannot disagree about the same robot.
        3. The action names at least one thing this driver can send. An action
           dict of unknown keys is refused rather than reported as a successful
           no-op.

        Degrees in, radians and pose matrices out. The caller-facing unit is
        degrees because the envelope is expressed in degrees and because the
        daemon's own RPC surface takes degrees; the wire wants radians for a
        joint and a 4x4 matrix for the head, which
        :func:`~strands_robots.device_connect.reachy_transport.rpy_to_pose`
        builds.

        Args:
            action: Any of ``head_pitch``, ``head_roll``, ``head_yaw`` (degrees),
                ``head_x``, ``head_y``, ``head_z`` (millimetres), ``body_yaw``
                (degrees), ``antenna_left``, ``antenna_right`` (degrees). Absent
                head axes default to zero, so ``{"head_yaw": 20}`` means "look
                20 degrees left, level" rather than "leave pitch as it was" -
                the daemon's head command is a whole pose, not a delta.
            robot_name: Accepted for contract parity. This driver fronts exactly
                one Mini, so a name that is neither ``None`` nor this driver's
                own is refused rather than silently applied to the wrong robot.

        Returns:
            A success envelope naming what was sent, or an error envelope naming
            the first thing that refused.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return _refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        if not self._connected:
            return _refuse("not connected - call connect_eagerly() first")

        for name, value in action.items():
            if (reason := finite_number_error(value, name, "send_action")) is not None:
                return _refuse(reason)
        if (reason := envelope_error(action, "send_action")) is not None:
            return _refuse(reason)

        commands = _wire_commands(action)
        if isinstance(commands, str):
            return _refuse(f"send_action: {commands}")
        if not commands:
            return _refuse(
                f"send_action: nothing to send - none of {sorted(action)} names a Reachy Mini axis; "
                f"expected any of {sorted(_ACTION_KEYS)}"
            )

        for command in commands:
            if (error := self._send_cmd(command)) is not None:
                return _refuse(f"send_action: {error}")
        return {
            "status": "success",
            "content": [{"json": {"sent": [sorted(c) for c in commands], "robot": self._tool_name}}],
        }

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
        """Refuse a policy-driven task: the Mini has no policy path.

        Not a stub awaiting wiring like a manipulator's would be. The Mini has
        no arms and no gait, so there is no action space a ``groot`` or lerobot
        policy is trained against; its expressive behaviour is a *recorded move*
        played by the daemon, which the ``reachy_*`` tool bundle plays by name.
        The refusal says so, rather than implying a rollout is coming.

        Args:
            instruction: Ignored; named for contract parity.
            policy_port: Ignored.
            policy_host: Ignored.
            policy_provider: Ignored.
            duration: Ignored.
            **policy_kwargs: Ignored.

        Returns:
            An error envelope naming the recorded-move path instead.
        """
        del instruction, policy_port, policy_host, policy_provider, duration, policy_kwargs
        return _refuse(
            "start_task: the Reachy Mini has no policy action space (no arms, no gait); "
            "play a recorded move through the reachy_* tools instead"
        )

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refuse a rollout, for the same reason as :meth:`start_task`.

        Args:
            policy_object: Ignored; named for contract parity.
            instruction: Ignored.
            duration: Ignored.
            n_steps: Ignored.

        Returns:
            An error envelope naming the recorded-move path instead.
        """
        del policy_object, instruction, duration, n_steps
        return _refuse(
            "run_policy: the Reachy Mini has no policy action space (no arms, no gait); "
            "play a recorded move through the reachy_* tools instead"
        )

    def get_task_status(self) -> dict[str, Any]:
        """Report that no task is running, because none can be started.

        Returns:
            A success envelope: the question is answerable even though
            :meth:`start_task` refuses.
        """
        return {
            "status": "success",
            "content": [{"json": {"running": False, "reason": "the Reachy Mini has no policy task path"}}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Stop motion, since a Mini's closest thing to a task is a recorded move.

        Returns:
            A success envelope describing the stop that was attempted.
        """
        result = self._daemon_post(_PATH_STOP)
        if (error := result.get("error")) is not None:
            return _refuse(f"stop_task: daemon refused the stop: {error}")
        self._stopped = True
        return {"status": "success", "content": [{"text": "asked the daemon to stop any recorded move in progress"}]}

    # ------------------------------------------------------------------ #
    # Link callbacks. Each runs on the loop thread; keep fast and pure.  #
    # ------------------------------------------------------------------ #

    def _on_joints(self, payload: dict[str, Any]) -> None:
        """Cache joint positions, converting the link's radians to degrees.

        Args:
            payload: The link's joints message, carrying
                ``head_joint_positions`` and ``antennas_joint_positions`` in
                radians.
        """
        try:
            head = [math.degrees(float(j)) for j in payload.get("head_joint_positions", [])]
            antennas = [math.degrees(float(j)) for j in payload.get("antennas_joint_positions", [])]
            with self._cache_lock:
                self._joints = {
                    "head_leg_deg": head,
                    "antennas_deg": antennas,
                    "t": time.time(),
                }
        except (TypeError, ValueError) as exc:
            logger.debug("%s: joints decode failed: %s", self._tool_name, exc)

    def _on_imu(self, payload: dict[str, Any]) -> None:
        """Cache the head IMU, and derive :attr:`_pose` from its quaternion.

        The Mini's IMU is mounted in the head, so its quaternion *is* the head's
        orientation - a measurement, not a model. That is why ``_pose`` is
        derived here rather than from the six leg positions, which would need
        Stewart-platform forward kinematics this repo does not have.

        Args:
            payload: The link's IMU message, carrying ``accelerometer``,
                ``gyroscope``, ``quaternion`` and ``temperature``.
        """
        try:
            imu = {
                "accelerometer": payload.get("accelerometer"),
                "gyroscope": payload.get("gyroscope"),
                "quaternion": payload.get("quaternion"),
                "temperature": payload.get("temperature"),
                "t": time.time(),
            }
            pose: dict[str, Any] | None = None
            quaternion = payload.get("quaternion")
            if quaternion is not None:
                pose = {
                    "quat": list(quaternion),
                    "frame": "head",
                    "source": "imu",
                    "t": imu["t"],
                }
            with self._cache_lock:
                self._imu = imu
                if pose is not None:
                    self._pose = pose
        except (TypeError, ValueError) as exc:
            logger.debug("%s: imu decode failed: %s", self._tool_name, exc)

    def _absorb_status(self, status: dict[str, Any]) -> None:
        """Populate :attr:`_battery` from a daemon status payload, if it has one.

        Args:
            status: The decoded ``/api/daemon/status`` body.
        """
        for key in _BATTERY_KEYS:
            value = status.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                pct = float(value)
            except (TypeError, ValueError):
                continue
            with self._cache_lock:
                self._battery = {"pct": pct, "source": key, "t": time.time()}
            return

    # ------------------------------------------------------------------ #
    # Internal helpers.                                                  #
    # ------------------------------------------------------------------ #

    def _daemon_get(self, path: str) -> dict[str, Any]:
        """Call the daemon's REST API with GET.

        Args:
            path: Request path, one of this module's ``_PATH_*`` constants.

        Returns:
            The decoded body, or ``{"error": ...}`` - the shape
            :func:`~strands_robots.device_connect.reachy_transport.api` returns
            for every failure, which is why no call here needs a ``try``.
        """
        transport = _resolve_transport()
        if isinstance(transport, str):
            return {"error": transport}

        result: dict[str, Any] = transport.api(self._host, self._api_port, path)
        return result

    def _daemon_post(self, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call the daemon's REST API with POST.

        Args:
            path: Request path, one of this module's ``_PATH_*`` constants.
            data: JSON body, or ``None``.

        Returns:
            The decoded body, or ``{"error": ...}``.
        """
        transport = _resolve_transport()
        if isinstance(transport, str):
            return {"error": transport}

        result: dict[str, Any] = transport.api(self._host, self._api_port, path, method="POST", data=data)
        return result

    def _send_cmd(self, command: dict[str, Any]) -> str | None:
        """Put one real-time command on the link.

        Args:
            command: The link-level command dict.

        Returns:
            ``None`` on success, or a reason naming the failure.
        """
        if self._link is None or self._loop is None:
            return "link is not running"
        try:
            asyncio.run_coroutine_threadsafe(self._link.send_cmd(command), self._loop).result(timeout=5)
        except Exception as exc:  # noqa: BLE001 - a link can fail in any way
            return f"link refused the command: {exc}"
        return None

    def _snapshot(self, attr: str) -> dict[str, Any] | None:
        """Return a copy of one cached sensor dict, or ``None``.

        Args:
            attr: Cache attribute name, e.g. ``"_imu"``.

        Returns:
            A shallow copy, so a caller who mutates the result does not corrupt
            the cache the link thread writes into.
        """
        with self._cache_lock:
            value: dict[str, Any] | None = getattr(self, attr, None)
            return None if value is None else dict(value)


#: Every key :func:`_wire_commands` understands. Surfaced so a refusal can name
#: the accepted set rather than leaving a caller to read the source.
_ACTION_KEYS: frozenset[str] = frozenset(
    {
        "head_pitch",
        "head_roll",
        "head_yaw",
        "head_x",
        "head_y",
        "head_z",
        "body_yaw",
        "antenna_left",
        "antenna_right",
    }
)

#: The head pose is one command built from up to six keys, so the presence of
#: any of them means a head command is wanted.
_HEAD_KEYS: tuple[str, ...] = ("head_pitch", "head_roll", "head_yaw", "head_x", "head_y", "head_z")


def _wire_commands(action: dict[str, Any]) -> list[dict[str, Any]] | str:
    """Translate a degrees-and-millimetres action into link commands.

    Three commands at most, because the daemon addresses the Mini's three
    movable groups separately: a 4x4 ``head_pose``, a scalar ``body_yaw`` in
    radians, and a two-element ``antennas_joint_positions`` in radians. Only the
    groups the action mentions are built, so commanding the antennas does not
    also re-send a head pose.

    Args:
        action: A validated action dict; see
            :meth:`ReachyDriver.send_action` for the accepted keys.

    Returns:
        The commands to send, in a fixed order - head, body, antennas - so two
        identical actions always produce the same sequence. Or a reason string
        if the transport module cannot be imported: the head pose is built by
        the transport's own ``rpy_to_pose``, so this is a refusal boundary like
        the daemon calls, and returning the reason keeps it out of the raising
        path even though ``send_action``'s connected gate should already have
        refused.
    """
    transport = _resolve_transport()
    if isinstance(transport, str):
        return transport

    commands: list[dict[str, Any]] = []
    if any(key in action for key in _HEAD_KEYS):
        commands.append(
            {
                "head_pose": transport.rpy_to_pose(
                    float(action.get("head_pitch", 0.0)),
                    float(action.get("head_roll", 0.0)),
                    float(action.get("head_yaw", 0.0)),
                    float(action.get("head_x", 0.0)),
                    float(action.get("head_y", 0.0)),
                    float(action.get("head_z", 0.0)),
                )
            }
        )
    if "body_yaw" in action:
        commands.append({"body_yaw": math.radians(float(action["body_yaw"]))})
    if "antenna_left" in action or "antenna_right" in action:
        commands.append(
            {
                "antennas_joint_positions": [
                    math.radians(float(action.get("antenna_left", 0.0))),
                    math.radians(float(action.get("antenna_right", 0.0))),
                ]
            }
        )
    return commands


def _split_host_port(port: str | None, api_port: int) -> tuple[str, int]:
    """Split a ``host[:port]`` string into a host and a validated TCP port.

    ``port=`` is polymorphic across drivers by contract, and for a daemon-backed
    robot the natural spelling is the one an operator already types into a
    browser. Accepting both ``"host"`` and ``"host:8000"`` means a caller does
    not have to know which one this driver wanted.

    Args:
        port: The caller's ``port=``, or ``None`` for ``localhost``.
        api_port: Fallback port when ``port`` carries no suffix.

    Returns:
        ``(host, port)``.

    Raises:
        ValueError: If either the suffix or ``api_port`` is not a usable TCP
            port. Raised rather than returned because a driver with no
            addressable daemon has nothing to degrade to.
    """
    host = "localhost"
    resolved = api_port
    if port:
        text = str(port)
        head, separator, tail = text.rpartition(":")
        if separator and head and tail.isdigit():
            host, resolved = head, int(tail)
        elif separator and head:
            raise ValueError(
                f"ReachyDriver: port {text!r} names a host and a port, but {tail!r} is not a number - "
                'expected "host" or "host:8000"'
            )
        else:
            host = text
    if (reason := tcp_port_error(resolved, "api_port", "ReachyDriver")) is not None:
        raise ValueError(reason)
    return host, resolved


def _refuse(reason: str) -> dict[str, Any]:
    """Return the driver's error envelope with ``reason`` inside.

    Args:
        reason: Text naming what refused.

    Returns:
        The error envelope, in the one shape every refusal path here renders.
    """
    return {"status": "error", "content": [{"text": reason}]}
