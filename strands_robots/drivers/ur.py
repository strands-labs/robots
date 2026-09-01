"""Native RTDE driver for the Universal Robots e-Series (UR5e, UR10e).

``Robot("ur5e", mode="real", driver="strands", port="192.168.1.10")`` builds one
of these. The instance satisfies
:class:`~strands_robots.drivers.base.HardwareDriver`, so
:func:`~strands_robots.robot.Robot` returns it and the mesh, teleop rail and
agent tool surface consume it exactly like the lerobot driver they replace for
this robot. There is nothing to replace in practice: lerobot registers no robot
type for a UR arm, so before this driver ``mode="real"`` could only answer
``Unsupported robot type: 'ur5e'``.

Why RTDE rather than a lerobot entry: a UR arm is not a serial servo bus. The
controller owns the joints and exposes them over the Real-Time Data Exchange
protocol on port 30004 - a fixed-rate register interface, 500 Hz on the
e-Series - with motion commands issued through the control interface. The
``ur_rtde`` package (``rtde_control``, ``rtde_receive``) speaks both, and this
driver is the adapter between them and this package's joint-name-keyed action
dicts.

What the driver actually does:

* :meth:`URDriver.connect_eagerly` builds the receive interface first and the
  control interface second, because the receive side answers the question that
  decides whether commanding is possible at all: a controller in
  ``PROTECTIVE_STOP`` accepts an RTDE connection and silently performs no
  motion. Off hardware it reports a reason and leaves the driver usable - every
  read returns its cache, every write refuses "not connected".
* :meth:`URDriver.state` reads the three things a UR caller asks for in one
  round trip: joint positions and velocities, the TCP pose, and the TCP wrench.
  The wrench is the arm's own force-torque estimate, which is why it belongs on
  the state read rather than behind a separate sensor surface.
* :meth:`URDriver.send_action` maps a joint-name-keyed action onto ``servoJ``.
  ``servoJ`` rather than ``moveJ`` because a policy streams: ``moveJ`` plans a
  trajectory per call and a 50 Hz stream of them fights itself, where
  ``servoJ`` is the controller's own streaming-setpoint primitive.
* Every write is gated on the controller's mode *and* on the size of the step it
  asks for. Both gates exist because a UR controller does not report a refusal
  the way a servo bus does: it accepts the register write and does nothing, so a
  driver that reported success from a queued setpoint would be reporting the
  wire and not the arm.
* :meth:`URDriver.run_policy` rolls a caller-built policy at a fixed cadence,
  and :meth:`URDriver.start_task` builds a policy from the provider registry and
  runs the same rollout on a background thread.

Deliberately absent, so a reader is not left guessing:

* **No inverse kinematics.** ``servoL``/``moveL`` take a Cartesian pose and the
  controller solves for it, but nothing here converts a pose into joint targets
  itself: the action space is joint space, which is the space the policies in
  this package emit.
* **No gripper.** A UR ships without an end-effector; the tool on the flange is
  a separate device (a Robotiq hand speaks its own protocol over the tool
  connector or a URCap). ``robotiq_2f85`` is its own registry entry, and pairing
  the two is a composition rather than a member of this driver.
* **No freedrive, no URScript upload.** Both take the arm out of the RTDE
  control mode this driver holds, so exposing them here would let an agent
  invalidate the state machine every other method depends on.

Nothing here imports ``ur_rtde`` at module load. Every SDK touch is inside a
function body, so the module imports on Thor, on CI and in every unit test with
a fake controller.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, cast

from strands_robots.mesh.pacing import Ticker
from strands_robots.registry import resolve_name
from strands_robots.utils import (
    finite_number_error,
    positive_count_error,
    positive_finite_number_error,
    tcp_port_error,
)

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The robots this driver serves, read by
#: :data:`strands_robots.drivers._SHIPPED_DRIVERS` so the family is declared
#: once, here, rather than restated in the registration table.
SUPPORTED_ROBOTS: tuple[str, ...] = ("ur5e", "ur10e")

#: Joint order for every vector crossing the RTDE wire, base to wrist. This is
#: the order ``getActualQ`` returns and ``servoJ`` expects, and it is also the
#: order the MuJoCo assets for both arms declare their joints in - so an action
#: dict recorded in simulation indexes onto the wire without a remap, and
#: :func:`targets_from_action` needs no per-robot key table.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

#: Per-joint position limit, radians. Every e-Series joint travels +/-360
#: degrees, so one number covers all six and both arms.
JOINT_LIMIT_RAD: float = 2.0 * math.pi

#: Maximum joint speed per model, radians/second, in :data:`JOINT_NAMES` order.
#: From the datasheets: every UR5e joint runs to 180 deg/s, where the UR10e's
#: three proximal joints are held to 120 deg/s by its longer reach. The
#: distinction is why one driver serving both arms still needs a table: sizing a
#: step budget with the UR5e's number on a UR10e would ask the controller for a
#: motion it will not perform.
MAX_JOINT_SPEED_RAD_S: dict[str, tuple[float, ...]] = {
    "ur5e": (math.pi,) * 6,
    "ur10e": (2.0 * math.pi / 3.0,) * 3 + (math.pi,) * 3,
}

#: Model whose limits an unrecognised UR name is held to. The slower arm on
#: purpose: an unknown model given the faster budget would have its steps
#: accepted here and dropped by the controller, which is the failure this gate
#: exists to prevent.
FALLBACK_MODEL: str = "ur10e"

#: The controller's own RTDE port. Fixed by the protocol; carried as a constant
#: so a caller passing ``port="10.0.0.2:30004"`` is answered rather than having
#: the suffix silently parsed as part of the host.
RTDE_PORT: int = 30004

#: Default cadence for :meth:`URDriver.run_policy`, hertz. Well under the
#: e-Series' 500 Hz RTDE rate: a policy that emits a chunk per inference cannot
#: fill 500 Hz, and asking for a rate the caller cannot feed produces gaps the
#: controller reads as a held setpoint.
DEFAULT_CONTROL_FREQUENCY: float = 125.0

#: ``servoJ`` arguments that are properties of the *controller*, not of a
#: caller's motion. ``speed`` and ``acceleration`` are ignored by ``servoJ``
#: (they are placeholders in the ur_rtde signature); ``lookahead_time`` and
#: ``gain`` are the servo's smoothing and stiffness, and these are ur_rtde's own
#: documented defaults for streamed setpoints.
SERVOJ_SPEED: float = 0.5
SERVOJ_ACCELERATION: float = 0.5
SERVOJ_LOOKAHEAD_TIME: float = 0.1
SERVOJ_GAIN: float = 300.0

#: ``getRobotMode`` value that means the arm is powered, braked off and running
#: a program - the only mode in which a motion command moves anything.
ROBOT_MODE_RUNNING: int = 7

#: ``getRobotMode`` values, for reporting a refusal in the controller's own
#: vocabulary rather than as a bare integer.
ROBOT_MODES: dict[int, str] = {
    -1: "NO_CONTROLLER",
    0: "DISCONNECTED",
    1: "CONFIRM_SAFETY",
    2: "BOOTING",
    3: "POWER_OFF",
    4: "POWER_ON",
    5: "IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "UPDATING_FIRMWARE",
}

#: ``getSafetyMode`` values in which the controller executes motion. NORMAL is
#: the everyday case; REDUCED is a configured slow zone, which still moves.
#: Every other mode - protective stop, safeguard stop, either emergency stop,
#: violation, fault, recovery - accepts the register write and performs nothing.
SAFETY_MODES_THAT_MOVE: frozenset[int] = frozenset({1, 2})

#: ``getSafetyMode`` values, for the same reason as :data:`ROBOT_MODES`.
SAFETY_MODES: dict[int, str] = {
    1: "NORMAL",
    2: "REDUCED",
    3: "PROTECTIVE_STOP",
    4: "RECOVERY",
    5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION",
    9: "FAULT",
}


def _mode_name(table: dict[int, str], value: int | None) -> str | None:
    """Name a controller mode word.

    Args:
        table: :data:`ROBOT_MODES` or :data:`SAFETY_MODES`.
        value: The word the controller reported, or ``None`` when it was not
            read - a driver that never connected, or a read that failed.

    Returns:
        The mode's name; the value as a string when the controller reports one
        this driver's table does not carry (a firmware newer than the table is
        reported, not swallowed); or ``None`` when there was nothing to name.
    """
    if value is None:
        return None
    return table.get(value, str(value))


def _refuse(reason: str) -> dict[str, Any]:
    """The driver's error envelope, one shape for every refusal path."""
    return {"status": "error", "content": [{"text": reason}]}


def _resolve_rtde() -> tuple[Any, Any] | str:
    """Import ``ur_rtde``'s two interfaces, or report why they are unavailable.

    The same shape as :func:`strands_robots.drivers.reachy._resolve_transport`:
    the seam's other drivers resolve a lazy SDK import through a helper that
    hands back a reason string, so a missing SDK reaches the caller as a named
    refusal from a method declared ``-> str | None`` rather than as an
    ``ImportError`` raised through the agent tool surface.

    Returns:
        ``(rtde_control, rtde_receive)`` on success, or a reason naming the
        module that failed and the package that supplies it. Both modules come
        from the single ``ur_rtde`` distribution, so one pip line is the remedy
        for either name.
    """
    import importlib

    try:
        control = importlib.import_module("rtde_control")
        receive = importlib.import_module("rtde_receive")
    except ImportError as exc:
        return (
            f"the ur_rtde SDK is not importable ({exc}). It supplies both rtde_control and "
            "rtde_receive; install it with 'pip install ur_rtde'."
        )
    return control, receive


def speed_limits(model: str) -> tuple[float, ...]:
    """Return the per-joint speed ceiling for ``model``, radians/second.

    Args:
        model: A UR robot name or alias.

    Returns:
        Six ceilings in :data:`JOINT_NAMES` order - the model's own row of
        :data:`MAX_JOINT_SPEED_RAD_S`, or :data:`FALLBACK_MODEL`'s row for a
        name the table does not carry.
    """
    return MAX_JOINT_SPEED_RAD_S.get(resolve_name(model), MAX_JOINT_SPEED_RAD_S[FALLBACK_MODEL])


def targets_from_action(
    action: dict[str, Any],
    reference: list[float],
    *,
    model: str = "ur5e",
    control_period: float | None = None,
) -> tuple[list[float], str | None]:
    """Turn a joint-name-keyed action into one ordered ``servoJ`` vector.

    The whole wire encoding, as a pure function: no SDK, no socket, no driver
    state. That is what lets the sim-to-real parity check drive it with an
    action dict recorded in MuJoCo and compare the vector against the joint
    trajectory the simulator produced, without a controller present.

    A key the action omits holds its reference value rather than defaulting to
    zero, because ``servoJ`` takes a whole-arm setpoint: a policy that names
    only the wrist would otherwise command the shoulder to fold.

    Args:
        action: Joint targets in radians, keyed by :data:`JOINT_NAMES` members.
        reference: The pose this setpoint is measured from, in
            :data:`JOINT_NAMES` order. Supplies the held value for any joint the
            action omits, and the origin of the step the gate sizes. While a
            stream is in progress that is the *last commanded setpoint* rather
            than the measured pose - see :meth:`URDriver.send_action` for why the
            distinction is load-bearing.
        model: Robot name deciding the speed ceilings, via :func:`speed_limits`.
        control_period: Seconds this setpoint has to be reached in. When given,
            a joint asked to move further than its ceiling allows in that time
            is refused. ``None`` skips the step gate - for a caller issuing a
            single setpoint with no cadence to size it against.

    Returns:
        ``(targets, reason)``. ``reason`` is ``None`` on success and ``targets``
        is then the six-vector to send; otherwise ``targets`` is empty and
        ``reason`` names what was wrong, the joint it was wrong on, and the
        limit it exceeded.
    """
    if control_period is not None and (
        reason := positive_finite_number_error(control_period, "control_period", "targets_from_action")
    ):
        # Not a formality: a nan period makes ``allowed`` nan, every
        # ``step > allowed`` comparison false, and the speed gate below
        # disappears without a word - the one failure mode a caller could not
        # detect from the envelope, because the envelope would report success.
        return [], reason
    if len(reference) != len(JOINT_NAMES):
        return [], (
            f"targets_from_action: reference holds {len(reference)} joint positions, expected "
            f"{len(JOINT_NAMES)} in the order {', '.join(JOINT_NAMES)}"
        )

    unknown = sorted(set(action) - set(JOINT_NAMES))
    if unknown:
        return [], (
            f"targets_from_action: {unknown} name no UR joint; expected any of {list(JOINT_NAMES)}. "
            "A UR arm has no gripper joint - drive an end-effector as its own device."
        )
    if not action:
        return [], f"targets_from_action: nothing to command - the action names none of {list(JOINT_NAMES)}"

    limits = speed_limits(model)
    targets = list(reference)
    for index, name in enumerate(JOINT_NAMES):
        if name not in action:
            continue
        value = action[name]
        if (reason := finite_number_error(value, name, "targets_from_action")) is not None:
            return [], reason
        target = float(value)
        if abs(target) > JOINT_LIMIT_RAD:
            return [], (
                f"targets_from_action: {name}={target:.4f} rad is outside the joint's "
                f"+/-{JOINT_LIMIT_RAD:.4f} rad travel"
            )
        if control_period is not None:
            allowed = limits[index] * control_period
            step = abs(target - reference[index])
            if step > allowed:
                return [], (
                    f"targets_from_action: {name} asks for {step:.4f} rad in {control_period:.4f} s, "
                    f"more than the {allowed:.4f} rad a {resolve_name(model)} joint travels in that time "
                    f"at its {limits[index]:.4f} rad/s ceiling. The controller would drop the setpoint "
                    "rather than track it; slow the policy or raise the control period."
                )
        targets[index] = target
    return targets, None


class URDriver:
    """Native RTDE driver for a Universal Robots e-Series arm.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver` structurally,
    so no import from that module is needed; the surface check
    :func:`~strands_robots.drivers.register_native_driver` runs at registration
    pins the contract.
    """

    def __init__(
        self,
        tool_name: str = "ur5e",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        model: str | None = None,
        control_frequency: float = DEFAULT_CONTROL_FREQUENCY,
        rtde_frequency: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` opens the RTDE sockets.

        Args:
            tool_name: Name the agent invokes the driver by, and the mesh peer
                id. The factory passes the canonical robot name, which is also
                what decides the speed ceilings unless ``model`` overrides it.
            cameras: Accepted for parity with other drivers; unused here. A UR
                controller carries no camera - a wrist camera on the flange is
                its own device.
            data_config: Accepted for parity; unused.
            port: The controller's IP address or hostname, optionally with a
                ``:30004`` suffix. ``port`` is the seam's polymorphic device
                address (a serial path for a servo bus, an address here).
            model: Which UR model's limits to enforce, when ``tool_name`` is not
                one - a renamed mesh peer, or a driver built directly.
            control_frequency: Default rollout cadence in hertz, used by
                :meth:`run_policy` and as the period the step gate sizes a
                setpoint against.
            rtde_frequency: Register-exchange rate handed to both interfaces.
                ``None`` lets ur_rtde choose the controller's maximum, which is
                what a caller wants unless they are sharing the arm with another
                RTDE client.
            **kwargs: Ignored; accepted so the factory can forward extras.

        Raises:
            ValueError: If ``control_frequency`` or ``rtde_frequency`` is not a
                positive finite number, or ``port`` carries an unusable TCP port
                suffix. Raised here rather than returned from
                :meth:`connect_eagerly`, which is declared ``-> str | None``: a
                cadence the loop cannot pace on is not a connection this driver
                can degrade to reporting.
        """
        del cameras, data_config
        if kwargs:
            logger.debug("URDriver ignoring extra kwargs: %s", sorted(kwargs))

        # Both rates reach a consumer that cannot report what it was handed:
        # ``control_frequency`` becomes a Ticker period (a nan or a zero
        # busy-spins the rollout thread) and is the denominator of the step
        # gate, where an inf would admit every step; ``rtde_frequency`` is
        # passed into the ur_rtde constructors, which raise out of
        # ``connect_eagerly`` naming neither this driver nor the parameter.
        if reason := positive_finite_number_error(control_frequency, "control_frequency", "URDriver"):
            raise ValueError(reason)
        if rtde_frequency is not None and (
            reason := positive_finite_number_error(rtde_frequency, "rtde_frequency", "URDriver")
        ):
            raise ValueError(reason)

        self._tool_name = tool_name
        self._model = resolve_name(model or tool_name)
        self._host, host_reason = self._split_host(port)
        if host_reason is not None:
            raise ValueError(host_reason)
        self._control_frequency = float(control_frequency)
        self._rtde_frequency = rtde_frequency

        self._lock = threading.Lock()
        self._control: Any = None
        self._receive: Any = None
        self._connect_error: str | None = None

        # Mesh sensor caches. Read with ``getattr(robot, name, None)``, so an
        # arm that has not connected publishes no topic and is otherwise
        # complete.
        self._joints: dict[str, float] = {}
        self._pose: dict[str, Any] | None = None

        # The last setpoint the controller accepted, or ``None`` when no stream
        # is in progress. ``_reference_pose`` explains why the step gate is
        # anchored here rather than on the measured pose.
        self._commanded: list[float] | None = None

        self._rollout: _Rollout | None = None
        self._task_admission = threading.Lock()

    @staticmethod
    def _split_host(port: str | None) -> tuple[str, str | None]:
        """Split ``port`` into a host, refusing a suffix that is not RTDE's.

        Args:
            port: The caller's ``port=``, or ``None``.

        Returns:
            ``(host, reason)``. ``host`` is empty when no address was given,
            which :meth:`connect_eagerly` reports rather than dialling; ``reason``
            names an unusable suffix.
        """
        if not port:
            return "", None
        text = str(port)
        if ":" not in text:
            return text, None
        host, _, suffix = text.rpartition(":")
        # ``tcp_port_error`` is the package's port domain and takes an int, so a
        # non-numeric suffix is handed over as-is for it to name rather than
        # being coerced into a ValueError from ``int()``.
        number: Any = int(suffix) if suffix.isdigit() else suffix
        if reason := tcp_port_error(number, "port", "URDriver"):
            return "", reason
        if number != RTDE_PORT:
            return "", (
                f"URDriver: port {number} is not the RTDE port. The protocol fixes it at "
                f"{RTDE_PORT}, so pass the address alone (port={host!r})."
            )
        return host, None

    # ------------------------------------------------------------------ #
    # Agent tool surface.                                                #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """The name the Strands agent invokes this driver by."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Tool kind reported to the agent runtime."""
        return "robot"

    @property
    def is_connected(self) -> bool:
        """Whether both RTDE interfaces are live.

        Read by :func:`strands_robots.bus_access.joint_read_source` alongside
        :meth:`get_observation` to decide whether this driver's joints reach the
        mesh state topic.
        """
        return self._receive is not None and self._control is not None

    @property
    def tool_spec(self) -> ToolSpec:
        """A minimal agent-facing spec: read state, report status, stop."""
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Universal Robots e-Series native RTDE driver. Reads joints, TCP pose and TCP "
                    "wrench from the controller, reports connection and safety state, and stops "
                    "motion. Joint targets go through send_action; a rollout through run_policy."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "state: joints, TCP pose and wrench read from the controller; "
                                    "status: reachability, robot mode and safety mode; "
                                    "stop: decelerate the arm and leave it connected"
                                ),
                                "enum": ["state", "status", "stop"],
                                "default": "state",
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
        action = (tool_use.get("input") or {}).get("action", "state")
        if action == "state":
            envelope = self.state()
        elif action == "status":
            envelope = {"status": "success", "content": [{"json": await self.get_status()}]}
        else:  # "stop"
            # ``stop`` is the protocol's shutdown hook and returns ``None``, so
            # an envelope built beside it could only restate the intent.
            # ``stop_task`` performs the same halt and already decides the
            # verdict, so the verb reports that.
            envelope = self.stop_task()
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Lifecycle.                                                         #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Open both RTDE interfaces and check the arm can actually be moved.

        Returns ``None`` on success. Off hardware - no controller at
        :attr:`_host` - returns a reason naming the address that did not answer
        and leaves the driver usable (every read returns its empty cache, every
        write refuses "not connected"). Idempotent: a second call on a live pair
        is a no-op success.

        The receive interface is built first and interrogated before the control
        interface is opened at all. A controller sitting in ``PROTECTIVE_STOP``
        or with the brakes on accepts an RTDE *control* connection and then
        performs no motion, so a driver that reported success from the socket
        alone would have reported the wire rather than the arm.
        """
        if self.is_connected:
            return None
        if not self._host:
            self._connect_error = (
                "URDriver: no controller address - pass the arm's IP as port=, e.g. "
                'Robot("ur5e", mode="real", driver="strands", port="192.168.1.10")'
            )
            return self._connect_error

        resolved = _resolve_rtde()
        if isinstance(resolved, str):
            self._connect_error = resolved
            return self._connect_error
        control_mod, receive_mod = resolved

        rate = self._rtde_frequency
        try:
            receive = (
                receive_mod.RTDEReceiveInterface(self._host)
                if rate is None
                else receive_mod.RTDEReceiveInterface(self._host, rate)
            )
        except (OSError, RuntimeError) as exc:
            self._connect_error = f"URDriver: controller at {self._host!r} did not answer RTDE receive: {exc}"
            return self._connect_error

        mode_reason = self._mode_refusal(receive)
        if mode_reason is not None:
            # Not a connection failure: the arm answered, and what it said is
            # that it cannot move. Held as the connect error so ``get_status``
            # reports the controller's own vocabulary rather than a bare
            # "disconnected", and the receive side is released because a caller
            # who cannot command has nothing to poll it for.
            self._release(receive)
            self._connect_error = mode_reason
            return self._connect_error

        try:
            control = (
                control_mod.RTDEControlInterface(self._host)
                if rate is None
                else control_mod.RTDEControlInterface(self._host, rate)
            )
        except (OSError, RuntimeError) as exc:
            self._release(receive)
            self._connect_error = f"URDriver: controller at {self._host!r} did not answer RTDE control: {exc}"
            return self._connect_error

        with self._lock:
            self._receive = receive
            self._control = control
        self._connect_error = None
        self._absorb_state()
        return None

    @staticmethod
    def _release(interface: Any) -> None:
        """Disconnect one RTDE interface, tolerating one already gone."""
        if interface is None:
            return
        try:
            interface.disconnect()
        except (OSError, RuntimeError) as exc:
            logger.debug("URDriver: disconnecting %s raised %s", type(interface).__name__, exc)

    def _mode_refusal(self, receive: Any) -> str | None:
        """Report why the controller cannot move, or ``None`` when it can.

        Args:
            receive: A live RTDE receive interface.

        Returns:
            A reason naming the robot mode or safety mode in the controller's
            own vocabulary, or ``None`` when the arm is running and its safety
            mode is one that executes motion.
        """
        try:
            robot_mode = int(receive.getRobotMode())
            safety_mode = int(receive.getSafetyMode())
        except (OSError, RuntimeError) as exc:
            return f"URDriver: controller at {self._host!r} did not report its mode: {exc}"

        if robot_mode != ROBOT_MODE_RUNNING:
            return (
                f"URDriver: controller at {self._host!r} is in robot mode "
                f"{_mode_name(ROBOT_MODES, robot_mode)}, not {ROBOT_MODES[ROBOT_MODE_RUNNING]}. "
                "Power the arm on and release the brakes from the teach pendant; a command sent now "
                "is accepted by the controller and moves nothing."
            )
        if safety_mode not in SAFETY_MODES_THAT_MOVE:
            return (
                f"URDriver: controller at {self._host!r} is in safety mode "
                f"{_mode_name(SAFETY_MODES, safety_mode)}, which executes no motion. "
                "Clear it from the teach pendant; a command sent now is accepted by the controller "
                "and moves nothing."
            )
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report reachability plus the controller's robot and safety modes."""
        robot_mode: int | None = None
        safety_mode: int | None = None
        with self._lock:
            receive = self._receive
        if receive is not None:
            try:
                robot_mode = int(receive.getRobotMode())
                safety_mode = int(receive.getSafetyMode())
            except (OSError, RuntimeError) as exc:
                logger.debug("URDriver.get_status(): mode read failed: %s", exc)
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "model": self._model,
                        "host": self._host,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        "robot_mode": _mode_name(ROBOT_MODES, robot_mode),
                        "safety_mode": _mode_name(SAFETY_MODES, safety_mode),
                        "task_running": self._rollout is not None and self._rollout.is_running,
                        # Always ``None``: a UR arm runs off mains through its
                        # control box and has no battery to report. The key is
                        # present rather than absent because it is part of the
                        # shape every driver's status carries, and a fleet view
                        # reading peers uniformly should see "no battery" rather
                        # than have to know which drivers omit the field.
                        "battery_pct": None,
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Stop motion, leaving both interfaces connected.

        Annotated ``-> None`` by the driver protocol, so it carries no verdict:
        the rollout is signalled and not waited for, and the loop's own step
        re-reads that signal before its next setpoint, which is what keeps a
        servoJ from landing after this servoStop. A caller that needs the halt
        outcome reads :meth:`stop_task`, which decides one.
        """
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
        with self._lock:
            control = self._control
        if control is None:
            return
        try:
            control.servoStop()
        except (OSError, RuntimeError) as exc:
            logger.warning("%s.stop(): controller refused servoStop: %s", self._tool_name, exc)
        self._drop_anchor()

    def cleanup(self) -> None:
        """Stop the arm and release both interfaces. Idempotent.

        The join outcome is not reported - the protocol annotates this ``-> None``
        - and it does not need to be: the interface handles are cleared under the
        lock before either is disconnected, so a rollout thread that outlasted
        the join finds ``None`` and refuses its own write rather than reaching a
        disconnected interface.
        """
        rollout = self._rollout
        if rollout is not None:
            rollout.request_stop()
            rollout.join()
        self._drop_anchor()
        with self._lock:
            control, receive = self._control, self._receive
            self._control = self._receive = None
        if control is not None:
            try:
                control.servoStop()
            except (OSError, RuntimeError) as exc:
                logger.debug("URDriver.cleanup(): servoStop raised %s", exc)
        self._release(control)
        self._release(receive)

    # ------------------------------------------------------------------ #
    # Command path.                                                      #
    # ------------------------------------------------------------------ #

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Command one joint-space setpoint through ``servoJ``.

        Gates, in order: this driver fronts this robot; both interfaces are
        live; the controller is in a mode that moves; the action names only UR
        joints, every value is finite and within travel, and no joint is asked
        to move further than its speed ceiling allows in one control period.
        Only then is the setpoint written - and the controller's own return
        value decides the verdict, so a rejected write is reported as one.

        Args:
            action: Joint targets in radians, keyed by :data:`JOINT_NAMES`
                members. A joint the action omits holds its measured value.
            robot_name: Which robot to command; ``None`` or this driver's own
                name. Any other name is refused rather than silently applied to
                the arm this driver holds.

        Returns:
            A success envelope carrying the vector that reached the wire, or a
            refusal naming the gate that stopped it.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return _refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        with self._lock:
            control, receive = self._control, self._receive
        if control is None or receive is None:
            return _refuse("send_action: not connected - call connect_eagerly() first")
        if (reason := self._mode_refusal(receive)) is not None:
            # The stream ends here when the halt is the controller's own, and
            # that is the halt after which the arm has most likely been moved:
            # clearing a protective stop from the pendant is when it gets
            # jogged. Drop the anchor exactly as this driver's own halt verbs
            # do, or the next setpoint the mode gate re-admits is sized from a
            # pose the arm no longer holds. See :meth:`_drop_anchor`.
            self._drop_anchor()
            return _refuse(f"send_action: {reason}")

        reference, read_reason = self._reference_pose(receive)
        if read_reason is not None:
            return _refuse(f"send_action: {read_reason}")

        period = 1.0 / self._control_frequency
        targets, reason = targets_from_action(action, reference, model=self._model, control_period=period)
        if reason is not None:
            return _refuse(f"send_action: {reason}")

        try:
            accepted = control.servoJ(
                targets,
                SERVOJ_SPEED,
                SERVOJ_ACCELERATION,
                period,
                SERVOJ_LOOKAHEAD_TIME,
                SERVOJ_GAIN,
            )
        except (OSError, RuntimeError) as exc:
            return _refuse(f"send_action: servoJ failed: {exc}")
        if accepted is False:
            # ur_rtde returns False when the controller declines the setpoint.
            # Reporting success here is the failure mode the mode gate above
            # cannot cover on its own: the arm can pass the gate and still
            # decline a particular write.
            return _refuse(
                f"send_action: the controller declined the servoJ setpoint {targets}. "
                "The arm is running but not tracking; check the teach pendant for an active program."
            )
        with self._lock:
            self._commanded = list(targets)
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "joints": dict(zip(JOINT_NAMES, targets, strict=True)),
                        "targets": targets,
                        "control_period": period,
                    }
                }
            ],
        }

    def state(self) -> dict[str, Any]:
        """Read joints, TCP pose and TCP wrench from the controller.

        The three quantities a UR caller asks for, in one round trip. The
        wrench is the controller's own force-torque estimate at the tool centre
        point, in the base frame: it needs no extra sensor, which is why it
        belongs here rather than behind a separate surface.

        Returns:
            A success envelope carrying ``joints`` (name -> radians),
            ``joint_velocities``, ``tcp_pose`` (x, y, z in metres then an
            axis-angle rotation vector), ``tcp_speed``, ``wrench`` (three forces
            in newtons then three torques in newton-metres) and the two
            controller modes; or a refusal when the arm is not connected.
        """
        with self._lock:
            receive = self._receive
        if receive is None:
            return _refuse("state: not connected - call connect_eagerly() first")
        try:
            joints = [float(value) for value in receive.getActualQ()]
            velocities = [float(value) for value in receive.getActualQd()]
            tcp_pose = [float(value) for value in receive.getActualTCPPose()]
            tcp_speed = [float(value) for value in receive.getActualTCPSpeed()]
            wrench = [float(value) for value in receive.getActualTCPForce()]
            robot_mode = int(receive.getRobotMode())
            safety_mode = int(receive.getSafetyMode())
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _refuse(f"state: the controller's RTDE read failed: {exc}")

        named = dict(zip(JOINT_NAMES, joints, strict=False))
        with self._lock:
            self._joints = dict(named)
            self._pose = {"tcp_pose": tcp_pose, "frame": "base"}
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "model": self._model,
                        "joints": named,
                        "joint_velocities": dict(zip(JOINT_NAMES, velocities, strict=False)),
                        "tcp_pose": tcp_pose,
                        "tcp_speed": tcp_speed,
                        "wrench": wrench,
                        "robot_mode": _mode_name(ROBOT_MODES, robot_mode),
                        "safety_mode": _mode_name(SAFETY_MODES, safety_mode),
                    }
                }
            ],
        }

    def _drop_anchor(self) -> None:
        """Forget the commanded setpoint, re-anchoring the next step on measurement.

        Called from every transition out of "a stream is in progress", whichever
        party ended it: this driver's own halt verbs (:meth:`stop`,
        :meth:`stop_task`, :meth:`cleanup`), a controller-initiated halt that
        :meth:`_mode_refusal` reports out of :meth:`send_action`, and a rollout
        leaving its loop for any reason. The anchor is a safe reference only
        while the setpoints keep coming, because an arm at rest may be moved -
        freedrive, or a manual jog while an operator clears a stop from the
        pendant - and :meth:`_reference_pose` describes what a stale anchor then
        commands.

        Deliberately *not* called when this driver's own value gates refuse a
        setpoint: the stream is intact there and the arm is still tracking toward
        the anchor, so re-anchoring on measurement would charge the next step for
        accumulated tracking lag, which is the refusal storm
        :meth:`_reference_pose` exists to avoid.
        """
        with self._lock:
            self._commanded = None

    def _reference_pose(self, receive: Any) -> tuple[list[float], str | None]:
        """Return the pose the next setpoint's step is measured from.

        The last commanded setpoint while a stream is in progress, and the
        measured pose for the first setpoint after connecting or after a stop.

        The distinction is the difference between a gate that bounds commanded
        velocity and one that bounds tracking error, and only the first is what
        the speed ceiling describes. ``servoJ`` setpoints form a trajectory the
        controller interpolates toward, so it is always some distance behind the
        setpoint it was last given - under load, more so. Measuring each step
        from the *measured* pose therefore charges the commanded step for
        accumulated lag: a policy streaming increments the arm can easily follow
        starts being refused the moment the arm falls a period behind, and the
        motion stalls for a reason the caller cannot act on. Measured in
        simulation at 50 Hz, that gate refused 288 of 420 setpoints from a
        trajectory whose per-step increments were a quarter of the ceiling.

        Anchoring on the last commanded setpoint bounds exactly what the
        datasheet bounds - how far the commanded target may move per period -
        and leaves following error to the controller, which monitors it and
        raises a protective stop the mode gate then reports. Re-anchoring on the
        measured pose once the stream ends matters for the opposite reason: the
        arm may have been moved (freedrive, a manual jog) while the stream was
        down, so resuming from a stale setpoint would command a jump from a pose
        the arm no longer holds. Which party ended the stream does not change
        that - see :meth:`_drop_anchor` for the transitions that drop it.

        Args:
            receive: A live RTDE receive interface.

        Returns:
            ``(reference, reason)`` - six positions in :data:`JOINT_NAMES`
            order, or an empty list and a reason.
        """
        with self._lock:
            commanded = self._commanded
        if commanded is not None:
            return list(commanded), None
        return self._read_joints(receive)

    def _read_joints(self, receive: Any) -> tuple[list[float], str | None]:
        """Read the measured joint vector, or report why it could not be read.

        Args:
            receive: A live RTDE receive interface.

        Returns:
            ``(joints, reason)`` - six positions in :data:`JOINT_NAMES` order,
            or an empty list and a reason.
        """
        try:
            joints = [float(value) for value in receive.getActualQ()]
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return [], f"the controller's joint read failed: {exc}"
        if len(joints) != len(JOINT_NAMES):
            return [], (
                f"the controller reported {len(joints)} joint positions, expected {len(JOINT_NAMES)}. "
                "This driver serves six-axis e-Series arms only."
            )
        return joints, None

    def _absorb_state(self) -> None:
        """Prime the mesh caches from one state read, ignoring a failure."""
        envelope = self.state()
        if envelope.get("status") != "success":
            logger.debug("URDriver: priming state read refused: %s", envelope)

    # ------------------------------------------------------------------ #
    # Mesh telemetry.                                                    #
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict[str, float]:
        """The six joint positions (name -> radians) for the mesh joint read."""
        with self._lock:
            receive = self._receive
        if receive is not None:
            joints, reason = self._read_joints(receive)
            if reason is None:
                with self._lock:
                    self._joints = dict(zip(JOINT_NAMES, joints, strict=True))
        with self._lock:
            return dict(self._joints)

    # ------------------------------------------------------------------ #
    # Task and policy path.                                              #
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
        """Build a policy from the provider registry and roll it out in the background.

        Args:
            instruction: Natural-language instruction handed to the policy.
            policy_port: Port the policy server listens on; ``None`` uses the
                provider's default.
            policy_host: Host the policy server runs on.
            policy_provider: Which provider to build, by registry name.
            duration: Wall-clock budget for the rollout, in seconds.
            **policy_kwargs: Extra provider-specific policy options.

        Returns:
            The envelope :meth:`run_policy` returns for the rollout it started,
            or a refusal naming the provider that could not be built.
        """
        from strands_robots.policies import create_policy

        kwargs: dict[str, Any] = {"host": policy_host, **policy_kwargs}
        if policy_port is not None:
            kwargs["port"] = policy_port
        try:
            policy = create_policy(policy_provider, **kwargs)
        except (ImportError, TypeError, ValueError) as exc:
            return _refuse(f"start_task: could not build the {policy_provider!r} policy: {exc}")
        return self.run_policy(policy, instruction=instruction, duration=duration)

    def run_policy(
        self,
        policy_object: Policy | Callable[[dict[str, Any]], dict[str, Any]],
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Roll out an already-built policy against the arm.

        The loop runs on a dedicated thread so the caller returns immediately;
        poll :meth:`get_task_status` to observe progress. Each step reads the
        arm, asks the policy for an action, and writes it through
        :meth:`send_action` - so every step re-crosses the mode and step gates,
        and a controller that leaves the running set mid-rollout ends the
        rollout with that refusal as its exit reason rather than being
        commanded into a stop it will not perform.

        ``policy_object`` is either a built
        :class:`~strands_robots.policies.Policy` or a bare callable taking an
        observation dict; the admission check accepts ``get_actions_sync``,
        ``step`` or a callable, which is the same set the refusal enforces.

        Args:
            policy_object: The policy to roll out.
            instruction: Natural-language instruction handed to the policy.
            duration: Wall-clock budget in seconds.
            n_steps: Step budget; when given it wins over ``duration``.

        Returns:
            A success envelope describing the rollout that started, or a
            refusal.
        """
        if err := positive_finite_number_error(duration, "duration", "run_policy"):
            return _refuse(err)
        if n_steps is not None and (err := positive_count_error(n_steps, "n_steps", "run_policy")):
            return _refuse(err)
        if policy_object is None:
            return _refuse("run_policy: policy_object is required")
        if not _policy_step(policy_object, instruction):
            return _refuse("run_policy: policy_object must be callable or expose get_actions_sync() or step()")
        if not self.is_connected:
            return _refuse("run_policy: not connected - call connect_eagerly() first")

        rollout = _Rollout(
            driver=self,
            policy=policy_object,
            instruction=instruction,
            duration=float(duration),
            n_steps=n_steps,
            period=1.0 / self._control_frequency,
        )
        # Admission held across the running check, the reference assignment and
        # start() so a second caller cannot pass the check before either
        # assigns ``self._rollout`` - two rollouts streaming setpoints to one
        # controller.
        with self._task_admission:
            if self._rollout is not None and self._rollout.is_running:
                return _refuse("run_policy: a task is already running; call stop_task first")
            self._rollout = rollout
            rollout.start()
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "robot": self._tool_name,
                        "instruction": instruction,
                        "control_frequency": self._control_frequency,
                        "duration": duration,
                        "n_steps": n_steps,
                    }
                }
            ],
        }

    def get_task_status(self) -> dict[str, Any]:
        """Report the rollout's progress, or that none has run."""
        rollout = self._rollout
        if rollout is None:
            return {"status": "success", "content": [{"json": {"running": False, "steps": 0}}]}
        return {"status": "success", "content": [{"json": rollout.snapshot()}]}

    def stop_task(self) -> dict[str, Any]:
        """Stop the rollout and decelerate the arm.

        Returns:
            A success envelope naming what was halted when the rollout thread
            left the loop, a refusal when the controller declined the stop, and
            an *error* envelope carrying ``stopped=False`` when the thread is
            still in the loop - a caller-supplied policy blocking on a remote
            inference call is the ordinary case. The arm is decelerated either
            way; what the third case refuses to do is claim a halt while
            :meth:`get_task_status` would report ``running=True``.
        """
        rollout = self._rollout
        unjoined: _Rollout | None = None
        if rollout is not None and rollout.is_running:
            rollout.request_stop()
            if not rollout.join():
                unjoined = rollout
        halted = 0 if rollout is None else rollout.steps
        with self._lock:
            control = self._control
        if control is None:
            return _refuse("stop_task: not connected")
        try:
            control.servoStop()
        except (OSError, RuntimeError) as exc:
            return _refuse(f"stop_task: the controller refused servoStop: {exc}")
        self._drop_anchor()
        if unjoined is not None:
            # The thread is still in the loop. It will not write another
            # setpoint - the step re-reads the stop event before sending - but it
            # holds the rollout, so reporting success here would hand a caller
            # that reads only ``status`` a halt the payload's own ``running``
            # contradicts.
            snapshot = unjoined.snapshot()
            snapshot["stopped"] = False
            snapshot["robot"] = self._tool_name
            snapshot["reason"] = (
                "stop_task: the rollout thread did not join within its budget; the policy is "
                "likely blocking - the arm is decelerated and the loop writes no further "
                "setpoint, but the task is still holding the arm"
            )
            return {"status": "error", "content": [{"json": snapshot}]}
        return {
            "status": "success",
            "content": [{"json": {"stopped": True, "steps": halted, "robot": self._tool_name}}],
        }


def _policy_step(
    policy_object: Any,
    instruction: str,
) -> Callable[[dict[str, Any]], Any] | None:
    """Return the one-step callable for ``policy_object``, or ``None``.

    Three shapes reach this driver and all three are legitimate: a built
    :class:`~strands_robots.policies.Policy` (``get_actions_sync``), a
    control-loop policy of the shape :mod:`strands_robots.drivers.g1` accepts
    (``step``), and a bare callable. Resolving them once, here, is what lets the
    rollout loop hold a single call site.

    Args:
        policy_object: The candidate policy.
        instruction: Instruction to bind into a ``get_actions_sync`` call, which
            takes it as its second argument.

    Returns:
        A callable taking an observation dict, or ``None`` when the object is
        none of the three shapes.
    """
    get_actions = getattr(policy_object, "get_actions_sync", None)
    if callable(get_actions):
        return lambda observation: get_actions(observation, instruction)
    step = getattr(policy_object, "step", None)
    if callable(step):
        return cast("Callable[[dict[str, Any]], Any]", step)
    if callable(policy_object):
        return cast("Callable[[dict[str, Any]], Any]", policy_object)
    return None


class _Rollout:
    """One policy rollout on its own thread, paced by :class:`~strands_robots.mesh.pacing.Ticker`.

    Holds the loop's counters and exit reason so :meth:`URDriver.get_task_status`
    has one snapshot to report, whether the loop is running, finished its budget
    or was refused mid-way.
    """

    def __init__(
        self,
        *,
        driver: URDriver,
        policy: Any,
        instruction: str,
        duration: float,
        n_steps: int | None,
        period: float,
    ) -> None:
        """Record the rollout's budget; :meth:`start` runs it.

        Args:
            driver: The driver whose arm is commanded.
            policy: The policy object, resolved to a callable by
                :func:`_policy_step`.
            instruction: Instruction handed to the policy each step.
            duration: Wall-clock budget in seconds.
            n_steps: Step budget; when given it wins over ``duration``.
            period: Seconds per step.
        """
        self._driver = driver
        self._policy = policy
        self._instruction = instruction
        self._duration = duration
        self._n_steps = n_steps
        self._period = period
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.steps = 0
        self._exit_reason: str | None = None
        self._refusal: str | None = None

    @property
    def is_running(self) -> bool:
        """Whether the rollout thread is still stepping."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Run the rollout on a daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            name=f"ur-rollout-{self._driver.tool_name}",
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        """Ask the loop to exit at its next tick."""
        self._stop.set()

    def join(self, timeout: float = 2.0) -> bool:
        """Wait for the loop to exit, reporting whether it did.

        Args:
            timeout: Seconds to wait. The loop checks the stop event once per
                period, so a bound a few periods long is enough.

        Returns:
            ``True`` when the thread is out of the loop - which is what makes a
            halt claim true, because the loop cannot write another setpoint once
            it has left. ``False`` when it is still in there: a caller-supplied
            policy blocking on a remote inference call outlasts any join budget,
            and :meth:`URDriver.stop_task` needs that fact rather than a
            ``stopped`` claim its own ``running`` flag contradicts.
        """
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def snapshot(self) -> dict[str, Any]:
        """The counters and exit reason, as one dict for the status envelope."""
        with self._lock:
            return {
                "running": self.is_running,
                "steps": self.steps,
                "exit_reason": self._exit_reason,
                "refusal": self._refusal,
                "instruction": self._instruction,
            }

    def _run(self) -> None:
        """Step the policy until the budget runs out, the arm refuses, or stop."""
        step_fn = _policy_step(self._policy, self._instruction)
        if step_fn is None:  # pragma: no cover - admitted by run_policy
            self._finish("policy")
            return
        deadline = time.monotonic() + self._duration
        with Ticker(self._period, self._stop) as ticker:
            while True:
                if self._stop.is_set():
                    self._finish("stopped")
                    return
                if self._n_steps is not None and self.steps >= self._n_steps:
                    self._finish("n_steps")
                    return
                if self._n_steps is None and time.monotonic() >= deadline:
                    self._finish("duration")
                    return

                observation = self._driver.get_observation()
                try:
                    action = step_fn(observation)
                except Exception as exc:  # noqa: BLE001 - a policy fault ends the rollout, not the thread
                    self._finish("policy", f"the policy raised {type(exc).__name__}: {exc}")
                    return
                if not isinstance(action, dict) or not action:
                    self._finish("policy", f"the policy returned {action!r}, expected a joint-keyed action dict")
                    return

                if self._stop.is_set():
                    # Re-read after the policy returns and before the setpoint
                    # goes out. A policy call is the longest thing in a step, so
                    # a stop signalled during one would otherwise be answered by
                    # one more servoJ - landing after the servoStop the halt
                    # verb just issued, and moving an arm an operator was told
                    # had stopped.
                    self._finish("stopped")
                    return

                envelope = self._driver.send_action(action)
                if envelope.get("status") != "success":
                    self._finish("refused", _envelope_text(envelope))
                    return
                with self._lock:
                    self.steps += 1
                if ticker.wait():
                    self._finish("stopped")
                    return

    def _finish(self, reason: str, refusal: str | None = None) -> None:
        """Record why the loop exited, and drop the driver's step-gate anchor.

        Every exit routes through here, so the anchor is dropped by construction
        rather than per reason: a rollout that ran its budget out leaves the arm
        at rest just as surely as one an operator stopped, and the arm may be
        moved before the next setpoint. Taken outside this object's lock - the
        driver's anchor is behind a different, non-reentrant lock.
        """
        with self._lock:
            self._exit_reason = reason
            self._refusal = refusal
        self._driver._drop_anchor()


def _envelope_text(envelope: dict[str, Any]) -> str:
    """Read the reason out of a refusal envelope, for the rollout snapshot."""
    for block in envelope.get("content") or []:
        text = block.get("text") if isinstance(block, dict) else None
        if text:
            return str(text)
    return "the arm refused the setpoint"
