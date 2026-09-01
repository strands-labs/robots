"""Native CycloneDDS driver for the Unitree Go2 quadruped.

``Robot("go2", mode="real", driver="strands", port=<ip>, network_interface="eth0")``
builds one of these. The instance satisfies
:class:`~strands_robots.drivers.base.HardwareDriver`, so
:func:`~strands_robots.robot.Robot` returns it and the mesh, teleop rail and
agent tool surface consume it exactly like the lerobot driver they replace -
except that lerobot has no robot type for the Go2 at all, which is the gap this
driver closes.

Why this is not :mod:`strands_robots.drivers.g1` with a different joint table
--------------------------------------------------------------------------

The Go2 is a different SDK generation from the G1, and the two differences that
matter are both on the wire:

* **A different IDL package.** The G1 publishes
  ``unitree_sdk2py.idl.unitree_hg.msg.dds_.LowCmd_``; the Go2 publishes
  ``unitree_sdk2py.idl.unitree_go.msg.dds_.LowCmd_``. The two structs are not
  interchangeable: ``unitree_hg``'s carries ``mode_pr`` and ``mode_machine``,
  ``unitree_go``'s carries ``head``, ``level_flag`` and ``gpio`` instead. A
  ``unitree_hg`` frame on a Go2 fails CRC and is dropped silently.
* **A different motor index order.** ``LowCmd_.motor_cmd`` is indexed by
  Unitree's ``LegID`` convention - front-right, front-left, rear-right,
  rear-left - while the Go2's own URDF/MJCF description declares its joints
  front-left, front-right, rear-left, rear-right. The two orders agree on
  nothing but the hip/thigh/calf triple inside each leg, so zipping the
  description's joint order onto ``motor_cmd`` swaps the robot's left and right
  legs. That is a quadruped that walks sideways into a wall with every gain and
  CRC correct, so :data:`GO2_JOINT_INDEX` is keyed by *name* and
  :meth:`Go2Driver.send_action` never accepts an index.

The safety gate is also a different question. The G1 gate reads a high-level FSM
id whose wire key is still unevidenced (see
:mod:`strands_robots.tools.g1._motion_switcher` and issue #2765). The Go2's
low-level write path has a simpler and fully-evidenced precondition: the onboard
sport-mode service must be *released* before ``rt/lowcmd`` reaches the motors,
and every SDK example tests exactly one key for it - ``CheckMode()``'s
``result["name"]``, which is ``""`` when no motion mode holds the robot. So this
driver gates on the one key the SDK evidences and needs no wire guess:
:meth:`Go2Driver.release_sport_mode` performs the release,
:meth:`Go2Driver.send_action` refuses until it is confirmed. Publishing
``rt/lowcmd`` while sport mode still holds the robot means the onboard
controller and the caller fight over the same motors, which is why this is a
refusal and not a warning.

What the driver does:

* Subscribes ``rt/lowstate`` (IMU, per-joint telemetry and the battery state of
  charge, which on the Go2 rides inside ``LowState_`` rather than on its own
  topic as it does on the G1) and ``rt/sportmodestate`` (body pose, velocity and
  the gait the onboard controller reports). Each callback drops into an
  in-memory cache the mesh reads at its own cadence; the DDS callback stays
  fast - parse, drop into a slot, return.
* :meth:`send_action` publishes one ``LowCmd_`` on ``rt/lowcmd`` for
  joint-name-keyed targets, gated on the sport-mode release and the battery
  floor.
* :meth:`run_policy` rolls a caller-built policy on a dedicated 500 Hz thread
  with a per-step re-gate and a zero-torque frame on exit;
  :meth:`stop_task` halts it and reports the join outcome;
  :meth:`get_task_status` reports the loop's snapshot or its last exit reason.

Nothing in this module imports ``unitree_sdk2py`` at module load. Every SDK
touch is inside a function body, so the module imports on Thor, on CI, and in
every unit test with a mocked bus.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, cast

from strands_robots.mesh.pacing import Ticker
from strands_robots.tools.g1._dds_engine import DDSPublisher, DDSSubscriberSet
from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK
from strands_robots.utils import (
    finite_number_error,
    positive_count_error,
    positive_finite_number_error,
)

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The robots this driver registers for, read by
#: :data:`strands_robots.drivers._SHIPPED_DRIVERS`. The Go2's ``go2`` alias
#: resolves through :func:`~strands_robots.registry.resolve_name`, so only the
#: canonical name is listed.
SUPPORTED_ROBOTS: tuple[str, ...] = ("unitree_go2",)

#: Percentage below which the write gate refuses. Same floor as the G1 driver:
#: a quadruped that browns out mid-step falls onto its own hardware.
_BATTERY_FLOOR_PCT: float = 15.0

#: Control-loop cadence. 500 Hz matches the SDK's own Go2 low-level example,
#: which sleeps 0.002 s between ``rt/lowcmd`` writes. Firmware holds the last
#: commanded posture only while frames keep arriving on cadence; a slower loop
#: lets the legs droop between frames. A module constant so a test can retune it
#: without patching a sleep.
_CONTROL_LOOP_HZ: float = 500.0
_CONTROL_LOOP_DT: float = 1.0 / _CONTROL_LOOP_HZ

# The topics the driver reads.
_TOPIC_LOWSTATE = "rt/lowstate"
_TOPIC_SPORTMODE = "rt/sportmodestate"

#: The topic the driver writes. A full ``LowCmd_`` shaped for the Go2's leg
#: actuator set; motion cannot go anywhere else without also crossing the
#: sport-mode release, so the write path is a single-topic path.
_TOPIC_LOWCMD = "rt/lowcmd"

#: ``LowCmd_.motor_cmd`` is a fixed-length array of 20 ``MotorCmd_`` on the
#: ``unitree_go`` IDL. The Go2 drives 12 of them; the remaining slots stay at
#: their zero default, which is ``mode = 0`` (Disable) and therefore commands
#: nothing.
_GO2_MOTOR_SLOTS: int = 20

#: The low-level frame header the ``unitree_go`` protocol requires. The SDK's
#: Go2 low-level example sets both bytes explicitly on every frame rather than
#: relying on the default constructor, and a frame with the wrong header is
#: dropped before CRC is even considered.
_LOWCMD_HEAD: tuple[int, int] = (0xFE, 0xEF)

#: ``level_flag`` selecting low-level (direct motor) control. The Go2 rejects a
#: ``rt/lowcmd`` frame that does not claim low level.
_LEVEL_FLAG_LOW: int = 0xFF

#: ``MotorCmd_.mode`` enabling the servo. Unset (``0`` = Disable) a frame with a
#: perfectly valid CRC still commands nothing, which is the most confusing way
#: for a driver to appear broken.
_MOTOR_MODE_SERVO: int = 0x01

#: The key ``MotionSwitcherClient.CheckMode()`` reports the active motion mode
#: under. The only key the SDK evidences - every example reads
#: ``result["name"]`` and nothing else.
_RESULT_NAME_KEY = "name"

#: Per-leg reference gains, ordered ``(hip, thigh, calf)``. Taken from the SDK's
#: own Go2 low-level position-control example, which holds every joint at
#: ``kp = 60, kd = 5``. Kept as a per-leg triple rather than a flat 12 so a
#: future per-joint retune has an obvious shape to land in.
_LEG_KP: tuple[float, float, float] = (60.0, 60.0, 60.0)
_LEG_KD: tuple[float, float, float] = (5.0, 5.0, 5.0)

#: Reference gains indexed by wire slot, so ``_SDK_KP[GO2_JOINT_INDEX[name]]``
#: is the default gain for a joint.
_SDK_KP: tuple[float, ...] = _LEG_KP * 4
_SDK_KD: tuple[float, ...] = _LEG_KD * 4
# The per-joint fields a caller may put on a ``LowCmd_`` motor slot, and the
# whole set this module reads off an action dict. Every one of them is a
# physical quantity the frame carries to the motor controller verbatim, so each
# is held to :func:`~strands_robots.utils.finite_number_error` before it is
# written: a ``nan`` target serializes as a valid IEEE-754 float and the
# controller integrates it, which poisons the pose rather than refusing the
# command. Single-sourced so the fields accepted and the fields checked cannot
# drift apart.
_WIRE_FIELDS: tuple[str, ...] = ("q", "kp", "kd", "dq", "tau")

#: Joint name -> ``LowCmd_.motor_cmd`` slot.
#:
#: The names are the ones the Go2's official URDF/MJCF description declares, so
#: a caller reading joint names off the model commands the joint they named. The
#: *slots* are Unitree's ``LegID`` order - FR, FL, RR, RL - which is **not** the
#: description's declaration order (FL, FR, RL, RR). Front-left and front-right
#: are transposed between the two, as are the rear pair, so this table is the
#: one place the two conventions are reconciled and the reason
#: :meth:`Go2Driver.send_action` is keyed by name.
GO2_JOINT_INDEX: dict[str, int] = {
    "FR_hip_joint": 0,
    "FR_thigh_joint": 1,
    "FR_calf_joint": 2,
    "FL_hip_joint": 3,
    "FL_thigh_joint": 4,
    "FL_calf_joint": 5,
    "RR_hip_joint": 6,
    "RR_thigh_joint": 7,
    "RR_calf_joint": 8,
    "RL_hip_joint": 9,
    "RL_thigh_joint": 10,
    "RL_calf_joint": 11,
}


def _refuse(reason: str) -> dict[str, Any]:
    """Return the driver's error envelope with ``reason`` inside.

    A free function so every refusal path renders the same shape and a test can
    grep for the reason without unpacking the envelope by hand.
    """
    return {"status": "error", "content": [{"text": reason}]}


def _resolve_message_class(cls_path: tuple[str, str]) -> Any:
    """Return the IDL class for ``(module_path, class_name)``, or a reason string.

    Lazy import so the driver module stays importable without the SDK. Called
    from :meth:`Go2Driver.connect_eagerly`, which turns a returned string into a
    named connect failure and leaves the driver in the "usable but not
    connected" state.

    Args:
        cls_path: The IDL module path and the class name inside it.

    Returns:
        The resolved class, or a string naming why it could not be resolved.
    """
    module_path, class_name = cls_path
    try:
        import importlib

        module = importlib.import_module(module_path)
    except ImportError as exc:
        return f"cannot import {module_path}: {exc}"
    if not hasattr(module, class_name):
        return f"{module_path} has no {class_name}"
    return getattr(module, class_name)


def decode_mode_name(check_mode_return: Any) -> tuple[str | None, str | None]:
    """Decode ``MotionSwitcherClient.CheckMode()`` into an active mode name.

    The Go2's low-level write path needs one fact from the motion switcher: is
    any motion mode still holding the robot? The SDK answers with
    ``(status, result)`` where ``result[_RESULT_NAME_KEY]`` is ``""`` for "no
    mode selected" and a label such as ``"ai"`` or ``"normal"`` otherwise. That
    key is the only one the SDK's own examples read, so decoding it needs no
    guess about the rest of the payload - unlike the G1's integer FSM id, whose
    key is still unevidenced (issue #2765).

    Args:
        check_mode_return: Whatever ``CheckMode()`` returned.

    Returns:
        ``(mode_name, None)`` on a decodable reading - ``mode_name`` is ``""``
        when the robot is free. ``(None, reason)`` when the reading cannot be
        trusted, so a caller refuses rather than treating an undecodable
        response as "released".
    """
    if not isinstance(check_mode_return, (tuple, list)) or len(check_mode_return) != 2:
        return None, (
            "CheckMode() must return a (status, result) pair; got "
            f"{type(check_mode_return).__name__}"
            + (f" of length {len(check_mode_return)}" if isinstance(check_mode_return, (tuple, list)) else "")
        )
    status, result = check_mode_return
    if isinstance(status, bool) or not isinstance(status, int):
        return None, f"CheckMode() status must be an int response code; got {type(status).__name__}"
    if status != 0:
        return None, f"CheckMode() failed: {status}"
    if not isinstance(result, dict):
        return None, f"CheckMode() result must be a dict; got {type(result).__name__}"
    if _RESULT_NAME_KEY not in result:
        return None, f"CheckMode() result has no {_RESULT_NAME_KEY!r} key"
    mode_name = result[_RESULT_NAME_KEY]
    if not isinstance(mode_name, str):
        return None, f"CheckMode() result {_RESULT_NAME_KEY!r} must be a string; got {type(mode_name).__name__}"
    return mode_name, None


def _new_lowcmd() -> tuple[Any, str | None]:
    """Build an empty Go2 ``LowCmd_`` with the protocol header already set.

    Both write paths - :func:`build_lowcmd_from_action` and
    :func:`build_zero_torque_lowcmd` - start here, so the header contract is
    written once. The SDK's default constructor does not set ``head`` or
    ``level_flag``, and the SDK's own Go2 example sets both on every frame; a
    frame without them is dropped before CRC is considered, which looks exactly
    like a driver that is publishing to the wrong topic.

    Returns:
        ``(cmd, None)`` on success, ``(None, reason)`` when the SDK is absent.
    """
    try:
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as _default_lowcmd
    except ImportError as exc:  # pragma: no cover - exercised on hardware
        return None, f"unitree_sdk2py is not installed: {exc}"
    cmd = _default_lowcmd()
    # The array length is part of the wire contract, so it is checked rather
    # than assumed: an SDK whose ``motor_cmd`` is shorter than the slots this
    # driver addresses would otherwise raise IndexError from inside the frame
    # builder, several frames into a rollout, instead of naming the mismatch.
    slots = len(cmd.motor_cmd)
    if slots != _GO2_MOTOR_SLOTS:
        return None, (
            f"unitree_go LowCmd_.motor_cmd has {slots} slots, expected {_GO2_MOTOR_SLOTS}; "
            "the installed unitree_sdk2py IDL does not match the Go2 wire format this driver writes"
        )
    cmd.head[0], cmd.head[1] = _LOWCMD_HEAD
    cmd.level_flag = _LEVEL_FLAG_LOW
    cmd.gpio = 0
    return cmd, None


def _seal(cmd: Any) -> str | None:
    """Stamp ``cmd`` with its CRC. The last write before a frame is published.

    Firmware silently drops a frame whose CRC does not match, so this runs after
    every other field is populated and nothing may write to ``cmd`` afterwards.

    Args:
        cmd: The fully-populated ``LowCmd_``.

    Returns:
        ``None`` on success, or a reason when the SDK is absent.
    """
    try:
        from unitree_sdk2py.utils.crc import CRC as _CRC
    except ImportError as exc:  # pragma: no cover - exercised on hardware
        return f"unitree_sdk2py is not installed: {exc}"
    cmd.crc = _CRC().Crc(cmd)
    return None


def build_lowcmd_from_action(action: dict[str, Any]) -> tuple[Any, str | None]:
    """Build a Go2 ``LowCmd_`` from a caller's :meth:`Go2Driver.send_action` dict.

    A free function so a test can walk the mapping without a driver instance,
    and so :meth:`Go2Driver.send_action` reads as "gate, build, publish".

    The mapping is:

    * Every joint name in ``action`` must be a key of :data:`GO2_JOINT_INDEX`.
      An unknown name refuses the whole action - the alternative is to silently
      drop a joint the caller believed was commanded, which is the worst
      failure mode on a legged robot.
    * A scalar value is the position target ``q``, taking the slot's
      :data:`_SDK_KP` / :data:`_SDK_KD` gains with zero ``dq`` and ``tau``.
    * A dict value must carry ``"q"``; ``"kp"``, ``"kd"``, ``"dq"`` and
      ``"tau"`` are optional. An unknown inner key is refused for the same
      reason an unknown joint name is.
    * Every field a caller supplies is held to
      :func:`~strands_robots.utils.finite_number_error` before it is written -
      the same domain the other native drivers put their action values through.
      A ``nan`` or ``inf`` survives a bare ``float()`` and serializes onto the
      wire as a valid IEEE-754 float, so the motor controller integrates it
      instead of rejecting it; ``True`` would land as a silent ``1.0`` rad. This
      is not a magnitude limit - it is the gate that keeps an unrepresentable
      target off the wire, and refusing the whole action is the same posture an
      unknown joint name gets, for the same reason.

    Wire-frame contract: the header and ``level_flag`` come from
    :func:`_new_lowcmd`; ``motor_cmd[i].mode`` is set to
    :data:`_MOTOR_MODE_SERVO` on every commanded slot, because an unset mode
    byte commands nothing however valid the CRC; untouched slots keep their zero
    default and stay disabled; and :func:`_seal` writes the CRC last.

    Args:
        action: Joint-name-keyed targets.

    Returns:
        ``(cmd, None)`` on success, ``(None, reason)`` when the action dict is
        unusable or the SDK is absent.
    """
    if not isinstance(action, dict):
        return None, f"action must be a dict, got {type(action).__name__}"
    if not action:
        return None, "action is empty; nothing to command"
    cmd, err = _new_lowcmd()
    if err is not None:
        return None, err
    known_inner = set(_WIRE_FIELDS)
    for name, value in action.items():
        slot = GO2_JOINT_INDEX.get(name)
        if slot is None:
            allowed = ", ".join(sorted(GO2_JOINT_INDEX))
            return None, f"unknown joint name {name!r}; expected one of: {allowed}"
        if isinstance(value, dict):
            unknown_inner = set(value) - known_inner
            if unknown_inner:
                return None, (
                    f"unknown per-joint keys for {name!r}: "
                    f"{sorted(unknown_inner)}; expected a subset of {sorted(known_inner)}"
                )
            if "q" not in value:
                return None, f"per-joint dict for {name!r} is missing required key 'q'"
            q = value["q"]
            kp = value.get("kp", _SDK_KP[slot])
            kd = value.get("kd", _SDK_KD[slot])
            dq = value.get("dq", 0.0)
            tau = value.get("tau", 0.0)
            supplied = {key: value[key] for key in _WIRE_FIELDS if key in value}
        else:
            q, kp, kd, dq, tau = value, _SDK_KP[slot], _SDK_KD[slot], 0.0, 0.0
            supplied = {"q": value}
        for key, raw in supplied.items():
            reason = finite_number_error(raw, f"{name}.{key}", "send_action")
            if reason is not None:
                return None, reason
        q_f, kp_f, kd_f = float(q), float(kp), float(kd)
        dq_f, tau_f = float(dq), float(tau)
        motor = cmd.motor_cmd[slot]
        motor.mode = _MOTOR_MODE_SERVO
        motor.q = q_f
        motor.dq = dq_f
        motor.tau = tau_f
        motor.kp = kp_f
        motor.kd = kd_f
    if (err := _seal(cmd)) is not None:
        return None, err
    return cmd, None


def build_zero_torque_lowcmd() -> tuple[Any, str | None]:
    """Return a Go2 ``LowCmd_`` with every gain and effort zeroed.

    A zero-kp/kd/tau motor holds no position and applies no torque - the softest
    frame the protocol accepts, and what the control loop publishes on the way
    out. The enable byte is still set on the twelve driven slots: a *Disable*
    frame cuts the motors dead and drops the robot onto its knees, whereas an
    enabled zero-gain frame lets it settle under its own weight.

    Returns:
        ``(cmd, None)`` on success, ``(None, reason)`` when the SDK is absent.
    """
    cmd, err = _new_lowcmd()
    if err is not None:
        return None, err
    for slot in GO2_JOINT_INDEX.values():
        motor = cmd.motor_cmd[slot]
        motor.mode = _MOTOR_MODE_SERVO
        motor.q = 0.0
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
    if (err := _seal(cmd)) is not None:
        return None, err
    return cmd, None


class Go2Driver:
    """Native CycloneDDS driver for one Unitree Go2. See the module docstring.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver` structurally -
    no import of or inheritance from anything there.
    """

    def __init__(
        self,
        tool_name: str = "go2",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        network_interface: str = "eth0",
        battery_floor_pct: float = _BATTERY_FLOOR_PCT,
        motion_switcher_client_factory: Callable[[str], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` does the DDS work.

        The three positional arguments are the ones every native driver takes -
        see :mod:`strands_robots.drivers.base`'s constructor contract - so the
        factory can build any driver the same way.

        Args:
            tool_name: Name the agent invokes the driver by. Also the mesh peer
                id when the driver is wrapped by
                :class:`~strands_robots.mesh.Mesh`.
            cameras: Accepted for parity with the lerobot driver; unused,
                because the Go2's onboard cameras are addressed over the DDS
                bus rather than v4l2.
            data_config: Accepted for parity; unused.
            port: The robot's IP address. Recorded but not dialled: CycloneDDS
                binds to a NIC, not an address. Kept for logging and for
                SSH-side helpers.
            network_interface: The interface CycloneDDS binds to.
            battery_floor_pct: Percentage below which :meth:`send_action`
                refuses to write. Separate from the sport-mode gate so a caller
                can see which check refused.
            motion_switcher_client_factory: Callable taking the network
                interface and returning an open ``MotionSwitcherClient``.
                Injected so a unit test can hand in a recording double without
                patching the SDK module - the factory only has to return an
                object with callable ``CheckMode`` and ``ReleaseMode``
                attributes. ``None`` selects the default lazy loader, which
                imports the client on first use, preserving module-load
                hygiene. Keyword-only so it cannot collide with the positional
                set the driver-base contract fixes.
            **kwargs: Ignored; accepted so the factory can forward extras
                without the driver knowing what they are.

        Raises:
            ValueError: If ``battery_floor_pct`` is not a finite number. A
                ``nan`` floor would compare False against every reading, so the
                driver would report a floor in :meth:`get_status` and enforce
                nothing.
        """
        del cameras, data_config  # accepted for parity; unused here
        if kwargs:
            logger.debug("Go2Driver ignoring extra kwargs: %s", sorted(kwargs))
        if err := finite_number_error(battery_floor_pct, "battery_floor_pct", "Go2Driver"):
            raise ValueError(err)
        self._tool_name = tool_name
        self._port = port
        self._network_interface = network_interface
        self._battery_floor_pct = float(battery_floor_pct)
        self._motion_switcher_client_factory = motion_switcher_client_factory

        self._subs: DDSSubscriberSet | None = None
        self._pubs: DDSPublisher | None = None
        self._connected = False
        self._connect_error: str | None = None

        # Sensor caches. Each is replaced wholesale by its DDS callback, so a
        # reader either sees the previous dict or the next one and never a
        # half-written one.
        self._imu: dict[str, Any] | None = None
        self._battery: dict[str, Any] | None = None
        self._joints: dict[str, Any] | None = None
        self._sport: dict[str, Any] | None = None

        # Sport-mode release state. ``None`` means "never asked".
        self._sport_mode_released: bool = False
        self._sport_mode_name: str | None = None
        self._sport_mode_refusal: str | None = None
        self._sport_mode_client_error: str | None = None
        self._msc: Any | None = None

        # Task path.
        self._task_admission = threading.RLock()
        self._loop: _ControlLoop | None = None
        self._last_task_snapshot: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Agent tool surface.                                                #
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
        """A minimal agent-facing spec: read-only verbs plus a controlled stop.

        ``sensors`` and ``status`` are reads; ``stop`` delegates to
        :meth:`stop_task`, which halts any rollout, lets the loop publish its
        zero-torque frame and reports whether the thread actually joined. Gait
        and locomotion verbs are a separate surface - this is the transport.
        """
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Unitree Go2 native driver: reads the Go2's CycloneDDS bus for IMU, "
                    "battery, per-joint telemetry and sport-mode body state; writes "
                    "joint-name-keyed low-level commands once sport mode is released."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "sensors: return the latest cached IMU/battery/joints/sport state; "
                                    "status: report connection, sport-mode release and battery; "
                                    "stop: halt any running control loop - it publishes a zero-torque "
                                    "frame on exit - and report whether it joined"
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
            tool_use: The agent's tool-use block; ``input.action`` selects the
                verb and defaults to ``"sensors"``.
            invocation_state: Unused; part of the tool protocol.
            **kwargs: Unused; forward-compatibility only.

        Yields:
            One tool-result envelope carrying the requested verb's payload.
        """
        del kwargs, invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        action = (tool_use.get("input") or {}).get("action", "sensors")
        if action == "sensors":
            envelope: dict[str, Any] = {"status": "success", "content": [{"json": self.state}]}
        elif action == "status":
            envelope = {"status": "success", "content": [{"json": await self.get_status()}]}
        else:  # "stop"
            # ``stop_task`` already decides the verdict, including the join
            # timeout, so the verb returns that envelope rather than
            # re-deriving one that could read "success" over a loop still
            # holding the wire.
            envelope = self.stop_task()
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Lifecycle and status.                                              #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> dict[str, Any]:
        """The driver's cached telemetry: IMU, battery, joints and sport state.

        A copy per call, so a caller iterating one cache cannot see a DDS
        callback replace it mid-read. ``None`` for a topic that has not
        delivered a message yet - a driver that just connected reports what it
        has rather than inventing zeros.
        """
        return {
            "imu": self._snapshot("_imu"),
            "battery": self._snapshot("_battery"),
            "joints": self._snapshot("_joints"),
            "sport": self._snapshot("_sport"),
            "sport_mode_released": self._sport_mode_released,
            "sport_mode_name": self._sport_mode_name,
        }

    def _snapshot(self, attr: str) -> dict[str, Any] | None:
        """Return a shallow copy of cache ``attr``, or ``None`` if unfilled."""
        cached = getattr(self, attr, None)
        return None if cached is None else dict(cached)

    def _subscription_plan(self) -> list[tuple[str, tuple[str, str], Any]]:
        """Return ``(topic, (idl_module, idl_class), decoder)`` for every topic.

        A method rather than a constant so a test can read the plan without
        constructing DDS, and so the IDL package this driver uses -
        ``unitree_go``, not the G1's ``unitree_hg`` - is stated in exactly one
        place.
        """
        idl = "unitree_sdk2py.idl.unitree_go.msg.dds_"
        return [
            (_TOPIC_LOWSTATE, (idl, "LowState_"), self._on_lowstate),
            (_TOPIC_SPORTMODE, (idl, "SportModeState_"), self._on_sportmode),
        ]

    def _abort_connect(self, subs: DDSSubscriberSet, reason: str) -> str:
        """Release a partially built subscriber set and record why.

        Args:
            subs: The subscriber set to close.
            reason: Why the connect failed.

        Returns:
            ``reason``, so a caller can ``return self._abort_connect(...)``.
        """
        try:
            subs.close()
        except Exception:  # noqa: BLE001 - teardown must not mask the reason
            logger.debug("%s: closing a partial subscriber set failed", self._tool_name, exc_info=True)
        self._connect_error = reason
        self._connected = False
        return reason

    def connect_eagerly(self) -> str | None:
        """Attach to the DDS bus and subscribe every sensor topic. Idempotent.

        The factory only constructs the driver; whoever performs the bring-up
        calls this, so a real bring-up fails here rather than on the first
        :meth:`get_status` poll. A second call on a connected driver is a no-op
        success - rebuilding the subscriber set would re-subscribe every topic
        and drop the only reference to the previous one, leaking its subscribers
        on a bus whose bindings segfault under concurrent construction.

        This does **not** release sport mode. Connecting is a read; releasing
        the onboard controller hands the robot's legs to whatever writes next,
        so it is a separate, explicit call - see :meth:`release_sport_mode`.

        Returns:
            ``None`` on success and on a call against an already-connected
            driver; a named reason on failure, leaving the driver disconnected
            but usable so a mesh peer for a robot that is off can still exist.
        """
        if self._connected:
            logger.debug("%s already connected; connect_eagerly() is a no-op", self._tool_name)
            return None
        subs = DDSSubscriberSet(self._network_interface)
        err = subs.start()
        if err is not None:
            return self._abort_connect(subs, err)
        for topic, cls_path, decoder in self._subscription_plan():
            message_class = _resolve_message_class(cls_path)
            if isinstance(message_class, str):
                return self._abort_connect(subs, message_class)
            err = subs.subscribe(topic, message_class, decoder)
            if err is not None:
                return self._abort_connect(subs, err)
        pubs = DDSPublisher(self._network_interface)
        err = pubs.start()
        if err is not None:
            # Readers are up, the writer failed. Roll back so the driver
            # reports one connect failure rather than a half-open state.
            return self._abort_connect(subs, err)
        self._pubs = pubs
        self._subs = subs
        self._connected = True
        self._connect_error = None
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report the driver's connection, gate and battery state.

        The shape matches the lerobot driver's ``get_status`` envelope so the
        mesh publishes both peers identically.

        Returns:
            A success envelope whose payload names the connection state, the
            sport-mode gate's inputs and the battery reading behind the floor.
        """
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self._connected,
                        "connect_error": self._connect_error,
                        "port": self._port,
                        "network_interface": self._network_interface,
                        "battery_pct": (self._battery or {}).get("pct"),
                        "battery_floor_pct": self._battery_floor_pct,
                        # Gate diagnostics. ``sport_mode_released`` is what the
                        # write gate reads; the other two say why it is not set,
                        # which is the question a caller asks next.
                        "sport_mode_released": self._sport_mode_released,
                        "sport_mode_name": self._sport_mode_name,
                        "sport_mode_refusal": self._sport_mode_refusal,
                        "sport_mode_client_error": self._sport_mode_client_error,
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Stop a running control loop. The mesh calls this during shutdown.

        A running rollout is signalled, publishes its zero-torque frame and is
        joined - a controlled stop rather than abrupt frame cessation on a robot
        standing on twelve position-controlled joints. Idempotent. A loop that
        outlasts the join budget is logged rather than waited on: this signature
        carries no envelope, and returning from shutdown while a thread still
        holds the wire is exactly what the log exists to say. The publisher is
        left open either way so that loop's zero-torque frame still reaches the
        wire when its policy finally returns.
        """
        with self._task_admission:
            loop = self._loop
        if loop is not None and loop.is_running and not loop.stop("stop"):
            logger.error(
                "%s.stop(): control loop did not join within the stop budget and still "
                "holds the wire; its zero-torque frame publishes when the policy returns",
                self._tool_name,
            )

    def cleanup(self) -> None:
        """Release every DDS endpoint. Idempotent.

        Halts a running loop first so its zero-torque frame goes out on a
        publisher that still exists. Halting is not the same as having halted:
        a caller-supplied policy that outlasts the join budget leaves the loop
        running, so the publisher is released only once the loop is provably
        gone. Otherwise the loop's own zero-torque frame would be dropped by
        the very teardown meant to make it safe.
        """
        with self._task_admission:
            loop = self._loop
        joined = True
        if loop is not None and loop.is_running:
            joined = loop.stop("cleanup")
        if joined and self._pubs is not None:
            try:
                self._pubs.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("%s: closing the publisher failed", self._tool_name, exc_info=True)
            self._pubs = None
        elif not joined:
            logger.error(
                "%s.cleanup(): control loop still running; keeping the publisher so its "
                "zero-torque frame can still reach the wire",
                self._tool_name,
            )
        if self._subs is not None:
            try:
                self._subs.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("%s: closing the subscriber set failed", self._tool_name, exc_info=True)
            self._subs = None
        self._connected = False

    # ------------------------------------------------------------------ #
    # The sport-mode gate. See the module docstring for why it exists.   #
    # ------------------------------------------------------------------ #

    def _open_motion_switcher_client(self) -> Any | None:
        """Return an open ``MotionSwitcherClient``, or ``None`` if it cannot open.

        Cached after the first success. The injected factory wins over the SDK
        import so a test never needs the SDK on the box. A failure is recorded in
        :attr:`_sport_mode_client_error` and surfaced by :meth:`get_status`
        rather than raised: the driver stays usable for reads, and the write gate
        refuses on its own terms.

        The client is opened under
        :data:`~strands_robots.tools.g1._g1_common._DDS_INIT_LOCK`. ``Init()``
        builds the client's DDS request/response endpoints, and the CycloneDDS
        bindings segfault when an endpoint is constructed concurrently with
        another - which this driver does on its own threads, because
        :class:`~strands_robots.tools.g1._dds_engine.DDSSubscriberSet` creates
        every subscriber under that same lock. A segfault is not catchable by the
        "record the error and stay usable for reads" boundary above: the process
        dies, possibly while the robot stands under its own controller.

        The lock covers the injected-factory branch too. The driver cannot know
        whether a factory builds a real client, and one that does owes the same
        serialisation; a factory that instead reaches back through the engine
        would deadlock, since the shared lock is not reentrant, so a factory must
        construct its client directly.
        """
        if self._msc is not None:
            return self._msc
        factory = self._motion_switcher_client_factory
        try:
            if factory is None:
                from strands_robots.tools.g1._motion_switcher import _load_motion_switcher_client

                # The import stays outside the lock: it creates no endpoint, and
                # holding the shared lock across a lazy SDK import would stall
                # every subscriber construction in the process for its duration.
                msc_class = _load_motion_switcher_client()
                with _DDS_INIT_LOCK:
                    client = msc_class()
                    client.SetTimeout(3.0)
                    client.Init()
            else:
                with _DDS_INIT_LOCK:
                    client = factory(self._network_interface)
        except Exception as exc:  # noqa: BLE001 - any SDK/transport failure is one reason
            self._sport_mode_client_error = f"cannot open MotionSwitcherClient: {exc}"
            logger.debug("%s: %s", self._tool_name, self._sport_mode_client_error, exc_info=True)
            return None
        self._sport_mode_client_error = None
        self._msc = client
        return client

    def release_sport_mode(self, attempts: int = 5) -> dict[str, Any]:
        """Release the onboard sport-mode service so ``rt/lowcmd`` reaches the motors.

        Until this succeeds the Go2's own controller owns the legs, and a
        low-level frame published alongside it makes two controllers fight over
        twelve motors. The SDK's examples release in a loop - ``ReleaseMode()``
        then re-read ``CheckMode()`` until the reported mode name is empty -
        because the release is asynchronous, so this polls rather than trusting a
        single call.

        Explicit rather than folded into :meth:`connect_eagerly`: it changes who
        controls the robot, which is not a side effect of connecting to read.

        Args:
            attempts: How many release-then-verify rounds to try before giving
                up. Must be a positive count.

        Returns:
            A success envelope naming the released mode when the robot reports no
            active mode, or an error envelope naming why the gate stays shut.
        """
        if err := positive_count_error(attempts, "attempts", "release_sport_mode"):
            return _refuse(err)
        client = self._open_motion_switcher_client()
        if client is None:
            return _refuse(self._sport_mode_client_error or "MotionSwitcherClient is unavailable")
        previous: str | None = None
        for _ in range(int(attempts)):
            mode_name, refusal = self._read_mode_name(client)
            if refusal is not None:
                return _refuse(refusal)
            if mode_name == "":
                self._sport_mode_released = True
                self._sport_mode_refusal = None
                return {
                    "status": "success",
                    "content": [
                        {
                            "json": {
                                "sport_mode_released": True,
                                "released_mode": previous,
                                "tool_name": self._tool_name,
                            }
                        }
                    ],
                }
            previous = mode_name
            try:
                client.ReleaseMode()
            except Exception as exc:  # noqa: BLE001 - any transport failure is one reason
                reason = f"ReleaseMode() failed while releasing {mode_name!r}: {exc}"
                self._sport_mode_refusal = reason
                return _refuse(reason)
        reason = f"sport mode {previous!r} still active after {int(attempts)} release attempts"
        self._sport_mode_refusal = reason
        return _refuse(reason)

    def _read_mode_name(self, client: Any) -> tuple[str | None, str | None]:
        """Read and decode the active motion mode, recording what was seen.

        Args:
            client: An open motion-switcher client.

        Returns:
            :func:`decode_mode_name`'s ``(mode_name, refusal)`` pair. A refusal
            clears :attr:`_sport_mode_released`, because a reading that cannot be
            decoded is not evidence that the robot is free.
        """
        try:
            reading = client.CheckMode()
        except Exception as exc:  # noqa: BLE001 - any transport failure is one reason
            reason = f"CheckMode() failed: {exc}"
            self._sport_mode_refusal = reason
            self._sport_mode_released = False
            return None, reason
        mode_name, refusal = decode_mode_name(reading)
        self._sport_mode_name = mode_name
        self._sport_mode_refusal = refusal
        if refusal is not None:
            self._sport_mode_released = False
        return mode_name, refusal

    def _check_motion_gates(self, scope: str) -> dict[str, Any] | None:
        """Report why a write must not happen, or ``None`` when it may.

        Two independent gates, each named separately so a caller sees which one
        refused:

        * The sport-mode release - the Go2's own controller must not be holding
          the legs. Unlike the G1's FSM gate this reads a cached boolean rather
          than performing a DDS round trip, so it is safe to call from the
          control-loop thread at 500 Hz and needs no off-thread refresher.
        * The battery floor, read from the ``rt/lowstate`` cache.

        Args:
            scope: What is being gated, quoted in the refusal so a caller can
                tell a ``send_action`` refusal from a rollout's per-step one.

        Returns:
            An error envelope, or ``None`` when the write may proceed.
        """
        if not self._sport_mode_released:
            return _refuse(
                f"{scope} refused: sport mode is not released"
                + (f" (active mode {self._sport_mode_name!r})" if self._sport_mode_name else "")
                + (f"; {self._sport_mode_refusal}" if self._sport_mode_refusal else "")
                + " - call release_sport_mode() first, or the onboard controller and this "
                "driver fight over the same motors"
            )
        battery_pct = (self._battery or {}).get("pct")
        if battery_pct is not None and battery_pct < self._battery_floor_pct:
            return _refuse(f"{scope} refused: battery {battery_pct:.1f}% is under floor {self._battery_floor_pct:.1f}%")
        return None

    # ------------------------------------------------------------------ #
    # Write path.                                                        #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Publish one ``LowCmd_`` on ``rt/lowcmd`` for the given joints.

        The action dict is keyed by joint name - see :data:`GO2_JOINT_INDEX` for
        the exact set, which is the Go2 description's own naming. A caller
        supplies either ``{joint_name: target_position_radians}`` to take the
        reference gains, or ``{joint_name: {"q": ..., "kp": ..., "kd": ...,
        "dq": ..., "tau": ...}}`` for per-joint control; a missing ``q`` refuses
        the whole action so a silently-zeroed target cannot reach the wire, and
        every field the caller does supply must be a finite number for the same
        reason - see :func:`build_lowcmd_from_action`.

        Two things this deliberately is not: a control loop (a caller wanting
        500 Hz calls this on their own timer, or uses :meth:`run_policy`), and a
        command-magnitude filter (the gates are the safety envelope). Requiring
        a field to be finite is not a magnitude limit: ``nan`` names no pose.

        Args:
            action: Joint-name-keyed targets.
            robot_name: Ignored - this driver fronts exactly one Go2. Accepted
                so the call shape matches the lerobot driver's.

        Returns:
            A success envelope naming the topic and the commanded joints, or an
            error envelope naming the gate, the mapping problem or the transport
            failure that stopped it.
        """
        del robot_name  # driver fronts one Go2
        refusal = self._check_motion_gates("send_action")
        if refusal is not None:
            return refusal
        if self._pubs is None:
            return _refuse("publisher not initialised - call connect_eagerly() first")
        cmd, err = build_lowcmd_from_action(action)
        if err is not None:
            return _refuse(err)
        try:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
        except ImportError as exc:  # pragma: no cover - exercised on hardware
            return _refuse(f"unitree_sdk2py is not installed: {exc}")
        pub_err = self._pubs.publish(_TOPIC_LOWCMD, LowCmd_, cmd)
        if pub_err is not None:
            return _refuse(pub_err)
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "topic": _TOPIC_LOWCMD,
                        "joints": sorted(action.keys()),
                        "slots": sorted(GO2_JOINT_INDEX[name] for name in action),
                        "sport_mode_released": self._sport_mode_released,
                    }
                }
            ],
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
        """Start a policy-driven task by provider name. Refuses today.

        The lerobot driver resolves ``start_task`` through the policy provider
        registry in :mod:`strands_robots.policies`. This driver's role is the
        transport primitive - a gated 500 Hz loop that publishes ``rt/lowcmd``
        frames and soft-stops on the way out - and a caller who already holds a
        built policy gets that whole loop through :meth:`run_policy` today.

        Rather than resolve a provider here and hand a quadruped an action
        vector whose joint semantics nothing has checked, this refuses with a
        message naming what to call instead. The gate runs first, so a caller
        whose robot is not even released hears that more specific reason.

        Args:
            instruction: Ignored while the provider registry is unwired.
            policy_port: Ignored.
            policy_host: Ignored.
            policy_provider: Ignored.
            duration: Ignored.
            **policy_kwargs: Ignored.

        Returns:
            An error envelope - either the gate's refusal or one naming
            :meth:`run_policy` as the wired path.
        """
        del instruction, policy_port, policy_host, policy_provider
        del duration, policy_kwargs
        refusal = self._check_motion_gates("start_task")
        if refusal is not None:
            return refusal
        return _refuse(
            "start_task: provider registry not wired for the Go2 yet; "
            "use run_policy(policy_object=...) to drive the control loop today"
        )

    def run_policy(
        self,
        policy_object: Policy | Callable[[Any], dict[str, Any]],
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Roll out an already-built policy on the 500 Hz control loop.

        The loop runs on its own thread so the caller returns immediately; poll
        :meth:`get_task_status` to observe progress. Every step re-gates through
        :meth:`_check_motion_gates`, and a gate that flips mid-rollout ends the
        loop with a zero-torque frame rather than leaving the robot frozen in the
        last commanded posture.

        ``policy_object`` is either a built
        :class:`~strands_robots.policies.Policy` or a bare callable - the
        admission check accepts a ``.step()`` attribute *or* a callable object,
        so the annotation admits the same set the refusal enforces. It is called
        each step with :attr:`state` and must return a joint-name-keyed action
        dict of the shape :meth:`send_action` accepts. A policy returning
        ``None`` or an unusable action is refused inside the loop, and the
        refusal count surfaces through :meth:`get_task_status`.

        Args:
            policy_object: The policy to roll out.
            instruction: Ignored; policies own their own conditioning.
            duration: Wall-clock budget in seconds. Must be positive and finite -
                ``nan`` poisons the deadline comparison so the loop would
                actuate with no budget, and ``inf`` never expires.
            n_steps: Optional step cap. Must be a positive count when given; a
                ``bool`` would silently cap at one and ``0`` would exit
                immediately inside a success envelope for a rollout that
                commanded nothing.

        Returns:
            A success envelope naming the running task's budgets, or an error
            envelope naming the gate or the argument that refused it.
        """
        del instruction  # policies own their own conditioning
        if err := positive_finite_number_error(duration, "duration", "run_policy"):
            return _refuse(err)
        if n_steps is not None and (err := positive_count_error(n_steps, "n_steps", "run_policy")):
            return _refuse(err)
        if policy_object is None:
            return _refuse("run_policy: policy_object is required")
        step_fn = getattr(policy_object, "step", None)
        if not callable(step_fn) and not callable(policy_object):
            return _refuse("run_policy: policy_object must be callable or expose a .step() method")
        refusal = self._check_motion_gates("run_policy")
        if refusal is not None:
            return refusal
        if self._pubs is None:
            return _refuse("publisher not initialised - call connect_eagerly() first")
        loop = _ControlLoop(driver=self, policy=policy_object, duration=float(duration), n_steps=n_steps)
        # Admission held across the ``is_running`` check, the reference
        # assignment and ``start()`` so a second caller cannot pass the check
        # before either assigns ``self._loop`` - two rollouts on one wire.
        with self._task_admission:
            if self._loop is not None and self._loop.is_running:
                return _refuse("run_policy: a task is already running; call stop_task first")
            # Clear any stashed terminal snapshot so a poller between here and
            # the first published frame sees this loop, not the previous one's
            # exit reason.
            self._last_task_snapshot = None
            self._loop = loop
            loop.start()
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "task_running": True,
                        "duration": float(duration),
                        "n_steps": n_steps,
                        "hz": _CONTROL_LOOP_HZ,
                    }
                }
            ],
        }

    def get_task_status(self) -> dict[str, Any]:
        """Report the running task's state. Safe to poll from any thread.

        A poller that missed the running window still sees the loop's final
        snapshot: it is stashed under the admission lock right before the loop
        clears itself, so every self-terminating exit reason round-trips to the
        caller instead of collapsing to "no task has been started".

        Returns:
            A success envelope carrying the loop's snapshot, the stashed
            terminal snapshot, or a note that no task has run on this driver.
        """
        with self._task_admission:
            loop = self._loop
            last = self._last_task_snapshot
        if loop is None:
            if last is not None:
                return {"status": "success", "content": [{"json": last}]}
            return {
                "status": "success",
                "content": [{"json": {"running": False, "reason": "no task has been started on this driver"}}],
            }
        return {"status": "success", "content": [{"json": loop.snapshot()}]}

    def stop_task(self) -> dict[str, Any]:
        """Stop the running task; the loop publishes a zero-torque frame on exit.

        Idempotent: no running task returns a success envelope naming the state.

        Returns:
            A success envelope carrying the loop's final snapshot when the thread
            joined. When it did not - a caller-supplied policy blocking on a
            remote inference call is the ordinary case - an *error* envelope with
            ``stopped=False``, so a caller reading only ``status`` cannot count
            the task as stopped while the payload's own ``running`` says the loop
            is still writing frames.
        """
        with self._task_admission:
            loop = self._loop
        if loop is None or not loop.is_running:
            return {"status": "success", "content": [{"text": "stop_task: no task is running"}]}
        joined = loop.stop("stop_task")
        snap = loop.snapshot()
        snap["stopped"] = joined
        if not joined:
            return {
                "status": "error",
                "content": [
                    {
                        "json": {
                            **snap,
                            "reason": (
                                "stop_task: control loop did not join within timeout; policy is "
                                "likely blocking - the loop will publish the zero-torque frame "
                                "when it exits"
                            ),
                        }
                    }
                ],
            }
        return {"status": "success", "content": [{"json": snap}]}

    # ------------------------------------------------------------------ #
    # DDS decoders. Each runs on the DDS thread; keep fast and pure.     #
    # ------------------------------------------------------------------ #

    def _on_lowstate(self, msg: Any) -> None:
        """Cache IMU, battery and per-joint telemetry from ``rt/lowstate``.

        The Go2 carries its battery state of charge inside ``LowState_`` under
        ``bms_state.soc``, where the G1 publishes a separate ``rt/lf/bmsstate``
        topic - so the battery floor's input arrives here rather than on its own
        subscription. Every field is read defensively: a firmware revision that
        drops one must cost that field, not the whole callback and with it the
        IMU the mesh publishes.

        Args:
            msg: The decoded ``unitree_go`` ``LowState_``.
        """
        imu = getattr(msg, "imu_state", None)
        if imu is not None:
            self._imu = {
                "quaternion": _to_float_list(getattr(imu, "quaternion", None)),
                "gyroscope": _to_float_list(getattr(imu, "gyroscope", None)),
                "accelerometer": _to_float_list(getattr(imu, "accelerometer", None)),
                "rpy": _to_float_list(getattr(imu, "rpy", None)),
            }
        bms = getattr(msg, "bms_state", None)
        if bms is not None:
            self._battery = {
                "pct": _to_float(getattr(bms, "soc", None)),
                "current": _to_float(getattr(bms, "current", None)),
                "cycle": _to_int(getattr(bms, "cycle", None)),
            }
        motors = getattr(msg, "motor_state", None)
        if motors is not None:
            joints: dict[str, Any] = {}
            for name, slot in GO2_JOINT_INDEX.items():
                try:
                    motor = motors[slot]
                except (IndexError, KeyError, TypeError):
                    continue
                joints[name] = {
                    "q": _to_float(getattr(motor, "q", None)),
                    "dq": _to_float(getattr(motor, "dq", None)),
                    "tau_est": _to_float(getattr(motor, "tau_est", None)),
                    "temperature": _to_int(getattr(motor, "temperature", None)),
                }
            self._joints = joints

    def _on_sportmode(self, msg: Any) -> None:
        """Cache body pose, velocity and gait from ``rt/sportmodestate``.

        Read-only telemetry the onboard controller publishes whether or not sport
        mode is released, which makes it the useful cross-check on a low-level
        rollout: body height and velocity say what the robot actually did with
        the frames this driver sent.

        Args:
            msg: The decoded ``unitree_go`` ``SportModeState_``.
        """
        self._sport = {
            "mode": _to_int(getattr(msg, "mode", None)),
            "gait_type": _to_int(getattr(msg, "gait_type", None)),
            "body_height": _to_float(getattr(msg, "body_height", None)),
            "position": _to_float_list(getattr(msg, "position", None)),
            "velocity": _to_float_list(getattr(msg, "velocity", None)),
            "yaw_speed": _to_float(getattr(msg, "yaw_speed", None)),
            "foot_force": _to_int_list(getattr(msg, "foot_force", None)),
        }


def _to_float(value: Any) -> float | None:
    """Coerce an SDK scalar to ``float``, or ``None`` when it is not numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    """Coerce an SDK scalar to ``int``, or ``None`` when it is not numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_list(value: Any) -> list[float] | None:
    """Coerce an SDK sequence to ``list[float]``, or ``None`` when unusable.

    Returns ``None`` rather than a partial list when any element fails to
    convert: half a quaternion is worse than no quaternion, because a consumer
    cannot tell it is half.
    """
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    out: list[float] = []
    for item in items:
        coerced = _to_float(item)
        if coerced is None:
            return None
        out.append(coerced)
    return out


def _to_int_list(value: Any) -> list[int] | None:
    """Coerce an SDK sequence to ``list[int]``, or ``None`` when unusable."""
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    out: list[int] = []
    for item in items:
        coerced = _to_int(item)
        if coerced is None:
            return None
        out.append(coerced)
    return out


class _ControlLoop:
    """The 500 Hz rollout thread for :meth:`Go2Driver.run_policy`.

    Simpler than the G1's loop in one respect that is worth stating, because the
    difference is deliberate rather than an omission: the G1 runs a second thread
    to refresh its FSM gate, because that gate's input is a synchronous DDS RPC
    which cannot be issued on a 2 ms control thread. Every input to the Go2's
    gate - the sport-mode release flag and the cached battery reading - is an
    in-memory read, so the per-step re-gate happens inline and there is no
    refresher thread, no staleness bound and no cache for one to go stale in.
    """

    def __init__(
        self,
        driver: Go2Driver,
        policy: Any,
        duration: float,
        n_steps: int | None,
    ) -> None:
        """Record the rollout's budgets. :meth:`start` spawns the thread.

        Args:
            driver: The driver whose gates, publisher and caches the loop uses.
            policy: A callable, or an object exposing ``.step()``.
            duration: Wall-clock budget in seconds.
            n_steps: Optional step cap.
        """
        self._driver = driver
        self._policy = policy
        self._duration = float(duration)
        self._n_steps = n_steps
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        # Snapshot fields, all read and written under ``_lock``.
        self._steps: int = 0
        self._refusals: int = 0
        self._exit_reason: str | None = None
        self._exit_detail: str | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None

    @property
    def is_running(self) -> bool:
        """Whether the loop thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Spawn the loop thread.

        Raises:
            RuntimeError: On a second call - a loop is single-use, and restarting
                one would silently share the snapshot of the previous rollout.
        """
        if self._thread is not None:
            raise RuntimeError("_ControlLoop.start() called twice")
        self._started_at = time.monotonic()
        # ``daemon=True`` so a caller who forgets ``stop_task`` cannot hang the
        # interpreter at exit. Every ordinary exit path still publishes the
        # zero-torque frame, because ``Go2Driver.stop`` and ``cleanup`` join the
        # loop before releasing the publisher.
        self._thread = threading.Thread(target=self._run, name=f"go2-control-{id(self):x}", daemon=True)
        self._thread.start()

    def stop(self, reason: str = "stop_task", timeout: float = 2.0) -> bool:
        """Signal the loop to exit and join its thread.

        The signal wins over policy work: the loop re-reads the event at the top
        of every step and again after the policy returns, before publishing.

        Args:
            reason: Recorded as the exit reason unless the loop already set one -
                a budget expiring concurrently with a caller's stop keeps its own,
                more specific reason.
            timeout: Seconds to wait for the join.

        Returns:
            ``True`` when the thread joined inside ``timeout``. ``False`` when it
            is still running, which a caller must report honestly rather than
            claiming a stop that has not happened.
        """
        self._stop_event.set()
        thread = self._thread
        joined = True
        if thread is not None:
            thread.join(timeout=timeout)
            joined = not thread.is_alive()
        with self._lock:
            if self._exit_reason is None:
                self._exit_reason = reason
        return joined

    def snapshot(self) -> dict[str, Any]:
        """Return the loop's public state. Safe from any thread."""
        with self._lock:
            elapsed: float | None
            if self._started_at is None:
                elapsed = None
            elif self._finished_at is None:
                elapsed = time.monotonic() - self._started_at
            else:
                elapsed = self._finished_at - self._started_at
            return {
                "running": self.is_running,
                "steps": self._steps,
                "refusals": self._refusals,
                "elapsed_s": elapsed,
                "duration_budget_s": self._duration,
                "n_steps_budget": self._n_steps,
                "exit_reason": self._exit_reason,
                "exit_detail": self._exit_detail,
                "hz": _CONTROL_LOOP_HZ,
            }

    def _set_exit(self, reason: str, detail: str | None = None) -> None:
        """Record the first terminal reason. Later calls do not overwrite it."""
        with self._lock:
            if self._exit_reason is None:
                self._exit_reason = reason
                self._exit_detail = detail

    def _call_policy(self) -> Any:
        """Invoke the policy for one step with the driver's cached state."""
        step_fn = getattr(self._policy, "step", None)
        if callable(step_fn):
            return step_fn(self._driver.state)
        return self._policy(self._driver.state)

    def _emit_zero_torque(self) -> None:
        """Publish the soft-stop frame. Best effort, and never raises.

        Called from the loop's ``finally``, so it runs on every exit path
        including a policy that raised. A publisher already released by a
        concurrent teardown means there is no wire to reach, which is reported
        and not an error here.
        """
        pubs = self._driver._pubs
        if pubs is None:
            logger.warning("go2 control loop: no publisher at shutdown; no zero-torque frame sent")
            return
        cmd, err = build_zero_torque_lowcmd()
        if err is not None:
            logger.error("go2 control loop: cannot build the zero-torque frame: %s", err)
            return
        try:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
        except ImportError as exc:  # pragma: no cover - exercised on hardware
            logger.error("go2 control loop: cannot publish the zero-torque frame: %s", exc)
            return
        pub_err = pubs.publish(_TOPIC_LOWCMD, LowCmd_, cmd)
        if pub_err is not None:
            logger.error("go2 control loop: zero-torque frame did not publish: %s", pub_err)

    def _run(self) -> None:
        """Roll the policy until a budget, a gate or the caller ends it.

        Terminal reasons are ``n_steps``, ``duration``, ``gate``, ``policy``,
        ``publish`` and the caller's own ``stop_task`` / ``stop`` / ``cleanup``.
        Whichever fires, the ``finally`` publishes the zero-torque frame and
        stashes the final snapshot on the driver so a late poller still sees it.
        """
        deadline = (self._started_at or time.monotonic()) + self._duration
        try:
            with Ticker(_CONTROL_LOOP_DT, self._stop_event) as ticker:
                while not self._stop_event.is_set():
                    if self._n_steps is not None and self._steps >= self._n_steps:
                        self._set_exit("n_steps")
                        break
                    if time.monotonic() >= deadline:
                        self._set_exit("duration")
                        break
                    refusal = self._driver._check_motion_gates("run_policy step")
                    if refusal is not None:
                        self._set_exit("gate", _refusal_text(refusal))
                        break
                    try:
                        action = self._call_policy()
                    except Exception as exc:  # noqa: BLE001 - a policy is caller code
                        self._set_exit("policy", f"{type(exc).__name__}: {exc}")
                        break
                    if self._stop_event.is_set():
                        break
                    cmd, err = build_lowcmd_from_action(action if isinstance(action, dict) else {})
                    if err is not None:
                        with self._lock:
                            self._refusals += 1
                        self._set_exit("policy", err)
                        break
                    pub_err = self._publish(cmd)
                    if pub_err is not None:
                        self._set_exit("publish", pub_err)
                        break
                    with self._lock:
                        self._steps += 1
                    if ticker.wait():
                        break
        finally:
            self._emit_zero_torque()
            with self._lock:
                self._finished_at = time.monotonic()
            # Stash the terminal snapshot and clear the driver's handle under the
            # admission lock, so a poller sees either this loop or its snapshot
            # and never "no task has been started" for a rollout that ran.
            snap = self.snapshot()
            snap["running"] = False
            with self._driver._task_admission:
                self._driver._last_task_snapshot = snap
                if self._driver._loop is self:
                    self._driver._loop = None

    def _publish(self, cmd: Any) -> str | None:
        """Publish one built frame on ``rt/lowcmd``.

        Args:
            cmd: The sealed ``LowCmd_``.

        Returns:
            ``None`` on success, or a reason naming why the frame did not reach
            the wire.
        """
        pubs = self._driver._pubs
        if pubs is None:
            return "publisher was released while the loop was running"
        try:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
        except ImportError as exc:  # pragma: no cover - exercised on hardware
            return f"unitree_sdk2py is not installed: {exc}"
        return pubs.publish(_TOPIC_LOWCMD, LowCmd_, cmd)


def _refusal_text(refusal: dict[str, Any]) -> str:
    """Extract the reason string from a refusal envelope for logging."""
    for block in refusal.get("content", []):
        text = block.get("text")
        if isinstance(text, str):
            return text
    return "refused"
