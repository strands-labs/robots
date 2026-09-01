"""Native driver for the Booster Robotics T1 humanoid.

``Robot("booster_t1", mode="real", driver="strands", port=<ip>)`` builds one of
these. The instance satisfies
:class:`~strands_robots.drivers.base.HardwareDriver` structurally, so
:func:`~strands_robots.robot.Robot` returns it and the mesh, teleop rail and
agent tool surface consume it exactly like any other driver.

Why a native driver: the T1 has no lerobot robot type, so ``driver="lerobot"``
cannot build it at all. Its own SDK (``booster_robotics_sdk_python``, a pybind11
wrapper over a DDS transport) is the honest client. That SDK is a vendor wheel,
not a declared dependency of this project - the same footing as the G1's
``unitree-sdk2`` - so it is imported lazily and its absence is a named refusal
from :meth:`BoosterDriver.connect_eagerly`, never an ``ImportError`` at import.

THE DECISIVE FACT - the T1 splits control in two, and the split is not
negotiable. The legs, waist and head are owned by an onboard whole-body
controller that keeps the robot standing; the eight upper-body joints can be
handed to a host, but only after ``B1LocoClient.UpperBodyCustomControl(True)``
and only while every other slot in the frame carries zero gain. A ``LowCmd``
that puts stiffness on a leg fights the balance controller with a robot's mass
behind it, and the publish reports success either way. So this driver:

* refuses :meth:`BoosterDriver.send_action` until upper-body control is enabled
  (:meth:`BoosterDriver.enable_upper_body`), naming the call that opens it;
* accepts targets for the eight upper-body slots only, and names
  :meth:`BoosterDriver.rotate_head` / :meth:`BoosterDriver.move` for the joints
  it will not write;
* emits ``q=0, kp=0, kd=0`` for every non-upper-body slot, so the onboard
  controller keeps the legs (:func:`build_frame`);
* holds an uncommanded upper-body joint at its last *observed* position rather
  than at zero, so commanding one arm does not drop the other;
* refuses a write while the robot reports a fall state other than ``IS_READY``
  (:data:`FALL_STATE_NAMES`).

Every wire literal here is transcribed from the vendor's own reference client
shipped inside the SDK wheel (``booster_robotics_sdk_python/arm_controller.py``
and ``move_controller.py``): ``mode = 0x0A`` is its position mode, ``kp=60 /
kd=3`` are its upper-body gains, ``UpperBodyCustomControl`` is its precondition,
and zero-gain legs are its rule. Values a robot reports are read, not assumed -
:meth:`BoosterDriver.send_action` refuses until a ``LowState`` frame has
arrived, because the frame width and the hold positions both come from it.

Nothing here imports the SDK at module load. Every SDK touch is inside a
function body, so the module imports on CI, on Thor, and in every unit test.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK
from strands_robots.utils import boolean_flag_error, dds_domain_id_error, finite_number_error

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: Canonical robot names this driver serves. Read by
#: :data:`strands_robots.drivers._SHIPPED_DRIVERS`, so the driver declares its
#: own family and the seam does not restate it.
SUPPORTED_ROBOTS: tuple[str, ...] = ("booster_t1",)

# The T1's motor slots, in the order the SDK's own ``JointIndex`` enum numbers
# them. Kept as a module constant rather than read from the SDK for the reason
# the G1 driver keeps its own map: a typo in a caller's action dict must surface
# here, on a machine that need not have the SDK installed at all, instead of as
# an import failure one layer down.
#
# The vendor spells two of these slots twice. ``JointIndex`` calls slots 4 and 5
# ``kLeftElbowPitch`` / ``kLeftElbowYaw``; the ``ArmJoint`` helper class in the
# same wheel calls the same two ``LeftYaw`` / ``LeftElbow``. The indices agree,
# which is what reaches the wire, so this table follows ``JointIndex`` - the
# enum the compiled core exposes - and records the disagreement rather than
# silently picking a side.
BOOSTER_JOINT_INDEX: dict[str, int] = {
    # Head - owned by the onboard controller; see :meth:`BoosterDriver.rotate_head`.
    "head_yaw": 0,
    "head_pitch": 1,
    # Left arm.
    "left_shoulder_pitch": 2,
    "left_shoulder_roll": 3,
    "left_elbow_pitch": 4,
    "left_elbow_yaw": 5,
    # Right arm.
    "right_shoulder_pitch": 6,
    "right_shoulder_roll": 7,
    "right_elbow_pitch": 8,
    "right_elbow_yaw": 9,
    # Waist and legs - owned by the onboard controller.
    "waist": 10,
    "left_hip_pitch": 11,
    "left_hip_roll": 12,
    "left_hip_yaw": 13,
    "left_knee_pitch": 14,
    "left_crank_up": 15,
    "left_crank_down": 16,
    "right_hip_pitch": 17,
    "right_hip_roll": 18,
    "right_hip_yaw": 19,
    "right_knee_pitch": 20,
    "right_crank_up": 21,
    "right_crank_down": 22,
}

#: The slots ``UpperBodyCustomControl`` hands to a host: the eight arm joints,
#: slots 2..9. The vendor's own reference holds exactly this range under host
#: gains and zero-gains everything else, head included - so the head is *not*
#: here even though it is upper body, and :meth:`BoosterDriver.rotate_head` is
#: the path to it.
UPPER_BODY_SLOTS: frozenset[int] = frozenset(range(2, 10))

#: Position mode. ``0x0A`` is the literal the vendor's reference client writes
#: into every ``MotorCmd.mode`` for position control.
POSITION_MODE: int = 0x0A

#: Default upper-body gains, from the same reference client. A caller may
#: override per joint; a supplied value always wins.
UPPER_BODY_KP: float = 60.0
UPPER_BODY_KD: float = 3.0

#: The two ``LowCmd.cmd_type`` conventions, and the ``LowState`` field each one
#: reads its positions from. They are two spellings of the same motors - the
#: difference is whether the ankle parallel linkage is addressed at the crank or
#: at the resolved joint - so a frame declaring one convention must hold its
#: uncommanded joints from *that* convention's state array or the held targets
#: are in the other frame of reference.
CMD_TYPE_STATE_FIELD: dict[str, str] = {
    "parallel": "motor_state_parallel",
    "serial": "motor_state_serial",
}

#: The fall states the T1 publishes, by the SDK's ``FallDownStateType`` member
#: name. Only ``IS_READY`` is a state a host may write arm targets in: the other
#: three are a robot on its way to, on, or getting off the floor, where a held
#: arm posture is at best noise and at worst an obstruction to the getting-up
#: routine.
FALL_STATE_NAMES: dict[int, str] = {0: "IS_READY", 1: "IS_FALLING", 2: "HAS_FALLEN", 3: "IS_GETTING_UP"}

#: The one fall state that admits a write.
FALL_STATE_READY: str = "IS_READY"

#: The modes ``B1LocoClient.ChangeMode`` accepts, by the SDK's ``RobotMode``
#: member name. Named here so a caller sees the set without the SDK installed -
#: which makes it a claim about the vendor's vocabulary that an SDK build can
#: contradict, so :func:`resolve_vendor_member` reads the member rather than
#: assuming it. ``kUnknown`` is a value ``GetMode`` reports and not a mode a host
#: may ask for, so it is deliberately absent: :meth:`BoosterDriver.read_mode`
#: passes the vendor's name through verbatim, this set bounds the write.
ROBOT_MODES: tuple[str, ...] = ("kDamping", "kPrepare", "kWalking", "kCustom", "kSoccer")


def _refuse(reason: str) -> dict[str, Any]:
    """Wrap ``reason`` in the error envelope every driver verb returns."""
    return {"status": "error", "content": [{"text": reason}]}


#: Attributes a pybind11 enum carries that are not members of it. Excluded from
#: the vocabulary :func:`resolve_vendor_member` reports so an operator reads
#: modes rather than binding noise.
_ENUM_NON_MEMBERS: frozenset[str] = frozenset({"name", "value"})

_ABSENT = object()


def declared_members(enum: object) -> tuple[str, ...]:
    """Return the member names the installed SDK's ``enum`` declares.

    Args:
        enum: A pybind11 enum type from ``booster_robotics_sdk_python`` -
            ``RobotMode``, ``LowCmdType``, ``FallDownStateType``.

    Returns:
        The member names, sorted. Underscore-prefixed bindings and the
        per-member ``name``/``value`` descriptors are not members and are left
        out (:data:`_ENUM_NON_MEMBERS`).
    """
    return tuple(sorted(n for n in dir(enum) if not n.startswith("_") and n not in _ENUM_NON_MEMBERS))


def resolve_vendor_member(enum: object, name: str, *, enum_name: str, verb: str) -> Any:
    """Resolve a vendor enum member by name, or say why this build cannot.

    Every name this driver hands an SDK enum is a *frozen claim about the
    vendor's vocabulary*: :data:`ROBOT_MODES` and the keys of
    :data:`CMD_TYPE_STATE_FIELD` are spelled here so a caller sees the set
    without the SDK installed, and the SDK is a vendor wheel pinned to the
    robot's firmware rather than a dependency this project resolves. A build
    that declares a different set is therefore a normal operational state, not
    an impossible one - and a bare ``getattr`` turns it into an
    :exc:`AttributeError` leaving a driver verb, past the envelope every verb is
    contracted to return.

    Args:
        enum: The SDK enum type to read the member off.
        name: The member name to resolve.
        enum_name: The enum's name as the vendor spells it, for the refusal.
        verb: The driver verb resolving it, for the refusal.

    Returns:
        The enum member, or a :class:`str` naming what this SDK build declares.
        A member is never a ``str`` - it is a pybind11 enum value - so
        ``isinstance(result, str)`` is the caller's check, the same shape
        :func:`resolve_targets` uses.
    """
    member = getattr(enum, name, _ABSENT)
    if member is not _ABSENT:
        return member
    declared = declared_members(enum)
    return (
        f"{verb}: the installed booster_robotics_sdk_python declares no "
        f"{enum_name}.{name}, so this build cannot be asked for it. It declares "
        f"{', '.join(declared) if declared else '(no members)'}. The names this driver asks for are "
        "the vendor's own, so a build that lacks one is an SDK version mismatch rather than a bad "
        "argument - install the SDK wheel that matches the robot's firmware"
    )


def resolve_targets(action: dict[str, Any]) -> dict[int, dict[str, float]] | str:
    """Resolve an action dict to per-slot upper-body targets, or say why not.

    Args:
        action: ``{joint_name: target_radians}``, or
            ``{joint_name: {"q": ..., "kp": ..., "kd": ...}}`` when a caller
            wants per-joint gains. A missing ``q`` inside the mapping form
            refuses the whole action - a silently zeroed target is a joint
            driving to zero under 60 N.m/rad of stiffness.

    Returns:
        Slot index -> ``{"q", "kp", "kd"}``, or a reason string. A reason names
        the joint at fault and, for a joint this driver will not write, the
        method that does reach it.
    """
    if not isinstance(action, dict) or not action:
        return "send_action: action must be a non-empty dict of joint name -> target radians"

    targets: dict[int, dict[str, float]] = {}
    for name, value in action.items():
        slot = BOOSTER_JOINT_INDEX.get(name)
        if slot is None:
            return f"send_action: unknown joint {name!r}. The T1's joints are: {', '.join(sorted(BOOSTER_JOINT_INDEX))}"
        if slot not in UPPER_BODY_SLOTS:
            reach = "rotate_head()" if name.startswith("head_") else "move()"
            return (
                f"send_action: {name!r} (slot {slot}) is owned by the T1's onboard whole-body "
                f"controller, which this driver must not fight - it commands only the upper-body "
                f"joints {', '.join(sorted(n for n, s in BOOSTER_JOINT_INDEX.items() if s in UPPER_BODY_SLOTS))}. "
                f"Reach this one with {reach}."
            )

        entry = {"kp": UPPER_BODY_KP, "kd": UPPER_BODY_KD}
        if isinstance(value, dict):
            if "q" not in value:
                return f"send_action: {name!r} has no 'q' - refusing rather than commanding it to zero"
            fields = value
        else:
            fields = {"q": value}
        for key in ("q", "kp", "kd"):
            if key not in fields:
                continue
            reason = finite_number_error(fields[key], f"{name}.{key}", "send_action")
            if reason is not None:
                return reason
            entry[key] = float(fields[key])
        targets[slot] = entry
    return targets


def build_frame(
    targets: dict[int, dict[str, float]],
    held_q: list[float],
) -> list[dict[str, float]]:
    """Build one full-width ``LowCmd`` frame as plain per-slot dicts.

    Plain dicts rather than ``MotorCmd`` objects so the rule that matters -
    which slots carry stiffness - is gradeable without the SDK installed;
    :meth:`BoosterDriver.send_action` writes them onto the SDK message.

    The three cases, and why each is what it is:

    * A commanded upper-body slot takes the caller's ``q`` and gains.
    * An *uncommanded* upper-body slot holds ``held_q[slot]`` under the same
      gains. Zero would be a target, not a hold: commanding the left arm would
      swing the right one to its zero pose.
    * Every other slot - legs, waist, head - gets ``q=0, kp=0, kd=0``. Zero
      gain is what leaves the onboard whole-body controller in charge of
      balance. This is the safety-critical row of the table.

    Args:
        targets: Slot -> ``{"q", "kp", "kd"}`` from :func:`resolve_targets`.
        held_q: Last observed position per slot, indexed by slot. Its length is
            the frame width, which is the motor count the robot itself
            reported.

    Returns:
        One dict per slot, in slot order, each with ``q``, ``dq``, ``tau``,
        ``kp``, ``kd`` and ``mode``.
    """
    frame: list[dict[str, float]] = []
    for slot in range(len(held_q)):
        if slot in targets:
            entry = targets[slot]
            q, kp, kd = entry["q"], entry["kp"], entry["kd"]
        elif slot in UPPER_BODY_SLOTS:
            q, kp, kd = held_q[slot], UPPER_BODY_KP, UPPER_BODY_KD
        else:
            q, kp, kd = 0.0, 0.0, 0.0
        frame.append({"q": q, "dq": 0.0, "tau": 0.0, "kp": kp, "kd": kd, "mode": float(POSITION_MODE)})
    return frame


def parse_low_state(msg: Any, state_field: str) -> dict[str, Any]:
    """Read one ``LowState`` into a plain snapshot.

    Args:
        msg: The SDK's ``LowState``.
        state_field: Which motor array to read - see
            :data:`CMD_TYPE_STATE_FIELD`.

    Returns:
        ``{"joints", "velocities", "torques", "temperatures", "imu"}``. Absent
        fields are omitted rather than defaulted: a snapshot that reports a
        zeroed IMU the robot never sent is worse than one that reports none.
    """
    snapshot: dict[str, Any] = {}
    motors = getattr(msg, state_field, None) or []
    snapshot["joints"] = [float(getattr(m, "q", 0.0)) for m in motors]
    snapshot["velocities"] = [float(getattr(m, "dq", 0.0)) for m in motors]
    snapshot["torques"] = [float(getattr(m, "tau_est", 0.0)) for m in motors]
    snapshot["temperatures"] = [float(getattr(m, "temperature", 0.0)) for m in motors]
    imu = getattr(msg, "imu_state", None)
    if imu is not None:
        snapshot["imu"] = {
            "rpy": [float(v) for v in getattr(imu, "rpy", []) or []],
            "gyro": [float(v) for v in getattr(imu, "gyro", []) or []],
            "acc": [float(v) for v in getattr(imu, "acc", []) or []],
        }
    return snapshot


class BoosterDriver:
    """Native driver for the Booster Robotics T1.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver` structurally
    - a Protocol - so no import from :mod:`strands_robots.drivers.base` is
    needed. The surface check
    :func:`~strands_robots.drivers.register_native_driver` runs at registration
    time is what pins the contract.
    """

    def __init__(
        self,
        tool_name: str = "booster_t1",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        domain_id: int = 0,
        robot_name: str | None = None,
        cmd_type: str = "parallel",
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` does the SDK work.

        Args:
            tool_name: Name the agent invokes the driver by, and the mesh peer
                id when the driver is wrapped by
                :class:`~strands_robots.mesh.Mesh`.
            cameras: Accepted for parity with the lerobot driver; unused,
                because the T1's cameras are addressed through its own
                ``CameraClient``, not v4l2.
            data_config: Accepted for parity; unused.
            port: The robot's IP address, handed to ``ChannelFactory.Init`` as
                the vendor's reference client hands it - an empty string there
                means "discover on the default interface".
            domain_id: DDS domain, ``0`` to
                :data:`~strands_robots.utils.MAX_DDS_DOMAIN_ID`. ``0`` is what
                the vendor's reference client passes and what the robot ships
                with. The ceiling is the RTPS port map's rather than a
                convention - see
                :func:`~strands_robots.utils.dds_domain_id_error`.
            robot_name: Multi-robot suffix. When set, the channels are opened
                with ``InitWithName`` so two T1s on one network do not share a
                topic; ``None`` selects the single-robot channels.
            cmd_type: ``"parallel"`` or ``"serial"`` - which
                :data:`CMD_TYPE_STATE_FIELD` convention frames declare and
                positions are held from. Defaults to ``"parallel"``, the
                convention the vendor's reference client leaves in place.
            **kwargs: Ignored; accepted so the factory can forward extras.

        Raises:
            ValueError: If ``cmd_type`` is not one of
                :data:`CMD_TYPE_STATE_FIELD`, or ``domain_id`` does not name a
                DDS domain (:func:`~strands_robots.utils.dds_domain_id_error`).
        """
        del cameras, data_config  # accepted for parity; unused here
        if kwargs:
            logger.debug("BoosterDriver ignoring extra kwargs: %s", sorted(kwargs))
        if cmd_type not in CMD_TYPE_STATE_FIELD:
            raise ValueError(
                f"BoosterDriver: cmd_type must be one of {', '.join(sorted(CMD_TYPE_STATE_FIELD))}, got {cmd_type!r}"
            )
        # Refuse a domain id through the shared domain rather than a local
        # test. A domain id indexes the RTPS port map, so it has a ceiling as
        # well as a floor, and the same id must not be accepted here and
        # refused by the telemetry bridges that advertise this robot's topics.
        if error := dds_domain_id_error(domain_id, "domain_id", type(self).__name__):
            raise ValueError(error)

        self._tool_name = tool_name
        self._port = port or ""
        self._domain_id = domain_id
        self._robot_name = robot_name
        self._cmd_type = cmd_type
        self._state_field = CMD_TYPE_STATE_FIELD[cmd_type]

        self._client: Any | None = None
        self._publisher: Any | None = None
        self._subscriber: Any | None = None
        self._battery_subscriber: Any | None = None
        self._fall_subscriber: Any | None = None
        self._connected = False
        self._connect_error: str | None = None
        self._upper_body_enabled = False

        self._cache_lock = threading.Lock()
        self._last_state: dict[str, Any] | None = None
        self._battery: dict[str, float] | None = None
        self._fall_state: str | None = None

    # ------------------------------------------------------------------ #
    # Agent tool surface.                                               #
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
        """Whether the SDK channels are open."""
        return self._connected

    @property
    def tool_spec(self) -> ToolSpec:
        """The universal read-only trio plus a controlled stop.

        Motion verbs are deliberately absent from the *agent* surface: on a
        1.2 m biped the write path is opened by a caller who has read
        :meth:`send_action`'s contract, not by a model choosing an enum value.
        """
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Booster Robotics T1 native driver: reads the T1's SDK bus for joint "
                    "positions, velocities, torques, temperatures and IMU; upper-body writes "
                    "go through send_action, locomotion through move()."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "sensors: return the latest joint and IMU snapshot; "
                                    "status: report connection, mode gate and frame width; "
                                    "stop: halt locomotion and hand the upper body back to the "
                                    "onboard controller"
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
        """Open the SDK channels: loco client, low-state subscriber, publisher.

        Returns ``None`` on success. Off hardware - no SDK, or no robot
        answering - returns a reason and leaves the driver usable: every read
        returns its empty cache and every write refuses "not connected".
        Idempotent.

        A partially built channel set is released rather than kept: a driver
        holding a live subscriber and no publisher reads fine and refuses every
        write, which is the state hardest to diagnose from the outside.

        Every endpoint is constructed under
        :data:`~strands_robots.tools.g1._g1_common._DDS_INIT_LOCK`. The channel
        factory, the loco client's ``Init()`` and each of the four channels build
        DDS readers and writers, and the CycloneDDS bindings segfault when one
        endpoint is constructed concurrently with another - which happens as soon
        as anything else in the process touches DDS, because
        :class:`~strands_robots.tools.g1._dds_engine.DDSSubscriberSet` creates
        every subscriber under that same lock. A segfault is not catchable by the
        "record the reason and stay usable for reads" boundary below: the process
        dies, possibly while the robot is standing under its own controller.

        The lazy SDK import stays outside the lock - it creates no endpoint, and
        holding the shared lock across it would stall every subscriber
        construction in the process for its duration. So does the teardown of a
        partial set, matching
        :meth:`~strands_robots.tools.g1._dds_engine.DDSSubscriberSet.close`:
        the lock serialises construction, not release.
        """
        if self._connected:
            return None
        try:
            import booster_robotics_sdk_python as sdk
        except ImportError as exc:
            self._connect_error = (
                f"booster_robotics_sdk_python is not installed: {exc}. "
                "Install it with: pip install booster_robotics_sdk_python"
            )
            return self._connect_error

        subscriber: Any | None = None
        publisher: Any | None = None
        battery: Any | None = None
        fall: Any | None = None
        try:
            with _DDS_INIT_LOCK:
                sdk.ChannelFactory.Instance().Init(self._domain_id, self._port)

                client = sdk.B1LocoClient()
                if self._robot_name is None:
                    client.Init()
                else:
                    client.InitWithName(self._robot_name)

                subscriber = sdk.B1LowStateSubscriber(self._on_low_state)
                publisher = sdk.B1LowCmdPublisher()
                battery = sdk.B1BatteryStateSubscriber(self._on_battery)
                fall = sdk.B1FallDownStateSubscriber(self._on_fall_state)
                for channel in (subscriber, publisher, battery, fall):
                    if self._robot_name is None:
                        channel.InitChannel()
                    else:
                        channel.InitChannelWithName(self._robot_name)
        except (RuntimeError, OSError, ValueError) as exc:
            for channel in (subscriber, publisher, battery, fall):
                if channel is not None:
                    try:
                        channel.CloseChannel()
                    except (RuntimeError, OSError):
                        logger.debug("%s: releasing a partial channel set", self._tool_name, exc_info=True)
            self._connect_error = (
                f"Booster SDK channels did not open (domain {self._domain_id}, ip {self._port!r}): {exc}"
            )
            return self._connect_error

        self._client = client
        self._subscriber = subscriber
        self._publisher = publisher
        self._battery_subscriber = battery
        self._fall_subscriber = fall
        self._connected = True
        self._connect_error = None
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report reachability, the write gate, and what the robot has reported."""
        with self._cache_lock:
            state = self._last_state
            battery = dict(self._battery or {})
            fall_state = self._fall_state
        joints = (state or {}).get("joints") or []
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self._connected,
                        "connect_error": self._connect_error,
                        "battery_pct": battery.get("pct"),
                        "fall_state": fall_state,
                        "ip": self._port,
                        "domain_id": self._domain_id,
                        "robot_name": self._robot_name,
                        "cmd_type": self._cmd_type,
                        "upper_body_enabled": self._upper_body_enabled,
                        "frame_width": len(joints),
                        "mode": self.read_mode(),
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Halt locomotion and hand the upper body back. Never raises."""
        self.stop_task()

    def cleanup(self) -> None:
        """Release the SDK channels. Idempotent, and stops first."""
        if self._connected:
            self.stop_task()
        for channel in (self._subscriber, self._publisher, self._battery_subscriber, self._fall_subscriber):
            if channel is None:
                continue
            try:
                channel.CloseChannel()
            except (RuntimeError, OSError):
                logger.debug("%s: channel close failed during cleanup", self._tool_name, exc_info=True)
        self._subscriber = None
        self._publisher = None
        self._battery_subscriber = None
        self._fall_subscriber = None
        self._client = None
        self._connected = False
        self._upper_body_enabled = False

    # ------------------------------------------------------------------ #
    # Write path.                                                        #
    # ------------------------------------------------------------------ #

    def enable_upper_body(self, on: bool = True) -> dict[str, Any]:
        """Open (or close) the host's claim on the eight upper-body joints.

        ``B1LocoClient.UpperBodyCustomControl`` is the T1's precondition for a
        host writing arm targets: without it the onboard controller keeps the
        arms and a published frame changes nothing while reporting success.
        :meth:`send_action` therefore refuses until this has succeeded.

        Args:
            on: ``True`` to claim the upper body, ``False`` to hand it back.

        Returns:
            A success envelope naming the new state, or a refusal.
        """
        reason = boolean_flag_error(on, "on", "enable_upper_body")
        if reason is not None:
            return _refuse(reason)
        if self._client is None:
            return _refuse("enable_upper_body: not connected - call connect_eagerly() first")
        try:
            self._client.UpperBodyCustomControl(on)
        except (RuntimeError, OSError, TypeError) as exc:
            return _refuse(f"enable_upper_body: the T1 refused UpperBodyCustomControl({on}): {exc}")
        self._upper_body_enabled = on
        return {
            "status": "success",
            "content": [{"json": {"upper_body_enabled": on}}],
        }

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Publish one ``LowCmd`` holding the upper body at ``action``.

        The action dict is keyed by joint name - the eight upper-body names in
        :data:`BOOSTER_JOINT_INDEX` whose slot is in :data:`UPPER_BODY_SLOTS` -
        and each value is either a target in radians or a
        ``{"q", "kp", "kd"}`` mapping.

        What this refuses, and why each refusal is a refusal rather than a
        best effort:

        * Upper-body control not enabled - the frame would be ignored and
          reported as sent.
        * No ``LowState`` seen yet - the frame width and the hold positions for
          the uncommanded arm joints both come from the robot's own report, and
          inventing either commands a joint to a position nothing measured.
        * A joint outside the upper body - the write would fight the onboard
          balance controller.

        This is one frame, not a control loop. The T1 holds a published
        posture, so a caller who wants a trajectory calls this on their own
        timer - the vendor's reference client runs 100 Hz.

        Args:
            action: Joint targets, as above.
            robot_name: Ignored; the driver fronts one robot.

        Returns:
            A success envelope naming the commanded joints and the frame width,
            or a refusal.
        """
        del robot_name  # driver fronts one T1
        if not self._connected or self._publisher is None:
            return _refuse("send_action: not connected - call connect_eagerly() first")
        if not self._upper_body_enabled:
            return _refuse(
                "send_action: the T1's onboard controller still owns the arms - "
                "call enable_upper_body() first, which is the SDK's own precondition "
                "(UpperBodyCustomControl); a frame sent without it is ignored and reported as sent"
            )
        with self._cache_lock:
            held_q = list((self._last_state or {}).get("joints") or [])
            fall_state = self._fall_state
        if fall_state is not None and fall_state != FALL_STATE_READY:
            return _refuse(
                f"send_action: the T1 reports {fall_state} - a held arm posture is noise while the "
                f"robot is on its way to, on, or getting off the floor, and can obstruct the "
                f"getting-up routine. Writes resume when it reports {FALL_STATE_READY}"
            )
        if not held_q:
            return _refuse(
                f"send_action: no LowState frame has arrived yet, so neither the frame width nor the "
                f"hold position of an uncommanded arm joint is known. The subscriber is open on "
                f"{self._state_field!r}; retry once the robot is publishing"
            )

        targets = resolve_targets(action)
        if isinstance(targets, str):
            return _refuse(targets)
        outside = sorted(slot for slot in targets if slot >= len(held_q))
        if outside:
            return _refuse(
                f"send_action: slots {outside} are past the {len(held_q)}-motor frame this T1 reports; "
                "the robot's own motor count bounds the frame"
            )

        frame = build_frame(targets, held_q)
        try:
            import booster_robotics_sdk_python as sdk
        except ImportError as exc:  # pragma: no cover - connect_eagerly already needed it
            return _refuse(f"booster_robotics_sdk_python is not installed: {exc}")
        cmd_type = resolve_vendor_member(
            sdk.LowCmdType, self._cmd_type.upper(), enum_name="LowCmdType", verb="send_action"
        )
        if isinstance(cmd_type, str):
            return _refuse(cmd_type)
        cmd = sdk.LowCmd()
        cmd.cmd_type = cmd_type
        cmd.resize_motor_cmd(len(frame))
        for slot, values in enumerate(frame):
            motor = cmd.motor_cmd_at(slot)
            motor.q = values["q"]
            motor.dq = values["dq"]
            motor.tau = values["tau"]
            motor.kp = values["kp"]
            motor.kd = values["kd"]
            motor.mode = int(values["mode"])
        try:
            accepted = self._publisher.Write(cmd)
        except (RuntimeError, OSError) as exc:
            return _refuse(f"send_action: publishing the LowCmd failed: {exc}")
        if not accepted:
            return _refuse("send_action: the SDK publisher rejected the LowCmd")
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "joints": sorted(action),
                        "frame_width": len(frame),
                        "cmd_type": self._cmd_type,
                    }
                }
            ],
        }

    def move(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> dict[str, Any]:
        """Command a base twist through the onboard locomotion controller.

        The T1 walks under its own controller; a host asks for a velocity, not
        for leg joint targets. Every component is signed, so
        :func:`~strands_robots.utils.finite_number_error` is the whole domain -
        a magnitude limit belongs to the controller that owns the legs.

        Args:
            vx: Forward velocity, m/s.
            vy: Lateral velocity, m/s.
            vyaw: Yaw rate, rad/s.

        Returns:
            A success envelope echoing the twist, or a refusal.
        """
        for value, name in ((vx, "vx"), (vy, "vy"), (vyaw, "vyaw")):
            reason = finite_number_error(value, name, "move")
            if reason is not None:
                return _refuse(reason)
        if self._client is None:
            return _refuse("move: not connected - call connect_eagerly() first")
        try:
            self._client.MoveCommand(float(vx), float(vy), float(vyaw))
        except (RuntimeError, OSError) as exc:
            return _refuse(f"move: the T1 refused the twist: {exc}")
        return {"status": "success", "content": [{"json": {"vx": vx, "vy": vy, "vyaw": vyaw}}]}

    def rotate_head(self, pitch: float = 0.0, yaw: float = 0.0) -> dict[str, Any]:
        """Point the head through the onboard controller.

        The head's two slots are excluded from :data:`UPPER_BODY_SLOTS` because
        the vendor's own reference client zero-gains them while driving the
        arms; ``B1LocoClient.RotateHead`` is the path that reaches them.

        Args:
            pitch: Head pitch, radians.
            yaw: Head yaw, radians.

        Returns:
            A success envelope echoing the angles, or a refusal.
        """
        for value, name in ((pitch, "pitch"), (yaw, "yaw")):
            reason = finite_number_error(value, name, "rotate_head")
            if reason is not None:
                return _refuse(reason)
        if self._client is None:
            return _refuse("rotate_head: not connected - call connect_eagerly() first")
        try:
            self._client.RotateHead(float(pitch), float(yaw))
        except (RuntimeError, OSError) as exc:
            return _refuse(f"rotate_head: the T1 refused the head angles: {exc}")
        return {"status": "success", "content": [{"json": {"pitch": pitch, "yaw": yaw}}]}

    def change_mode(self, mode: str) -> dict[str, Any]:
        """Ask the T1 for a whole-body mode.

        Args:
            mode: A member name of the SDK's ``RobotMode`` - see
                :data:`ROBOT_MODES`. A name outside that set is refused here; a
                name inside it that the *installed* SDK build does not declare
                is refused by :func:`resolve_vendor_member`, which names the
                vocabulary the build has.

        Returns:
            A success envelope naming the requested mode, or a refusal.
        """
        if mode not in ROBOT_MODES:
            return _refuse(f"change_mode: mode must be one of {', '.join(ROBOT_MODES)}, got {mode!r}")
        if self._client is None:
            return _refuse("change_mode: not connected - call connect_eagerly() first")
        try:
            import booster_robotics_sdk_python as sdk
        except ImportError as exc:  # pragma: no cover - connect_eagerly already needed it
            return _refuse(f"booster_robotics_sdk_python is not installed: {exc}")
        member = resolve_vendor_member(sdk.RobotMode, mode, enum_name="RobotMode", verb="change_mode")
        if isinstance(member, str):
            return _refuse(member)
        try:
            self._client.ChangeMode(member)
        except (RuntimeError, OSError) as exc:
            return _refuse(f"change_mode: the T1 refused {mode}: {exc}")
        return {"status": "success", "content": [{"json": {"mode": mode}}]}

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
        """Refuse: no provider registry is plumbed to this driver yet."""
        del instruction, policy_port, policy_host, policy_provider, duration, policy_kwargs
        return _refuse(
            "start_task: no policy provider is wired to the T1 yet. A caller with a built policy "
            "drives the upper body by calling send_action on their own timer"
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
            "run_policy: this driver publishes one frame per call and owns no control loop. "
            "Call send_action on your own timer (the vendor's reference client runs 100 Hz), "
            'or use mode="sim" for a host-driven rollout'
        )

    def get_task_status(self) -> dict[str, Any]:
        """Report the write gate, the only task state this driver holds."""
        return {
            "status": "success",
            "content": [{"json": {"running": False, "upper_body_enabled": self._upper_body_enabled}}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Halt locomotion, then hand the upper body back to the robot.

        Both halves are attempted even if the first refuses: a driver that
        stopped walking and kept the arms is not stopped. The envelope reports
        each half rather than asserting the pair succeeded.
        """
        if self._client is None:
            return _refuse("stop_task: not connected")
        halted = self.move(0.0, 0.0, 0.0)
        released = self.enable_upper_body(False) if self._upper_body_enabled else None
        outcome = {
            "locomotion_halted": halted["status"] == "success",
            "upper_body_released": released is None or released["status"] == "success",
        }
        if not all(outcome.values()):
            return {"status": "error", "content": [{"json": outcome}]}
        return {"status": "success", "content": [{"json": outcome}]}

    # ------------------------------------------------------------------ #
    # Read path.                                                         #
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict[str, float]:
        """Joint positions by name, for the mesh's state topic.

        Returns:
            ``{joint_name: radians}`` for as many slots as the robot reported,
            or ``{}`` before the first frame.
        """
        with self._cache_lock:
            joints = list((self._last_state or {}).get("joints") or [])
        by_slot = {slot: name for name, slot in BOOSTER_JOINT_INDEX.items()}
        return {by_slot[slot]: value for slot, value in enumerate(joints) if slot in by_slot}

    def read_state(self) -> dict[str, Any]:
        """The latest ``LowState`` snapshot, or an empty dict before the first."""
        with self._cache_lock:
            return dict(self._last_state or {})

    def read_mode(self) -> str | None:
        """The T1's current whole-body mode name, or ``None`` if unreadable.

        Reported rather than raised: :meth:`get_status` must answer for a robot
        that is not talking, and a mode read is one RPC that can time out on a
        healthy robot (the vendor's client retries code 100 for exactly that).
        """
        if self._client is None:
            return None
        try:
            response = self._client.GetMode()
        except (RuntimeError, OSError) as exc:
            logger.debug("%s: GetMode failed: %s", self._tool_name, exc)
            return None
        mode = getattr(response, "mode", None)
        return getattr(mode, "name", None) if mode is not None else None

    def _on_battery(self, msg: Any) -> None:
        """Absorb one ``BatteryState``. Never raises - the SDK owns this thread.

        The SDK names the charge field ``soc`` and documents no scale, so the
        driver passes it through as the shared ``battery_pct`` and gates
        *nothing* on it. A floor compared against an unverified scale is worse
        than no floor: on a 0..1 scale it refuses every frame, on a 0..100 scale
        it refuses none, and both look like a working safety check.
        """
        try:
            reading = {
                "pct": float(getattr(msg, "soc", 0.0)),
                "voltage": float(getattr(msg, "voltage", 0.0)),
                "current": float(getattr(msg, "current", 0.0)),
            }
        except (AttributeError, TypeError, ValueError):
            logger.debug("%s: unreadable BatteryState frame", self._tool_name, exc_info=True)
            return
        with self._cache_lock:
            self._battery = reading

    def _on_fall_state(self, msg: Any) -> None:
        """Absorb one ``FallDownState``, resolving it to a name.

        The gate reads on *evidence of a fall*, not on the absence of a reading:
        a T1 whose fall topic is silent leaves :attr:`_fall_state` ``None`` and
        keeps writing, because refusing on absence would refuse every frame on a
        robot that simply does not publish it.
        """
        state = getattr(msg, "fall_down_state", None)
        if state is None:
            logger.debug("%s: FallDownState frame carried no state", self._tool_name)
            return
        try:
            code = int(getattr(state, "value", state))
        except (AttributeError, TypeError, ValueError):
            logger.debug("%s: unreadable FallDownState frame", self._tool_name, exc_info=True)
            return
        name = FALL_STATE_NAMES.get(code)
        if name is None:
            logger.debug("%s: unknown fall state %s", self._tool_name, code)
            return
        with self._cache_lock:
            self._fall_state = name

    def _on_low_state(self, msg: Any) -> None:
        """Absorb one ``LowState`` from the SDK's subscriber thread.

        Never raises: this runs on a thread the SDK owns, where an exception
        would kill the subscription and leave the driver silently blind.
        """
        try:
            snapshot = parse_low_state(msg, self._state_field)
        except (AttributeError, TypeError, ValueError):
            logger.debug("%s: unreadable LowState frame", self._tool_name, exc_info=True)
            return
        joints = snapshot.get("joints") or []
        if not joints:
            # A frame carrying no motors is not a frame. Caching it would erase
            # the last good positions - which are the hold source every
            # subsequent :meth:`send_action` builds its frame from - so an
            # uncommanded arm joint would be dropped by the very next write.
            logger.debug("%s: LowState carried no %s motors", self._tool_name, self._state_field)
            return
        if any(not math.isfinite(value) for value in joints):
            logger.debug("%s: LowState carried a non-finite joint position", self._tool_name)
            return
        with self._cache_lock:
            self._last_state = snapshot
