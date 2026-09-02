"""Native Franka FCI driver for the Panda and the Research 3.

``Robot("panda", mode="real", driver="strands", port="172.16.0.2")`` builds one
of these. The instance satisfies
:class:`~strands_robots.drivers.base.HardwareDriver` structurally, so
:func:`~strands_robots.robot.Robot` returns it and the mesh, the teleop rail and
the agent tool surface consume it exactly like the lerobot driver they replace
for these robots. There is nothing to replace in practice: lerobot registers no
Franka robot type, so before this driver every Franka arm in the registry was
simulation-only and ``mode="real"`` refused all three by name.

The link is the **Franka Control Interface** (FCI), reached through
`panda-py <https://github.com/JeanElsner/panda-py>`_ - an MIT-licensed binding
over libfranka. FCI is a hard-realtime UDP link from the workstation to the arm's
control box: libfranka receives a robot state every millisecond and expects a
command back inside the same tick, which is why the vendor SDK owns that loop and
this driver does not attempt to. What the driver owns is the *decode* of a state
into this package's joint vocabulary, the *cadence* a telemetry consumer reads at,
and the *envelope* a motion command must satisfy before libfranka is asked to
move a 3 kg payload.

What it actually does:

* Resolves ``panda_py`` inside :meth:`FrankaDriver.connect_eagerly` (never at
  module import), the same shape as
  :func:`strands_robots.drivers.g1._resolve_message_class`: a failure hands back
  a reason string and leaves the driver constructed-but-disconnected rather than
  raising ``ModuleNotFoundError`` through the agent tool surface. The module
  therefore imports on any machine, and every test here runs against a fake FCI.
* Decodes a libfranka ``RobotState`` into the joint vocabulary *this arm's own
  simulated model uses*, so an action dict authored against the simulated arm
  commands the real one unchanged. FCI carries an unnamed seven-element vector,
  so the names are this package's choice, and the choice is each arm's MuJoCo
  asset: the Panda's joints are ``joint1..joint7`` while the FR3 assets prefix
  theirs (``fr3_joint1..``, ``fr3v2_joint1..``). One shared vocabulary would have
  been simpler and wrong for two of the three arms - see :data:`JOINT_PREFIXES`.
  Positions, velocities and measured link-side torques come from one state read,
  and the gripper width from the Franka Hand's own state.
* Reports the FCI **downsample stride**: the arm sources state at
  :data:`FCI_RATE_HZ`, a mesh publishes at tens of hertz, and
  :func:`downsample_stride` is the ratio between them. It is reported rather than
  hidden because a consumer that reads every tick is a consumer that will miss
  its own deadline.
* Gates a motion command before libfranka sees it: this driver fronts this robot,
  it is connected, the action names joints this arm has, every value is finite,
  and a joint-space command names *all seven* joints. Then, and only then,
  ``panda_py``'s guarded motion generator is asked to move.

Deliberately absent, so a reader is not left guessing:

* **No 1 kHz control loop and no torque control.** Streaming a torque command at
  the FCI rate is what libfranka's realtime context exists for; a Python thread
  that misses a tick triggers a reflex stop on the arm. Joint-space motion goes
  through ``panda_py``'s own motion generator, which owns the realtime loop and
  enforces the arm's limits itself.
* **No joint-limit table.** The Panda and the FR3 have different limits and this
  repository has neither arm to measure, so the authority is libfranka - it
  refuses an out-of-envelope target and the refusal is reported verbatim.
  Restating published numbers here would put a safety envelope in the tree that
  nothing verified against the arm it guards.
* **No cartesian control and no kinematics.** ``O_T_EE`` is on the state this
  driver decodes and is deliberately not published as a pose: the mesh's
  ``_pose`` is a base pose, and an end-effector transform published under that
  name would be a wrong answer rather than a missing one.
* ``start_task`` / ``run_policy`` refuse. Both need a policy provider that emits
  Franka-shaped actions, which the provider registry does not carry yet; the
  refusal names that rather than standing in for it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from strands_robots.registry import resolve_name
from strands_robots.utils import finite_number_error, positive_finite_number_error

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The Franka arms this driver serves, as canonical registry names. One driver
#: for three product generations because FCI is the same interface on all of
#: them: the differences (reach, limits, payload) live in the control box's own
#: firmware, which libfranka reads off the arm rather than being told.
SUPPORTED_ROBOTS: tuple[str, ...] = ("panda", "fr3", "fr3_v2")

#: The rate the control box sources robot state at, in hertz. Fixed by FCI, not
#: a tunable: libfranka receives one state per millisecond and a command is due
#: back within the same tick.
FCI_RATE_HZ: int = 1000

#: Cadence a telemetry consumer reads this driver at, in hertz, when the caller
#: names none. Matches the teleop publisher's rate, which is the fastest
#: consumer in this package.
DEFAULT_STREAM_RATE_HZ: float = 30.0

#: Arm joints on every Franka generation. FCI reports them as one fixed-length
#: vector, so a state carrying any other length is not the state it is taken for.
DOF: int = 7

#: Canonical robot name -> the prefix its joints are named with. Read off each
#: arm's own MuJoCo asset, because that is the vocabulary a caller who developed
#: against the simulated arm already writes: ``panda.xml`` names its joints
#: ``joint1..joint7``, ``fr3.xml`` names them ``fr3_joint1..``, and
#: ``fr3v2.xml`` names them ``fr3v2_joint1..``.
#:
#: One shared vocabulary across the family would be simpler and wrong for two of
#: the three arms: an action dict authored against the simulated FR3 would name
#: joints the driver did not accept, and the caller would be told their own
#: robot's joint names are unknown keys.
JOINT_PREFIXES: dict[str, str] = {"panda": "", "fr3": "fr3_", "fr3_v2": "fr3v2_"}

#: Action key naming the Franka Hand's commanded aperture, in metres. Not a
#: joint: the Hand is a separate FCI device with its own state, so it is
#: commanded and reported beside the arm rather than as an eighth joint. Shared
#: across the family - the Hand is the same product on all three arms.
GRIPPER_KEY: str = "gripper_width"

#: ``panda_py``'s ``speed_factor`` when the caller names none: a fifth of the
#: arm's rated speed. Conservative on purpose - the default must be a speed that
#: is safe in a cell whose layout this driver knows nothing about, and a caller
#: who has measured their cell raises it explicitly.
DEFAULT_SPEED_FACTOR: float = 0.2

#: The binding every FCI touch here goes through, imported lazily.
_PANDA_PY_MODULE = "panda_py"

#: Refusal shared by the two policy verbs. Quoted in tests, so a change here is
#: a change to the driver's contract.
_NO_POLICY_PROVIDER = (
    "no policy provider emits Franka-shaped actions yet - "
    "the provider registry (strands_robots.registry.policies) carries none for "
    f"{', '.join(SUPPORTED_ROBOTS)}. Command joint targets with send_action() in the meantime"
)

_TOOL_TYPE = "robot"


# --------------------------------------------------------------------------- #
# Pure helpers. Every one is total - it answers with a value or a reason - so a #
# refusal boundary can render it into an envelope without a try block, and a   #
# test can grade it with no FCI at all.                                        #
# --------------------------------------------------------------------------- #
def joint_names_for(robot_name: str) -> tuple[str, ...]:
    """Return the joint names ``robot_name``'s own model uses.

    Args:
        robot_name: A canonical registry name or any alias of one -
            :func:`~strands_robots.registry.resolve_name` is applied, so
            ``"franka"`` and ``"panda"`` answer the same names.

    Returns:
        :data:`DOF` names, in FCI wire order, prefixed per
        :data:`JOINT_PREFIXES`. A name with no entry there gets the unprefixed
        spelling: this driver is registered for exactly
        :data:`SUPPORTED_ROBOTS`, so an unlisted name reaching here is a caller
        holding the class directly, and the Panda vocabulary is the least
        surprising answer for them - not a refusal, because there is no wrong
        answer to give about an arm this table does not describe.
    """
    prefix = JOINT_PREFIXES.get(resolve_name(robot_name), "")
    return tuple(f"{prefix}joint{index}" for index in range(1, DOF + 1))


def action_keys_for(joint_names: Sequence[str]) -> tuple[str, ...]:
    """Return every key an action for ``joint_names`` may carry.

    Args:
        joint_names: The arm's joint names, from :func:`joint_names_for`.

    Returns:
        The joint names followed by :data:`GRIPPER_KEY`. Derived rather than
        stored so the accepted set and the arm's vocabulary cannot disagree -
        a stored copy is the one that drifts.
    """
    return (*joint_names, GRIPPER_KEY)


def downsample_stride(stream_rate_hz: float) -> int | str:
    """Return how many FCI ticks make one published frame, or a reason.

    The arm sources state at :data:`FCI_RATE_HZ` and a consumer reads at tens of
    hertz, so a published frame stands for a whole run of ticks. Naming the ratio
    is what lets a consumer size its own budget; a consumer that reads every tick
    is a consumer that misses its deadline.

    Args:
        stream_rate_hz: The cadence a consumer wants frames at, in hertz.

    Returns:
        The stride, at least ``1``, or a reason naming why the rate is not one
        this link can serve. A rate above :data:`FCI_RATE_HZ` is refused rather
        than clamped: the answer would be a stride of 1 delivering 1 kHz, which
        is not the rate that was asked for, and a caller told "fine" who then
        measures 1 kHz has been given a wrong answer.
    """
    if (reason := positive_finite_number_error(stream_rate_hz, "stream_rate_hz", "downsample_stride")) is not None:
        return reason
    rate = float(stream_rate_hz)
    if rate > FCI_RATE_HZ:
        return (
            f"downsample_stride: stream_rate_hz={rate} exceeds the FCI state rate of "
            f"{FCI_RATE_HZ} Hz - the link cannot source frames faster than the control box sends them"
        )
    return max(1, round(FCI_RATE_HZ / rate))


def _read_joint_vector(state: Any, field: str, joint_names: Sequence[str]) -> list[float] | str:
    """Read one :data:`DOF`-element float field off a libfranka robot state.

    Args:
        state: The object ``panda_py.Panda.get_state()`` returned.
        field: The libfranka field name (``q``, ``dq``, ``tau_J``).
        joint_names: The names the values will be keyed by, quoted in the
            length refusal so a caller sees the vocabulary that was expected.

    Returns:
        The values as floats, or a reason naming the field and what arrived
        instead. A short vector is the failure that matters: libfranka reports a
        fixed :data:`DOF`-element array, so anything else means the object being
        decoded is not the state it is taken for, and mapping it onto the joint
        names positionally would silently attribute one joint's number to
        another.
    """
    raw = getattr(state, field, None)
    if raw is None:
        return f"robot state carries no {field!r} - not a libfranka RobotState"
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        return f"robot state {field!r} is {type(raw).__name__}, expected a sequence of {DOF} floats"
    if len(raw) != len(joint_names):
        return f"robot state {field!r} has {len(raw)} values, expected {len(joint_names)} ({', '.join(joint_names)})"
    values: list[float] = []
    for index, value in enumerate(raw):
        if (reason := finite_number_error(value, f"{field}[{index}]", "decode_robot_state")) is not None:
            return reason
        values.append(float(value))
    return values


def decode_robot_state(state: Any, joint_names: Sequence[str], gripper_state: Any = None) -> dict[str, Any] | str:
    """Decode an FCI state into this package's joint vocabulary, or a reason.

    Three of libfranka's fields are read: ``q`` (measured joint positions, rad),
    ``dq`` (measured joint velocities, rad/s) and ``tau_J`` (measured link-side
    joint torques, Nm). All three are :data:`DOF`-element arrays in the same
    order, so all three are keyed by the same names and a consumer can zip them.

    Args:
        state: What ``panda_py.Panda.get_state()`` returned.
        joint_names: The arm's joint names, from :func:`joint_names_for`. Passed
            in rather than read off a module constant, because the vocabulary is
            per-arm (:data:`JOINT_PREFIXES`) and a decode that guessed it would
            key an FR3's state under the Panda's names.
        gripper_state: What ``panda_py.libfranka.Gripper.read_once()`` returned,
            or ``None`` for an arm with no Franka Hand attached. The gripper is a
            separate FCI device, so a missing one is a smaller snapshot rather
            than a failure.

    Returns:
        A snapshot mapping, or a reason string naming the field that could not be
        decoded. ``gripper_max_width`` is the Hand's *reported* stroke rather
        than a constant in this module: the Hand measures it during its own
        homing, so reading it is the only way to know it for the hand that is
        actually bolted on.
    """
    decoded: dict[str, dict[str, float]] = {}
    for field, section in (("q", "joints"), ("dq", "velocities"), ("tau_J", "torques")):
        values = _read_joint_vector(state, field, joint_names)
        if isinstance(values, str):
            return f"decode_robot_state: {values}"
        decoded[section] = dict(zip(joint_names, values, strict=True))

    snapshot: dict[str, Any] = dict(decoded)
    snapshot["gripper_width"] = None
    snapshot["gripper_max_width"] = None
    if gripper_state is not None:
        for attribute, key in (("width", "gripper_width"), ("max_width", "gripper_max_width")):
            raw = getattr(gripper_state, attribute, None)
            if raw is None:
                continue
            if (reason := finite_number_error(raw, attribute, "decode_robot_state")) is not None:
                return f"decode_robot_state: gripper {reason}"
            snapshot[key] = float(raw)
    return snapshot


def action_to_targets(
    action: Mapping[str, Any], joint_names: Sequence[str]
) -> tuple[list[float] | None, float | None] | str:
    """Split an action dict into a joint target and a gripper width, or a reason.

    Args:
        action: Joint targets keyed by ``joint_names``, and/or
            :data:`GRIPPER_KEY` in metres.
        joint_names: The arm's joint names, from :func:`joint_names_for`.

    Returns:
        ``(joint_target, gripper_width)`` - either half may be ``None``, so a
        gripper-only or arm-only command is expressible - or a reason string.

        A *partial* joint command is refused rather than completed from the
        current pose. FCI joint control commands a whole configuration, so the
        unnamed joints would have to be filled in from somewhere, and filling
        them from the arm's present pose turns "move joint4" into a seven-joint
        motion the caller did not write. Naming the missing joints costs the
        caller one line and costs nobody a surprise trajectory.
    """
    accepted = action_keys_for(joint_names)
    if not isinstance(action, Mapping) or not action:
        return f"action must be a non-empty mapping keyed by {', '.join(accepted)}, got {action!r}"
    unknown = sorted(set(action) - set(accepted))
    if unknown:
        return f"action names {unknown} which this arm does not have - expected any of {list(accepted)}"
    for key, value in action.items():
        if (reason := finite_number_error(value, key, "send_action")) is not None:
            return reason

    named = [name for name in joint_names if name in action]
    if named and len(named) != len(joint_names):
        missing = [name for name in joint_names if name not in action]
        return (
            f"a joint-space command names all {len(joint_names)} joints; {missing} are missing. "
            "FCI commands a whole configuration, so the unnamed joints would be filled in from the "
            "arm's current pose - a motion you did not ask for"
        )
    joint_target = [float(action[name]) for name in joint_names] if named else None
    gripper = float(action[GRIPPER_KEY]) if GRIPPER_KEY in action else None
    return joint_target, gripper


def _resolve_panda_py() -> Any:
    """Return the ``panda_py`` module, or a reason naming what failed.

    The same shape as :func:`strands_robots.drivers.g1._resolve_message_class`
    and :func:`strands_robots.drivers.reachy._resolve_transport`: the seam's other
    drivers resolve their vendor SDK through a helper that hands back a reason
    string, and every refusal boundary turns that string into a named failure.
    That is what keeps this driver's no-raise contract intact when the binding is
    absent - ``connect_eagerly`` reports a reason and leaves the driver
    disconnected but usable.

    Returns:
        The imported module, or a reason string naming it, the cause, and the
        install that supplies it.
    """
    try:
        import importlib

        return importlib.import_module(_PANDA_PY_MODULE)
    except ImportError as exc:
        return (
            f"cannot import {_PANDA_PY_MODULE}: {exc}. The Franka Control Interface is reached through "
            "the panda-py binding over libfranka - install it with: pip install panda-py"
        )


def _refuse(reason: str) -> dict[str, Any]:
    """Return the driver's error envelope with ``reason`` inside.

    Args:
        reason: Text naming what refused.

    Returns:
        The error envelope, in the one shape every refusal path here renders.
    """
    return {"status": "error", "content": [{"text": reason}]}


class FrankaDriver:
    """Native FCI driver for the arms in :data:`SUPPORTED_ROBOTS`.

    Constructor contract matches
    :class:`~strands_robots.drivers.base.HardwareDriver`: the factory builds
    every native driver as ``driver_cls(tool_name=..., cameras=...,
    data_config=..., **kwargs)`` and forwards the caller's extras in ``kwargs``.
    Franka-specific keywords land there:

    * ``port`` - the control box's hostname or IP, as printed in Desk (a Franka
      install's is typically ``172.16.0.2``). ``port`` rather than ``hostname``
      because that is the keyword the factory forwards and the base contract
      keeps polymorphic: a serial path for a servo bus, an address here.
    * ``stream_rate_hz`` - the telemetry cadence, default
      :data:`DEFAULT_STREAM_RATE_HZ`. Refused at construction if it is not a rate
      this link can serve, because a driver built with an impossible cadence is a
      driver whose every read is wrong.
    * ``speed_factor`` - ``panda_py``'s motion speed scaling, default
      :data:`DEFAULT_SPEED_FACTOR`. Must be in ``(0, 1]``.
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
        # The arm's own vocabulary, resolved once: an FR3 names its joints
        # differently from a Panda (:data:`JOINT_PREFIXES`), and every read and
        # every command below must speak the one this arm's model speaks.
        self._joint_names = joint_names_for(tool_name)
        port = kwargs.pop("port", None)
        self._hostname: str | None = str(port) if port else None

        rate = kwargs.pop("stream_rate_hz", DEFAULT_STREAM_RATE_HZ)
        stride = downsample_stride(rate)
        if isinstance(stride, str):
            raise ValueError(f"FrankaDriver({tool_name!r}): {stride}")
        self._stream_rate_hz = float(rate)
        self._stride = stride

        speed_factor = kwargs.pop("speed_factor", DEFAULT_SPEED_FACTOR)
        if (reason := positive_finite_number_error(speed_factor, "speed_factor", "FrankaDriver")) is not None:
            raise ValueError(reason)
        if float(speed_factor) > 1.0:
            raise ValueError(
                f"FrankaDriver({tool_name!r}): speed_factor must be in (0, 1], got {speed_factor} - "
                "it scales the arm's rated joint speed, so a value above 1 asks for a speed it does not have"
            )
        self._speed_factor = float(speed_factor)

        self._panda: Any = None
        self._gripper: Any = None
        self._connected = False
        self._connect_error: str | None = None
        # Two locks, because one FCI touch is not like the others.
        #
        # ``_lock`` covers the short ones - a state read, a handle swap - since
        # libfranka's Robot is not safe to enter twice at once and the mesh reads
        # telemetry on its own thread.
        #
        # ``_motion_lock`` is held across the blocking motion alone. It
        # serialises motions against each other, and it is deliberately not the
        # lock a halt or a state read takes: ``move_to_joint_position`` runs the
        # whole trajectory before returning, so a halt that waited on it would
        # land after the motion it was meant to interrupt, and telemetry would go
        # blank for the motion's full duration.
        #
        # Ordering rule: a thread never waits for ``_motion_lock`` while holding
        # ``_lock``. The command path snapshots the handles under ``_lock`` and
        # releases it before taking ``_motion_lock``; ``cleanup`` takes them in
        # the other order, which is safe only because of that rule.
        self._lock = threading.RLock()
        self._motion_lock = threading.Lock()
        # Extras are kept rather than refused: a downstream driver package may
        # consume them, and refusing them here would refuse a valid extension.
        self._extras = kwargs

    # ------------------------------------------------------------------ #
    # Tool surface.                                                      #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """Name the agent invokes this robot by."""
        return self._tool_name

    @property
    def is_connected(self) -> bool:
        """Whether the FCI link is live, for the mesh joint read."""
        return self._connected and self._panda is not None

    @property
    def tool_spec(self) -> ToolSpec:
        """A minimal agent-facing spec: read sensors, report status, stop.

        Motion is deliberately not an agent verb. A joint-space Franka command is
        seven numbers whose safety depends on the cell the arm stands in, and the
        command path (``send_action``) is where a caller who knows that cell
        already reaches it. Declaring a verb an agent would have to guess seven
        radians for is worse than not declaring it.
        """
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    f"Franka FCI native driver for {self._tool_name}: reads joint positions, velocities "
                    "and measured torques plus the Franka Hand's width over the Franka Control Interface, "
                    "reports link status, and stops the arm. Joint motion goes through send_action."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "sensors: one FCI state read - joints, velocities, torques, gripper width; "
                                    "status: link reachability, cadence and configuration; "
                                    "stop: stop the arm's motion"
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

        The ``sensors`` verb reports the FCI downsample stride beside the
        snapshot: the state it returns stands for :attr:`downsample_stride` ticks
        of the 1 kHz link, and a consumer sizing a budget needs to know that the
        frame it holds is one of a run rather than the only one there was.
        """
        del kwargs, invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        action = (tool_use.get("input") or {}).get("action", "sensors")
        envelope: dict[str, Any]
        if action == "sensors":
            snapshot = self.read_state()
            if isinstance(snapshot, str):
                envelope = _refuse(f"sensors: {snapshot}")
            else:
                envelope = {
                    "status": "success",
                    "content": [
                        {
                            "json": {
                                **snapshot,
                                "fci_rate_hz": FCI_RATE_HZ,
                                "stream_rate_hz": self._stream_rate_hz,
                                "downsample_stride": self._stride,
                            }
                        }
                    ],
                }
        elif action == "status":
            envelope = {"status": "success", "content": [{"json": await self.get_status()}]}
        else:  # "stop"
            envelope = self.stop_task()
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Reads.                                                             #
    # ------------------------------------------------------------------ #

    @property
    def joint_names(self) -> tuple[str, ...]:
        """The arm's joint names, in FCI wire order.

        Public because a caller building an action dict needs the vocabulary this
        arm accepts, and reading it off the driver is what stops them guessing a
        prefix. The discovery surface behind ``action names [...] which this arm
        does not have``.
        """
        return self._joint_names

    @property
    def downsample_stride(self) -> int:
        """FCI ticks per published frame at this driver's configured cadence."""
        return self._stride

    def read_state(self) -> dict[str, Any] | str:
        """Read one FCI state and decode it, or return a reason.

        One read of the arm and one of the Hand, under the driver's state lock -
        not the motion lock, so telemetry keeps answering while the arm moves -
        then :func:`decode_robot_state`. On demand rather than from a background poll:
        the 1 kHz loop belongs to libfranka's realtime context, and a Python
        thread standing in for it would be a second, slower copy of the state
        with no way to say how stale it is.

        Returns:
            The decoded snapshot, or a reason naming why there is none. A
            disconnected driver is a reason rather than an exception - the mesh
            reads this on its own schedule and a driver whose arm is off must
            report "no state" without taking the publisher down.
        """
        if not self.is_connected:
            return f"not connected to {self._hostname or 'any FCI host'} - call connect_eagerly() first"
        try:
            with self._lock:
                state = self._panda.get_state()
                gripper_state = self._gripper.read_once() if self._gripper is not None else None
        except (OSError, RuntimeError) as exc:
            return f"FCI state read failed: {exc}"
        return decode_robot_state(state, self._joint_names, gripper_state)

    def get_observation(self) -> dict[str, float]:
        """Report joint positions the way a lerobot observation spells them.

        Present so the mesh publishes this arm's joints:
        :func:`strands_robots.bus_access.joint_read_source` admits a native driver
        that answers either a motor-bus read or this call, and a driver with
        neither publishes no ``joints`` topic while every other section of its
        telemetry appears - which reads as a healthy arm reporting nothing.

        Returns:
            ``{"<joint>.pos": radians}`` for the seven arm joints, plus
            ``"gripper.pos"`` in metres when a Franka Hand answered. Empty when
            the link is down or the state could not be decoded, because "no
            joints" is the answer a telemetry consumer handles and an exception
            is the one it does not.
        """
        snapshot = self.read_state()
        if isinstance(snapshot, str):
            logger.debug("Franka observation unavailable: %s", snapshot)
            return {}
        observation = {f"{name}.pos": value for name, value in snapshot["joints"].items()}
        if snapshot.get("gripper_width") is not None:
            observation["gripper.pos"] = float(snapshot["gripper_width"])
        return observation

    # ------------------------------------------------------------------ #
    # Command path.                                                      #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
        """Command a joint configuration and/or a gripper width.

        Gates, in order: this driver fronts this robot; the link is live; the
        action names only keys this arm has, all finite, with a joint command
        naming all seven joints. Only then is libfranka asked to move, through
        ``panda_py``'s guarded motion generator - which owns the realtime loop and
        enforces the arm's own joint limits, so those limits are not restated
        here and its refusal is reported verbatim.

        That refusal has to be collected rather than caught. ``panda_py`` runs the
        trajectory on its own realtime thread and catches ``franka::Exception``
        there, parking it for ``raise_error()``, so the motion call returns
        normally after a reflex stop or an out-of-limit target and reports the
        outcome as a ``bool`` instead. Both are read here: the parked error is
        drained after every motion, and a motion that ended away from the goal is
        refused rather than reported as the configuration the arm holds.

        The blocking motion is run outside the state lock, so a halt issued while
        the arm is moving preempts it instead of queueing behind it - see
        :meth:`stop_task`.

        Args:
            action: Joint targets in radians keyed by :attr:`joint_names`, and/or
                :data:`GRIPPER_KEY` in metres.
            robot_name: Which robot to command; ``None`` or this driver's own
                name means this arm, anything else is refused.

        Returns:
            A success envelope naming what was commanded, or an error envelope
            naming which gate refused and why.
        """
        if robot_name is not None and robot_name != self._tool_name:
            return _refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        if not self.is_connected:
            return _refuse(f"send_action: not connected to {self._hostname or 'any FCI host'}")
        targets = action_to_targets(action, self._joint_names)
        if isinstance(targets, str):
            return _refuse(f"send_action: {targets}")
        joint_target, gripper_width = targets
        if gripper_width is not None and self._gripper is None:
            return _refuse(
                f"send_action: {GRIPPER_KEY} was commanded but no Franka Hand answered at "
                f"{self._hostname!r} - connect_eagerly() reports which devices are live"
            )

        panda, gripper = self._live_handles()
        if panda is None:
            return _refuse(f"send_action: not connected to {self._hostname or 'any FCI host'}")

        commanded: dict[str, Any] = {}
        try:
            with self._motion_lock:
                if joint_target is not None:
                    named = dict(zip(self._joint_names, joint_target, strict=True))
                    reached = panda.move_to_joint_position(joint_target, speed_factor=self._speed_factor)
                    # Drain the control thread's parked error first. panda_py's
                    # controller catches franka::Exception inside the realtime
                    # loop and holds it for raise_error(), so the motion call
                    # itself never raises for a reflex stop or an out-of-limit
                    # target. Drained after every motion, so a parked error is
                    # reported against the motion that caused it rather than
                    # against whichever command runs next.
                    panda.raise_error()
                    if reached is False:
                        # The motion generator's own verdict: the arm finished
                        # away from the goal. Compared against False rather than
                        # falsiness so a binding that returns nothing is not read
                        # as a failure it did not report.
                        return _refuse(
                            f"send_action: the arm moved but did not reach the commanded configuration "
                            f"{named} - libfranka's motion generator reports the goal was not met, so "
                            "the arm is somewhere between where it was and where it was asked to be"
                        )
                    commanded["joints"] = named
                if gripper_width is not None:
                    # The Hand's own verdict, read for the same reason as the
                    # arm's: it reports False for a width it did not reach, which
                    # is what a grasp that closed on an object looks like.
                    if gripper.move(gripper_width, self._gripper_speed()) is False:
                        return _refuse(
                            f"send_action: the Hand did not reach {gripper_width} m - it reports the "
                            "commanded width was not met, which is what it reports when the fingers "
                            "closed on an object instead"
                        )
                    commanded[GRIPPER_KEY] = gripper_width
        except (OSError, RuntimeError) as exc:
            # libfranka's own refusal - an out-of-limit target, a reflex stop, a
            # dropped link. Reported verbatim: it names the limit that was hit,
            # which is more than any envelope in this module could establish.
            return _refuse(f"send_action: FCI refused the command: {exc}")
        return {"status": "success", "content": [{"json": {"commanded": commanded, "robot": self._tool_name}}]}

    def _live_handles(self) -> tuple[Any, Any]:
        """Snapshot the arm and Hand handles under the short lock.

        Taken so a caller can use them outside the lock: the strong references
        keep the devices alive even if :meth:`cleanup` clears the attributes
        meanwhile, which is what lets the blocking motion run without holding a
        lock a halt would then have to wait for.

        Returns:
            ``(arm, hand)``, either of which is ``None`` when not connected.
        """
        with self._lock:
            return self._panda, self._gripper

    def _halt(self, panda: Any, gripper: Any) -> str | None:
        """Interrupt whatever the arm is doing, and the Hand with it.

        Through ``libfranka``'s own ``Robot::stop()`` - the call designed to be
        made from another thread to abort a running control loop. ``panda_py``
        exposes it on the object ``Panda.get_robot()`` returns; the ``Panda``
        wrapper itself has no halt of its own, only ``stop_controller()``, which
        ends a torque controller rather than the motion generator this driver
        commands.

        Args:
            panda: The live arm handle.
            gripper: The live Hand handle, or ``None`` when none answered.

        Returns:
            ``None`` on success, or a reason naming what refused. A reason and
            never an exception: this is the halt path, and every caller of it -
            the agent ``stop`` verb, a task stop, shutdown - is a place where a
            raise would replace a stopped arm with a traceback.
        """
        try:
            get_robot = panda.get_robot
        except AttributeError:
            return (
                "this panda_py binding exposes no get_robot(), so libfranka's own "
                "Robot.stop() cannot be reached and the arm cannot be halted through it"
            )
        try:
            get_robot().stop()
            if gripper is not None:
                # The Hand runs its own motion; an arm that stopped while the
                # fingers keep closing is a partial halt.
                gripper.stop()
        except (AttributeError, OSError, RuntimeError) as exc:
            # AttributeError included on purpose: a binding whose surface differs
            # from the one measured here must be reported, because a halt verb
            # that raises is worse than one that says why it could not halt.
            return f"FCI stop failed: {exc}"
        return None

    def _gripper_speed(self) -> float:
        """Gripper closing speed in m/s, scaled by this driver's speed factor.

        The Hand's rated speed is 0.1 m/s, and scaling it by the same factor the
        arm moves at keeps one knob for "how fast is this robot allowed to be"
        rather than two that can disagree.
        """
        return 0.1 * self._speed_factor

    # ------------------------------------------------------------------ #
    # Task and policy paths - no Franka policy provider yet, so refuse.  #
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
        """Refuse: no provider emits Franka-shaped actions yet."""
        del instruction, policy_port, policy_host, policy_provider, duration, policy_kwargs
        return _refuse(f"start_task: {_NO_POLICY_PROVIDER}")

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refuse: no provider emits Franka-shaped actions yet."""
        del policy_object, instruction, duration, n_steps
        return _refuse(f"run_policy: {_NO_POLICY_PROVIDER}")

    def get_task_status(self) -> dict[str, Any]:
        """Report that no policy task is in flight, and why none can be."""
        return {
            "status": "success",
            "content": [{"json": {"in_flight": False, "reason": _NO_POLICY_PROVIDER}}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Stop the arm's motion, reporting the outcome.

        Also the agent ``stop`` verb's implementation, so the halt an agent asks
        for and the halt a task stop performs are the same call rather than two
        that can diverge.

        Preempts a motion in flight. It takes the state lock only long enough to
        read the handles and never the motion lock, so it reaches
        ``libfranka``'s ``Robot::stop()`` while the control loop that call exists
        to abort is still running. A halt that waited on the motion would report
        success for an arm that had already finished moving on its own.

        Returns:
            A success envelope, or an error envelope naming why the arm could not
            be halted. Never raises: this verb is reached from ``stream``, and an
            exception there is a stop request that produced a traceback instead of
            a stopped arm.
        """
        if not self.is_connected:
            return _refuse(f"stop_task: not connected to {self._hostname or 'any FCI host'}")
        panda, gripper = self._live_handles()
        if panda is None:
            return _refuse(f"stop_task: not connected to {self._hostname or 'any FCI host'}")
        reason = self._halt(panda, gripper)
        if reason is not None:
            return _refuse(f"stop_task: {reason}")
        return {"status": "success", "content": [{"text": f"stop_task: {self._tool_name} motion stopped"}]}

    # ------------------------------------------------------------------ #
    # Lifecycle.                                                         #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Open the FCI link, and the Franka Hand's beside it.

        Returns ``None`` on success. Off hardware - no binding installed, no
        control box at :attr:`_hostname` - returns a reason naming what did not
        answer and leaves the driver usable: every read reports that reason and
        every write refuses "not connected". Idempotent: a second call on a live
        link returns ``None`` without touching the arm.

        A Hand that does not answer is **not** a connect failure. It is a
        separate FCI device and an arm can legitimately run without one, so the
        arm stays connected, the gripper stays ``None``, and a later
        gripper command refuses by name rather than the whole arm being refused
        at connect time.

        Returns:
            ``None`` when the arm is connected, or a reason string.
        """
        if self.is_connected:
            return None
        if not self._hostname:
            return (
                f"FrankaDriver({self._tool_name!r}): no FCI host - pass port='<control box IP>' "
                "(the address Franka Desk is served on, typically 172.16.0.2)"
            )
        panda_py = _resolve_panda_py()
        if isinstance(panda_py, str):
            self._connect_error = panda_py
            return panda_py
        try:
            with self._lock:
                panda = panda_py.Panda(self._hostname)
        except (OSError, RuntimeError) as exc:
            reason = (
                f"FCI connect to {self._hostname!r} failed: {exc}. Check the arm is unlocked in Desk, "
                "FCI is activated, and this host is on the arm's network"
            )
            self._connect_error = reason
            return reason
        gripper: Any = None
        try:
            with self._lock:
                gripper = panda_py.libfranka.Gripper(self._hostname)
        except (AttributeError, OSError, RuntimeError) as exc:
            logger.info("Franka Hand did not answer at %r (%s); arm connected without a gripper", self._hostname, exc)
        self._panda = panda
        self._gripper = gripper
        self._connected = True
        self._connect_error = None
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report the link, the cadence and the configuration.

        Returns:
            A status envelope the mesh publishes as this peer's presence. Shaped
            like the other native drivers' so every peer publishes identically;
            fields an FCI arm has and a servo bus does not (the stride, the
            gripper) are simply present here and absent there.
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
                        "hostname": self._hostname,
                        "gripper": self._gripper is not None,
                        "fci_rate_hz": FCI_RATE_HZ,
                        "stream_rate_hz": self._stream_rate_hz,
                        "downsample_stride": self._stride,
                        "speed_factor": self._speed_factor,
                        "joint_names": list(self._joint_names),
                        "supported_robots": list(SUPPORTED_ROBOTS),
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Stop motion, leaving the link open.

        The lifecycle half of :meth:`stop_task`: same call, no envelope, because
        a caller shutting down has nothing to do with a verdict.
        """
        if not self.is_connected:
            return
        panda, gripper = self._live_handles()
        if panda is None:
            return
        reason = self._halt(panda, gripper)
        if reason is not None:
            logger.warning("Franka stop failed on %r: %s", self._tool_name, reason)

    def cleanup(self) -> None:
        """Release the FCI link and the Hand.

        libfranka holds a realtime UDP socket and a control-box session for as
        long as its ``Robot`` lives, and the control box admits one. A driver
        that dropped its references without closing would leave the arm
        unreachable to the next process until the session timed out.
        """
        # The motion lock first: a link closed under a running control loop
        # leaves the control box with a session it never ended, so shutdown waits
        # for an in-flight motion. This is the one path that waits on it, and it
        # is not a halt - a caller wanting the motion to end now calls stop().
        with self._motion_lock, self._lock:
            for device, label in ((self._gripper, "gripper"), (self._panda, "arm")):
                if device is None:
                    continue
                closer = getattr(device, "close", None)
                if closer is None:
                    continue
                try:
                    closer()
                except (OSError, RuntimeError) as exc:
                    logger.warning("Franka %s close failed on %r: %s", label, self._tool_name, exc)
            self._panda = None
            self._gripper = None
            self._connected = False
