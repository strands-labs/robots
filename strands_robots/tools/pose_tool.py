#!/usr/bin/env python3
"""
Robot Pose Management Tool

This tool provides comprehensive pose management for robotic arms, including:
- Storing and retrieving named poses
- Fine-grained motor control with small incremental movements
- Safety checks and validation
- Integration with LeRobot and serial communication
- Pose interpolation and smooth transitions
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import serial
import serial.tools.list_ports
from strands import tool

from strands_robots.tools._path_validation import resolve_output_path, validate_save_path
from strands_robots.utils import finite_number_error, positive_count_error, positive_finite_number_error

logger = logging.getLogger(__name__)


# Interpolation options: which actions read them, and the domain they must be in.
#
# ``steps`` and ``step_delay`` are only consumed on the interpolated path, so an
# action that moves in one shot never reads them and must never be refused for
# them. ``reset_to_home`` interpolates unconditionally; ``load_pose`` and
# ``move_multiple`` interpolate only when the caller leaves ``smooth`` truthy.
_INTERPOLATING_ACTIONS = frozenset({"load_pose", "move_multiple"})
_ALWAYS_INTERPOLATING_ACTIONS = frozenset({"reset_to_home"})


def _interpolates(action: str, smooth: bool) -> bool:
    """Report whether ``action`` will build an interpolated trajectory.

    Args:
        action: The requested action.
        smooth: The caller's ``smooth`` flag, which only some actions consult.

    Returns:
        True when the action reads ``steps`` / ``step_delay``.
    """
    if action in _ALWAYS_INTERPOLATING_ACTIONS:
        return True
    return action in _INTERPOLATING_ACTIONS and bool(smooth)


def _smooth_move_option_error(action: str, *, smooth: bool, steps: Any, step_delay: Any) -> str | None:
    """Error text for an interpolation option this action cannot honor.

    Both options are consumed inside the interpolation loop, on a live servo
    bus: ``steps`` divides the travel into increments and bounds the loop, and
    ``step_delay`` is the pause between successive goal positions. Their product
    is the trajectory's duration, so neither has a usable value outside its
    domain and each fails in its own way if one is not applied.

    ``steps`` is checked against
    :func:`~strands_robots.utils.positive_count_error`: it is a divisor and a
    ``range()`` bound, so ``0`` raises ``ZeroDivisionError``, a negative count
    makes the loop body unreachable and reports a move that never happened, and
    a fractional or string count raises ``TypeError`` from ``range()``. ``bool``
    is refused with it, because ``True`` is a silent single increment - a
    full-travel jump on exactly the path a caller asking to interpolate wanted
    to avoid.

    ``step_delay`` is checked against
    :func:`~strands_robots.utils.positive_finite_number_error` - the same domain
    the library's other pacing knobs use - and that includes refusing ``0``. The
    pause *is* the smoothing: with no pause the increments are written as fast as
    the bus accepts them (a six-motor 20-step move is ~126 short writes, on the
    order of ten milliseconds at 1 Mbaud) instead of over the requested
    ``steps * step_delay`` seconds, so the interpolation the caller asked for
    cannot be honored. A caller who wants to go straight to the target already
    has ``smooth=False`` for it. A negative or ``nan`` delay raises
    ``ValueError`` from ``time.sleep`` and ``inf`` blocks the call forever,
    leaving the arm stopped part-way through its trajectory with the port still
    open.

    Args:
        action: The requested action; decides whether the options are read.
        smooth: The caller's ``smooth`` flag.
        steps: Interpolation step count, as supplied.
        step_delay: Seconds between increments, as supplied.

    Returns:
        An error message naming the action and the option, or ``None`` when the
        action reads neither option or both values are usable.
    """
    if not _interpolates(action, bool(smooth)):
        return None
    if error := positive_count_error(steps, "steps", action):
        return error
    return positive_finite_number_error(step_delay, "step_delay", action)


@dataclass
class RobotPose:
    """Represents a robot pose with metadata."""

    name: str
    positions: dict[str, float]  # motor_name -> position
    timestamp: float
    description: str | None = None
    safety_bounds: dict[str, tuple[float, float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotPose":
        """Create from dictionary."""
        return cls(**data)


class PoseManager:
    """Manages robot poses with persistence and safety."""

    def __init__(self, robot_id: str, storage_dir: Path | None = None):
        self.robot_id = robot_id
        raw_dir = str(storage_dir) if storage_dir else str(Path.cwd() / ".strands_robots" / "poses")
        self.storage_dir = Path(validate_save_path(raw_dir, label="storage_dir"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.pose_file = Path(resolve_output_path(str(self.storage_dir), f"{robot_id}_poses.json", label="robot_id"))
        self.poses: dict[str, RobotPose] = {}
        self._load_poses()

    def _load_poses(self) -> None:
        """Load poses from storage."""
        if self.pose_file.exists():
            try:
                with open(self.pose_file) as f:
                    data = json.load(f)
                    self.poses = {name: RobotPose.from_dict(pose_data) for name, pose_data in data.items()}
                logger.info(f"Loaded {len(self.poses)} poses for robot {self.robot_id}")
            except Exception as e:
                logger.error(f"Failed to load poses: {e}")
                self.poses = {}

    def _save_poses(self) -> None:
        """Save poses to storage."""
        try:
            data = {name: pose.to_dict() for name, pose in self.poses.items()}
            with open(self.pose_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.poses)} poses for robot {self.robot_id}")
        except Exception as e:
            logger.error(f"Failed to save poses: {e}")

    def store_pose(
        self,
        name: str,
        positions: dict[str, float],
        description: str | None = None,
        safety_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> RobotPose:
        """Store a new pose."""
        pose = RobotPose(
            name=name,
            positions=positions.copy(),
            timestamp=time.time(),
            description=description,
            safety_bounds=safety_bounds,
        )
        self.poses[name] = pose
        self._save_poses()
        return pose

    def get_pose(self, name: str) -> RobotPose | None:
        """Get a stored pose."""
        return self.poses.get(name)

    def list_poses(self) -> list[str]:
        """List all pose names."""
        return list(self.poses.keys())

    def delete_pose(self, name: str) -> bool:
        """Delete a pose."""
        if name in self.poses:
            del self.poses[name]
            self._save_poses()
            return True
        return False

    def validate_pose(self, pose: RobotPose) -> tuple[bool, str]:
        """Validate pose is within safety bounds."""
        if not pose.safety_bounds:
            return True, "No safety bounds defined"

        for motor, position in pose.positions.items():
            if motor in pose.safety_bounds:
                min_pos, max_pos = pose.safety_bounds[motor]
                if not (min_pos <= position <= max_pos):
                    return False, f"Motor {motor} position {position} outside bounds [{min_pos}, {max_pos}]"

        return True, "Pose is valid"


class MotorConfig(TypedDict):
    """Configuration for a single servo motor."""

    id: int
    range: tuple[int, int]
    resolution: int


# Default motor configurations for SO-101.
#
# Module-level rather than built inside ``MotorController.__init__`` because the
# ``range`` of each joint is consulted twice: by
# :meth:`MotorController.degrees_to_position`, which converts a target into a
# ``Goal_Position``, and by :func:`_joint_target_error`, which refuses a target
# that conversion could not represent. A second copy of these bounds could
# disagree with the one the servo is actually driven from, which is the failure
# the guard exists to prevent.
_DEFAULT_MOTOR_CONFIGS: dict[str, MotorConfig] = {
    "shoulder_pan": {"id": 1, "range": (-180, 180), "resolution": 4095},
    "shoulder_lift": {"id": 2, "range": (-90, 90), "resolution": 4095},
    "elbow_flex": {"id": 3, "range": (-150, 150), "resolution": 4095},
    "wrist_flex": {"id": 4, "range": (-90, 90), "resolution": 4095},
    "wrist_roll": {"id": 5, "range": (-180, 180), "resolution": 4095},
    "gripper": {"id": 6, "range": (0, 100), "resolution": 4095},
}

# The degree-valued target each action reads. An action absent from this map
# commands no joint and is never refused here.
_TARGET_OPTION_BY_ACTION: dict[str, str] = {
    "move_motor": "position",
    "move_multiple": "positions",
    "incremental_move": "delta",
}


def _target_unit(motor_name: str) -> str:
    """The unit a target for ``motor_name`` is quoted in.

    Args:
        motor_name: The motor the target is for.

    Returns:
        ``"percent"`` for the gripper, which is configured 0-100, else
        ``"degrees"``.
    """
    return "percent" if motor_name == "gripper" else "degrees"


def _joint_target_error(action: str, label: str, motor_name: str | None, value: Any) -> str | None:
    """Error text when ``value`` is not a target ``motor_name`` can be driven to.

    ``degrees_to_position`` clamps its argument into the joint's configured
    ``range`` before scaling it onto the 12-bit ``Goal_Position`` register, so
    every value outside that range shares one encoding: the mechanical limit.
    That makes the clamp a silent rewrite rather than a safety net -- the arm
    travels to the end stop, and the caller is told it moved to the value it
    asked for, because the success text echoes the request. ``nan`` lands there
    too, since ``min(max_deg, nan)`` returns ``max_deg``.

    Finiteness, numeric-ness and ``bool`` are delegated to
    :func:`~strands_robots.utils.finite_number_error` so an off-domain target is
    reported in the words every other surface uses; only the per-joint bounds
    are decided here, because they are a property of the arm this module drives.

    A motor absent from :data:`_DEFAULT_MOTOR_CONFIGS` has no bounds to check
    against and is left to the action's own unknown-motor path.

    Args:
        action: The requested action, used as the message prefix.
        label: How the target is named in the message.
        motor_name: The motor the target is for.
        value: The caller-supplied target.

    Returns:
        An error message, or ``None`` when the target can be honored.
    """
    if error := finite_number_error(value, label, action):
        return error
    name = motor_name or ""
    config = _DEFAULT_MOTOR_CONFIGS.get(name)
    if config is None:
        return None
    low, high = config["range"]
    if not low <= value <= high:
        return (
            f"{action}: {label} must be within [{low}, {high}] {_target_unit(name)} "
            f"(the configured travel of '{name}'), got {value}."
        )
    return None


def _joint_delta_error(action: str, motor_name: str | None, delta: Any) -> str | None:
    """Error text when ``delta`` is not a displacement ``motor_name`` can travel.

    Bounded by the joint's *full travel* rather than by its endpoints, which is
    the one way this domain differs from :func:`_joint_target_error`: the
    endpoints are absolute and a displacement is not, so the value cannot be
    compared against them without knowing where the joint currently is. A
    magnitude larger than the whole range is unhonorable from every starting
    position, which is checkable without that reading.

    The resulting absolute target is left to ``degrees_to_position``, whose clamp
    remains the last resort for a target computed from a live position reading
    rather than supplied by the caller.

    A motor absent from :data:`_DEFAULT_MOTOR_CONFIGS` has no travel to bound a
    displacement against, so this domain defers exactly as
    :func:`_joint_target_error` does. What it defers to is the action itself:
    ``incremental_move`` needs a current position before it can compute anything,
    and neither ``read_motor_position`` nor ``move_motor`` can address a motor
    absent from that table, so the move is refused before any ``Goal_Position``
    is written.

    Args:
        action: The requested action, used as the message prefix.
        motor_name: The motor the displacement is for.
        delta: The caller-supplied displacement.

    Returns:
        An error message, or ``None`` when the displacement can be honored.
    """
    if error := finite_number_error(delta, "delta", action):
        return error
    name = motor_name or ""
    config = _DEFAULT_MOTOR_CONFIGS.get(name)
    if config is None:
        return None
    low, high = config["range"]
    span = high - low
    if abs(delta) > span:
        return (
            f"{action}: delta must be at most {span} {_target_unit(name)} in magnitude "
            f"(the full travel of '{name}', so no starting position could honor more), got {delta}."
        )
    return None


def _pose_target_error(
    action: str,
    *,
    motor_name: str | None,
    position: Any,
    delta: Any,
    positions: Any,
) -> str | None:
    """Error text for a degree-valued target ``action`` reads but cannot honor.

    Only the target the requested action consumes is checked, so a caller is
    never refused for a value the action never looks at. A target left unset
    stays the concern of the action's own "required" check, which reports the
    whole missing pair at once.

    Args:
        action: The requested action.
        motor_name: The motor ``position`` / ``delta`` are for.
        position: The absolute target for ``move_motor``.
        delta: The displacement for ``incremental_move``.
        positions: The per-motor targets for ``move_multiple``.

    Returns:
        The first error message, or ``None`` when every target read is usable.
    """
    param = _TARGET_OPTION_BY_ACTION.get(action)
    if param is None:
        return None

    if param == "positions":
        # A non-mapping (or empty) ``positions`` is reported by the action's own
        # required check, which names the parameter rather than one motor.
        if not isinstance(positions, dict):
            return None
        for name, value in positions.items():
            if error := _joint_target_error(action, f"positions[{name!r}]", name, value):
                return error
        return None

    if param == "position":
        if position is None:
            return None
        return _joint_target_error(action, "position", motor_name, position)

    if delta is None:
        return None
    return _joint_delta_error(action, motor_name, delta)


class MotorController:
    """Low-level motor control for fine movements."""

    def __init__(self, port: str, baudrate: int = 1000000):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: serial.Serial | None = None

        self.motor_configs: dict[str, MotorConfig] = {
            name: config.copy() for name, config in _DEFAULT_MOTOR_CONFIGS.items()
        }

    def connect(self) -> tuple[bool, str]:
        """Connect to robot.

        Returns:
            Tuple[bool, str]: (success, error_message) - error_message is empty on success
        """
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1.0)
            return True, ""
        except Exception as e:
            error_msg = f"Failed to connect to {self.port}: {e}"
            logger.error(error_msg)
            return False, error_msg

    def disconnect(self) -> None:
        """Disconnect from robot."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

    def build_feetech_packet(self, motor_id: int, instruction: int, params: list[int]) -> bytes:
        """Build Feetech servo protocol packet."""
        packet = [0xFF, 0xFF, motor_id, len(params) + 2, instruction] + params
        checksum = ~sum(packet[2:]) & 0xFF
        packet.append(checksum)
        return bytes(packet)

    def degrees_to_position(self, motor_name: str, degrees: float) -> int:
        """Convert degrees to motor position."""
        if motor_name not in self.motor_configs:
            raise ValueError(f"Unknown motor: {motor_name}")

        config = self.motor_configs[motor_name]
        min_deg, max_deg = config["range"]

        # Clamp to range
        degrees = max(min_deg, min(max_deg, degrees))

        # Convert to position (0-4095 for most servos)
        if motor_name == "gripper":
            # Gripper uses 0-100 percentage
            return int((degrees / 100.0) * config["resolution"])
        else:
            # Regular joints use degree range
            normalized = (degrees - min_deg) / (max_deg - min_deg)
            return int(normalized * config["resolution"])

    def position_to_degrees(self, motor_name: str, position: int) -> float:
        """Convert motor position to degrees."""
        if motor_name not in self.motor_configs:
            raise ValueError(f"Unknown motor: {motor_name}")

        config = self.motor_configs[motor_name]
        min_deg, max_deg = config["range"]

        if motor_name == "gripper":
            return (position / config["resolution"]) * 100.0
        else:
            normalized = position / config["resolution"]
            return min_deg + normalized * (max_deg - min_deg)

    def move_motor(self, motor_name: str, position_degrees: float) -> bool:
        """Move a single motor to position in degrees."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False

        try:
            motor_id = self.motor_configs[motor_name]["id"]
            position = self.degrees_to_position(motor_name, position_degrees)

            # Feetech position command: INST_WRITE (0x03), Goal_Position address (0x2A)
            params = [0x2A, position & 0xFF, (position >> 8) & 0xFF]
            packet = self.build_feetech_packet(motor_id, 0x03, params)
            self.serial_conn.write(packet)
            return True
        except Exception as e:
            logger.error(f"Failed to move motor {motor_name}: {e}")
            return False

    def disable_torque(self) -> list[str]:
        """De-energize every configured motor, returning the ones that failed.

        Writes ``Torque_Enable = 0`` to each motor. That register is address 40
        (1 byte) on the Feetech STS/SMS control table -- the authority is
        ``lerobot.motors.feetech.tables``, the same table that gives
        ``Goal_Position`` address 42 used by :meth:`move_motor`.

        Every motor is attempted even after one fails: a stop that gave up on
        the remaining joints would be worse than no stop at all, because the
        caller would be told the whole arm was released when part of it is
        still driven.

        Returns:
            Names of the motors whose write failed, empty when all succeeded. A
            non-empty list means the arm is NOT fully de-energized.
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            return list(self.motor_configs)

        failed: list[str] = []
        for motor_name, config in self.motor_configs.items():
            try:
                # INST_WRITE (0x03), Torque_Enable address (0x28), value 0.
                packet = self.build_feetech_packet(config["id"], 0x03, [0x28, 0x00])
                self.serial_conn.write(packet)
            except OSError as e:
                # Narrow to the transport: ``serial.SerialException`` subclasses
                # ``OSError``. Record and continue so the remaining motors are
                # still attempted, then report which ones are still driven.
                logger.error(f"Failed to disable torque on {motor_name}: {e}")
                failed.append(motor_name)
        return failed

    def read_motor_position(self, motor_name: str) -> float | None:
        """Read current motor position in degrees."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return None

        try:
            motor_id = self.motor_configs[motor_name]["id"]

            # Feetech read command: INST_READ (0x02), Present_Position address (0x38), 2 bytes
            params = [0x38, 0x02]
            packet = self.build_feetech_packet(motor_id, 0x02, params)
            self.serial_conn.write(packet)

            time.sleep(0.01)  # Small delay for response
            response = self.serial_conn.read(10)

            if len(response) >= 7:
                position = response[5] | (response[6] << 8)
                return self.position_to_degrees(motor_name, position)
        except Exception as e:
            logger.error(f"Failed to read motor {motor_name}: {e}")

        return None

    def read_all_positions(self) -> dict[str, float]:
        """Read all motor positions."""
        positions = {}
        for motor_name in self.motor_configs:
            pos = self.read_motor_position(motor_name)
            if pos is not None:
                positions[motor_name] = pos
        return positions

    def move_multiple_motors(
        self,
        positions: dict[str, float],
        smooth: bool = True,
        steps: int = 20,
        step_delay: float = 0.05,
    ) -> bool:
        """Move multiple motors simultaneously.

        Args:
            positions: Target position per motor name.
            smooth: Interpolate towards the targets instead of commanding them
                in one shot.
            steps: Number of increments when interpolating. Forwarded to
                :meth:`_smooth_move`, which requires a positive integer.
            step_delay: Seconds between increments when interpolating.
                Forwarded to :meth:`_smooth_move`, which requires a positive
                finite value.

        Returns:
            True when every motor was commanded successfully.
        """
        if smooth:
            return self._smooth_move(positions, steps=steps, step_delay=step_delay)
        else:
            success = True
            for motor_name, position in positions.items():
                if not self.move_motor(motor_name, position):
                    success = False
            return success

    def _smooth_move(self, target_positions: dict[str, float], steps: int = 20, step_delay: float = 0.05) -> bool:
        """Smoothly move to target positions.

        Args:
            target_positions: Target position per motor name.
            steps: Number of increments; must be a positive integer, since it is
                the divisor for each motor's per-step increment and the bound of
                the write loop.
            step_delay: Seconds between increments; must be positive and finite,
                since it is passed straight to ``time.sleep``.
                :func:`_smooth_move_option_error` is what holds both to those
                domains for every caller that reaches here through
                :func:`pose_tool`.

        Returns:
            True once every increment has been written.
        """
        current_positions = self.read_all_positions()

        # Calculate step increments
        step_increments = {}
        for motor, target in target_positions.items():
            if motor in current_positions:
                current = current_positions[motor]
                step_increments[motor] = (target - current) / steps

        # Execute smooth movement
        for step in range(steps + 1):
            for motor, target in target_positions.items():
                if motor in current_positions and motor in step_increments:
                    current = current_positions[motor]
                    new_position = current + (step_increments[motor] * step)
                    self.move_motor(motor, new_position)

            time.sleep(step_delay)

        return True

    def incremental_move(self, motor_name: str, delta_degrees: float) -> bool:
        """Move motor by a small increment."""
        current_pos = self.read_motor_position(motor_name)
        if current_pos is None:
            return False

        new_pos = current_pos + delta_degrees
        return self.move_motor(motor_name, new_pos)


@tool
def pose_tool(
    action: str,
    robot_id: str = "so101_follower",
    port: str | None = "/dev/ttyACM0",
    pose_name: str | None = None,
    motor_name: str | None = None,
    position: float | None = None,
    delta: float | None = None,
    positions: dict[str, float] | None = None,
    description: str | None = None,
    smooth: bool = True,
    steps: int = 20,
    step_delay: float = 0.05,
) -> dict[str, Any]:
    """
    Advanced robot pose management tool with fine motor control.

    Actions:
        Pose Management:
        - "store_pose": Store current robot pose with a name
        - "load_pose": Move robot to a stored pose
        - "list_poses": List all stored poses
        - "delete_pose": Delete a stored pose
        - "show_pose": Display pose information

        Motor Control:
        - "move_motor": Move single motor to position
        - "move_multiple": Move multiple motors simultaneously
        - "incremental_move": Small incremental motor movement
        - "read_position": Read current motor position
        - "read_all": Read all motor positions

        System:
        - "connect": Test robot connection
        - "emergency_stop": De-energize every motor (Torque_Enable=0). The arm
          goes LIMP and falls under gravity -- it does not hold position, so
          anything it is grasping is dropped. Reports an error when any motor
          could not be released.
        - "reset_to_home": Move to safe home position

    Calibration:
        This tool performs no calibration - every action above drives or reads a
        motor through the calibration already on disk. Stored calibrations are
        managed by the separate lerobot_calibrate tool (list, view, backup,
        restore), and the interactive prompt LeRobot shows when a device has
        none is answered by a lerobot_teleoperate session's
        ``auto_accept_calibration``.

    Args:
        action: Action to perform
        robot_id: Robot identifier for pose storage. Becomes part of the pose
            file's name, so a value resolving outside the storage directory is
            refused rather than written there.
        port: Serial port for robot communication
        pose_name: Name for pose operations
        motor_name: Motor name for single motor operations
        position: Target position in degrees (or 0-100% for gripper). A finite
            number within the motor's configured travel - a value outside it is
            refused rather than clamped to the mechanical limit, because the
            clamp cannot be told apart from a typo and the success text echoes
            the value asked for.
        delta: Incremental movement in degrees. A finite number whose magnitude
            is at most the motor's full travel, which no starting position could
            exceed.
        positions: Dictionary of motor positions {motor_name: degrees}. Every
            value is held to the same domain as ``position``, and the first that
            is not names the motor it came from.
        description: Description for stored poses
        smooth: Use smooth interpolated movement
        steps: Number of increments for an interpolated move. A positive
            integer - it divides the travel and bounds the write loop.
        step_delay: Seconds between increments of an interpolated move. A
            positive finite number - this pause is what makes the move smooth,
            so ``0`` is refused; use ``smooth=False`` to go straight to the
            target. Together with ``steps`` it sets the trajectory duration
            (the default 20 x 0.05s = ~1s).

    Both interpolation options are read only by ``load_pose`` and
    ``move_multiple`` (when ``smooth`` is left truthy) and by
    ``reset_to_home``, which always interpolates; any other action ignores them
    and is never refused for them.

    Returns:
        Dict containing status and response content, or an error dict when an
        interpolation option or a joint target the requested action reads cannot
        be honored.
    """

    # Both interpolation options are consumed on a live servo bus - one as a
    # divisor and loop bound, one as the pause between goal positions - so an
    # unusable value is refused here, before any pose file is read or the port
    # is opened, rather than raising part-way through a trajectory.
    if option_error := _smooth_move_option_error(action, smooth=smooth, steps=steps, step_delay=step_delay):
        return {"status": "error", "content": [{"text": option_error}]}

    # Every degree-valued target is scaled onto ``Goal_Position`` by
    # ``degrees_to_position``, which clamps into the joint's configured range -
    # so a target outside it is not refused but silently rewritten to the
    # mechanical limit, while the success text echoes the value asked for. It is
    # refused here, before the port is opened, so the arm never travels to an
    # end stop on a request that could not be honored.
    if target_error := _pose_target_error(
        action, motor_name=motor_name, position=position, delta=delta, positions=positions
    ):
        return {"status": "error", "content": [{"text": target_error}]}

    # Initialize managers. PoseManager composes its pose file name from
    # robot_id, so an id that resolves outside the storage directory is refused
    # there rather than here. This construction sits outside the try below, so
    # the refusal is caught explicitly - otherwise it would leave this tool as
    # an exception instead of the error envelope every other refusal here uses.
    try:
        pose_manager = PoseManager(robot_id)
    except ValueError as e:
        return {"status": "error", "content": [{"text": str(e)}]}

    try:
        if action == "list_poses":
            poses = pose_manager.list_poses()
            if not poses:
                return {
                    "status": "success",
                    "content": [
                        {"text": f"No poses stored for robot {robot_id}"},
                        {"json": {"poses": []}},
                    ],
                }

            # Get detailed pose information
            pose_details = []
            for name in poses:
                pose = pose_manager.get_pose(name)
                if pose is None:
                    continue
                pose_details.append(
                    {
                        "name": name,
                        "description": pose.description or "No description",
                        "timestamp": time.ctime(pose.timestamp),
                        "motors": len(pose.positions),
                    }
                )

            pose_list = "\n".join(
                [f"- {p['name']} - {p['description']} ({p['motors']} motors) - {p['timestamp']}" for p in pose_details]
            )

            return {
                "status": "success",
                "content": [
                    {"text": f"Stored poses for {robot_id}:\n{pose_list}"},
                    {"json": {"poses": pose_details}},
                ],
            }

        if action == "show_pose":
            if not pose_name:
                return {"status": "error", "content": [{"text": "pose_name required"}]}

            pose = pose_manager.get_pose(pose_name)
            if not pose:
                return {"status": "error", "content": [{"text": f"Pose '{pose_name}' not found"}]}

            motor_info = "\n".join([f"  - {motor}: {pos:.2f} deg" for motor, pos in pose.positions.items()])

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Pose: {pose.name}\n"
                        f"Description: {pose.description or 'None'}\n"
                        f"Created: {time.ctime(pose.timestamp)}\n"
                        f"Motor Positions:\n{motor_info}"
                    },
                    {"json": {"pose": pose.to_dict()}},
                ],
            }

        if action == "delete_pose":
            if not pose_name:
                return {"status": "error", "content": [{"text": "pose_name required"}]}

            if pose_manager.delete_pose(pose_name):
                return {"status": "success", "content": [{"text": f"Deleted pose '{pose_name}'"}]}
            else:
                return {"status": "error", "content": [{"text": f"Pose '{pose_name}' not found"}]}

        # Actions that need motor controller
        if not port:
            return {"status": "error", "content": [{"text": "port required for motor operations"}]}

        controller = MotorController(port)

        if action == "connect":
            connected, error = controller.connect()
            if connected:
                controller.disconnect()
                return {"status": "success", "content": [{"text": f"Successfully connected to robot on {port}"}]}
            else:
                return {"status": "error", "content": [{"text": f"{error}"}]}

        if action == "read_position":
            if not motor_name:
                return {"status": "error", "content": [{"text": "motor_name required"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                position = controller.read_motor_position(motor_name)
                if position is not None:
                    unit = "%" if motor_name == "gripper" else " deg"
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"{motor_name}: {position:.2f}{unit}"},
                            {"json": {"position": position}},
                        ],
                    }
                else:
                    return {"status": "error", "content": [{"text": f"Failed to read {motor_name}"}]}
            finally:
                controller.disconnect()

        if action == "read_all":
            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                positions = controller.read_all_positions()
                if positions:
                    pos_text = "\n".join(
                        [
                            f"  - {motor}: {pos:.2f}{'%' if motor == 'gripper' else ' deg'}"
                            for motor, pos in positions.items()
                        ]
                    )
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"Current robot positions:\n{pos_text}"},
                            {"json": {"positions": positions}},
                        ],
                    }
                else:
                    return {"status": "error", "content": [{"text": "Failed to read positions"}]}
            finally:
                controller.disconnect()

        if action == "store_pose":
            if not pose_name:
                return {"status": "error", "content": [{"text": "pose_name required"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                current_positions = controller.read_all_positions()
                if not current_positions:
                    return {"status": "error", "content": [{"text": "Failed to read current positions"}]}

                pose = pose_manager.store_pose(pose_name, current_positions, description)

                pos_text = "\n".join(
                    [
                        f"  - {motor}: {pos:.2f}{'%' if motor == 'gripper' else ' deg'}"
                        for motor, pos in current_positions.items()
                    ]
                )

                return {
                    "status": "success",
                    "content": [
                        {"text": f"Stored pose '{pose_name}':\n{pos_text}"},
                        {"json": {"pose": pose.to_dict()}},
                    ],
                }
            finally:
                controller.disconnect()

        if action == "load_pose":
            if not pose_name:
                return {"status": "error", "content": [{"text": "pose_name required"}]}

            pose = pose_manager.get_pose(pose_name)
            if not pose:
                return {"status": "error", "content": [{"text": f"Pose '{pose_name}' not found"}]}

            # Validate pose
            is_valid, msg = pose_manager.validate_pose(pose)
            if not is_valid:
                return {"status": "error", "content": [{"text": f"Pose validation failed: {msg}"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                success = controller.move_multiple_motors(pose.positions, smooth, steps=steps, step_delay=step_delay)
                if success:
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"Moved to pose '{pose_name}'"},
                            {"json": {"target_positions": pose.positions}},
                        ],
                    }
                else:
                    return {"status": "error", "content": [{"text": f"Failed to move to pose '{pose_name}'"}]}
            finally:
                controller.disconnect()

        if action == "move_motor":
            if not motor_name or position is None:
                return {"status": "error", "content": [{"text": "motor_name and position required"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                success = controller.move_motor(motor_name, position)
                if success:
                    unit = "%" if motor_name == "gripper" else " deg"
                    return {"status": "success", "content": [{"text": f"Moved {motor_name} to {position}{unit}"}]}
                else:
                    return {"status": "error", "content": [{"text": f"Failed to move {motor_name}"}]}
            finally:
                controller.disconnect()

        if action == "move_multiple":
            if not positions:
                return {"status": "error", "content": [{"text": "positions dict required"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                success = controller.move_multiple_motors(positions, smooth, steps=steps, step_delay=step_delay)
                if success:
                    pos_text = "\n".join(
                        [
                            f"  - {motor}: {pos:.2f}{'%' if motor == 'gripper' else ' deg'}"
                            for motor, pos in positions.items()
                        ]
                    )
                    return {"status": "success", "content": [{"text": f"Moved multiple motors:\n{pos_text}"}]}
                else:
                    return {"status": "error", "content": [{"text": "Failed to move motors"}]}
            finally:
                controller.disconnect()

        if action == "incremental_move":
            if not motor_name or delta is None:
                return {"status": "error", "content": [{"text": "motor_name and delta required"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                success = controller.incremental_move(motor_name, delta)
                if success:
                    unit = "%" if motor_name == "gripper" else " deg"
                    sign = "+" if delta >= 0 else ""
                    return {"status": "success", "content": [{"text": f"Moved {motor_name} by {sign}{delta}{unit}"}]}
                else:
                    return {"status": "error", "content": [{"text": f"Failed to move {motor_name}"}]}
            finally:
                controller.disconnect()

        if action == "reset_to_home":
            # Define safe home position
            home_positions = {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            }

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                success = controller.move_multiple_motors(
                    home_positions, smooth=True, steps=steps, step_delay=step_delay
                )
                if success:
                    return {
                        "status": "success",
                        "content": [
                            {"text": "Robot moved to home position"},
                            {"json": {"home_positions": home_positions}},
                        ],
                    }
                else:
                    return {"status": "error", "content": [{"text": "Failed to move to home position"}]}
            finally:
                controller.disconnect()

        if action == "emergency_stop":
            # This handler used to return "Emergency stop executed (torque
            # disabled)" while executing no code at all. A fabricated
            # confirmation is the worst possible failure on a safety path: an
            # operator or agent reading success believes a moving arm has been
            # released, and the one action they would reach for in an emergency
            # is the one that never did anything.
            connected, connect_error = controller.connect()
            if not connected:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"EMERGENCY STOP FAILED - the arm was NOT de-energized: {connect_error}. "
                                "Use the hardware power cutoff."
                            )
                        }
                    ],
                }
            try:
                failed = controller.disable_torque()
            finally:
                controller.disconnect()

            if failed:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "EMERGENCY STOP INCOMPLETE - torque is still enabled on: "
                                f"{', '.join(failed)}. Those joints are still driven; use the "
                                "hardware power cutoff."
                            )
                        }
                    ],
                }
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "Emergency stop executed - torque disabled on "
                            f"{len(controller.motor_configs)} motors. The arm is limp and will "
                            "fall under gravity; anything held has been dropped."
                        )
                    }
                ],
            }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}\n"
                        "Available actions: store_pose, load_pose, list_poses, delete_pose, show_pose, "
                        "move_motor, move_multiple, incremental_move, read_position, read_all, "
                        "connect, reset_to_home, emergency_stop"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"Pose tool error: {e}")
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}
