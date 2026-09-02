# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Feetech serial bus :class:`FeetechDriver` writes through.

:mod:`~strands_robots.drivers.feetech.protocol` builds and parses the frames -
including the two-byte word order, which is a property of the STS/SMS series and
not of the family, so it is read from there rather than spelled again here. This
module puts those frames on a wire and turns the two byte values a servo speaks
into the units a caller uses. It is the half :issue:`360` scope 1 named as
deferred when the driver landed as a stub.

Units, stated once because a wrong unit is a wrong motion and not an error:
every joint is **degrees**, and ``gripper`` is **percent open** (0 closed,
100 open). That is the domain :mod:`strands_robots.tools.pose_tool` already
established for this arm family, and its ``MotorConfig`` ranges are the
source of truth :data:`SO_ARM_MOTORS` is distilled from - the same six IDs in
the same wire order that lerobot's ``SOFollower`` drives.

A target outside a joint's range is **refused, not clamped**. A clamp turns a
caller's 400-degree command into a 180-degree motion and reports success, so
the caller learns nothing and the arm goes somewhere it was not told to go.

Nothing here imports :mod:`serial` at module load: the import happens in
:meth:`FeetechBus.connect`, so the package stays importable - and its codec
gradeable - on a box with no serial stack at all.
"""

from __future__ import annotations

import logging
import math
import numbers
import time
from typing import Any, Final

from strands_robots.drivers.feetech.protocol import (
    MAX_GOAL_POSITION,
    Instruction,
    ProtocolError,
    Register,
    build_packet,
    decode_word,
    encode_word,
    parse_status_packet,
    read_packet,
    sync_write_packet,
)
from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)


class MotorSpec:
    """One servo on the bus: its ID and the caller-facing range it spans.

    Attributes:
        motor_id: The servo's ID on the shared half-duplex bus.
        low: Lowest caller value, in the joint's own unit.
        high: Highest caller value, in the joint's own unit.
        resolution: Encoder counts spanning ``low``..``high``. Defaults to
            :data:`~strands_robots.drivers.feetech.protocol.MAX_GOAL_POSITION`,
            the STS/SMS full scale - an SCS-series servo spans a quarter of it
            and must be given its own.
    """

    __slots__ = ("high", "low", "motor_id", "resolution")

    def __init__(self, motor_id: int, low: float, high: float, resolution: int = MAX_GOAL_POSITION) -> None:
        self.motor_id = motor_id
        self.low = low
        self.high = high
        self.resolution = resolution

    def to_counts(self, value: float) -> int:
        """Encode a caller value as encoder counts.

        Args:
            value: The target, in this joint's unit.

        Returns:
            Counts in ``0..resolution``.

        Raises:
            ValueError: When ``value`` is outside ``low``..``high``. Refused
                rather than clamped - see the module docstring.
        """
        if not self.low <= value <= self.high:
            raise ValueError(f"target {value} outside range ({self.low}, {self.high})")
        span = self.high - self.low
        return round((value - self.low) / span * self.resolution)

    def to_value(self, counts: int) -> float:
        """Decode encoder counts back into this joint's unit."""
        span = self.high - self.low
        return self.low + counts / self.resolution * span


#: The six servos of an SO-100 / SO-101 follower, in wire order.
#:
#: IDs, ranges and the gripper's percent domain are carried over verbatim from
#: :data:`strands_robots.tools.pose_tool._DEFAULT_MOTOR_CONFIGS`, which is the
#: map proven against the physical arm. Keeping one map for the tool and the
#: driver is what stops the two commanding different joints by the same name.
SO_ARM_MOTORS: Final[dict[str, MotorSpec]] = {
    "shoulder_pan": MotorSpec(1, -180, 180),
    "shoulder_lift": MotorSpec(2, -90, 90),
    "elbow_flex": MotorSpec(3, -150, 150),
    "wrist_flex": MotorSpec(4, -90, 90),
    "wrist_roll": MotorSpec(5, -180, 180),
    "gripper": MotorSpec(6, 0, 100),
}

#: Readable registers, keyed by the name a caller asks for.
#:
#: Each entry is ``(address, sign_bit)``. ``sign_bit`` is ``None`` for a plain
#: unsigned value, otherwise the bit carrying direction - the STS/SMS series
#: encodes these as sign-magnitude, not two's complement, and reading bit 15 as
#: part of the magnitude reports a stopped joint as moving fast (see
#: :mod:`strands_robots.tools.serial_tool`, which pins the same convention for
#: the write direction).
#:
#: ``Present_Current`` (0x45) is deliberately absent: its sign encoding is not
#: established anywhere in this package, and a register decoded by guess
#: reports a number that looks like a measurement. A caller asking for it gets
#: a refusal naming the readable set instead.
READABLE_REGISTERS: Final[dict[str, tuple[int, int | None]]] = {
    "Present_Position": (Register.PRESENT_POSITION, None),
    "Present_Velocity": (Register.PRESENT_VELOCITY, 15),
    "Present_Load": (Register.PRESENT_LOAD, 10),
}

#: Bytes each readable register carries.
_REGISTER_WIDTH: Final[int] = 2

#: Longest reply frame a two-byte read produces (``FF FF ID LEN ERR P0 P1 CHK``)
#: plus slack for bytes the half-duplex bus echoes in front of it, which
#: :func:`~strands_robots.drivers.feetech.protocol.parse_status_packet` skips.
_READ_BUFFER: Final[int] = 10

#: Seconds to let a servo answer before reading. The vendor SDK polls; a fixed
#: settle is enough at 1 Mbaud for a two-byte reply and keeps the read simple.
_REPLY_SETTLE_S: Final[float] = 0.01


def _decode(raw: bytes, sign_bit: int | None) -> int:
    """Turn a two-byte reply into a signed integer.

    The byte order comes from
    :func:`~strands_robots.drivers.feetech.protocol.decode_word`; only the sign
    convention is applied here, because which bit carries direction is a
    per-register property the codec deliberately leaves to its caller.

    Args:
        raw: The register's parameter bytes, in the order the servo sent them.
        sign_bit: Bit carrying direction, or ``None`` when unsigned.

    Returns:
        The register value, negative when ``sign_bit`` is set.
    """
    value = decode_word(raw)
    if sign_bit is None:
        return value
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if (value >> sign_bit) & 1 else magnitude


class FeetechBus:
    """A half-duplex Feetech bus carrying one SO-arm's servos.

    Args:
        port: Serial device path. ``None`` is accepted so a driver can be
            constructed before its port is known; :meth:`connect` refuses.
        baud_rate: Bus speed. 1 Mbaud is the STS3215 default.
        motors: The servos on this bus, defaulting to :data:`SO_ARM_MOTORS`.
        timeout: Serial read timeout in seconds.
    """

    def __init__(
        self,
        port: str | None,
        baud_rate: int = 1_000_000,
        motors: dict[str, MotorSpec] | None = None,
        timeout: float = 1.0,
    ) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.motors = dict(SO_ARM_MOTORS if motors is None else motors)
        self.timeout = timeout
        self._conn: Any | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle.                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        """Whether the port is open, so a caller can tell live from stale."""
        conn = self._conn
        return bool(conn is not None and getattr(conn, "is_open", False))

    def connect(self) -> None:
        """Open the serial port.

        Raises:
            ValueError: When no port was configured.
            OSError: When the port cannot be opened. ``serial.SerialException``
                subclasses ``OSError``, so one except clause covers both.
        """
        if self.is_connected:
            return
        if not self.port:
            raise ValueError("FeetechBus: no port configured; pass port= to open the SCS bus")
        serial = require_optional(
            "serial",
            pip_install="pyserial",
            purpose="the Feetech SCS serial bus",
        )
        self._conn = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)  # type: ignore[attr-defined]

    def disconnect(self) -> None:
        """Close the port. Safe to call when already closed."""
        conn, self._conn = self._conn, None
        if conn is not None and getattr(conn, "is_open", False):
            conn.close()

    def _require_open(self, what: str) -> Any:
        """Return the open connection, or raise naming what was attempted."""
        if not self.is_connected:
            raise RuntimeError(f"FeetechBus: {what} needs an open bus; call connect() first (port={self.port!r})")
        return self._conn

    # ------------------------------------------------------------------ #
    # Reads.                                                              #
    # ------------------------------------------------------------------ #

    def sync_read(self, register: str = "Present_Position", num_retry: int = 0) -> dict[str, float]:
        """Read ``register`` from every motor.

        Named and shaped for :func:`strands_robots.bus_access.read_joints`,
        which calls ``bus.sync_read("Present_Position")`` and appends the
        ``.pos`` suffix itself - so exposing this method is what puts an
        SO-arm's joints on the mesh state topic.

        The servos are read one at a time. The SCS SYNC_READ instruction is not
        available on every servo in this family, and a per-motor READ is what
        the arm is known to answer; a motor whose reply does not verify is
        omitted rather than guessed at, exactly as
        :meth:`strands_robots.tools.pose_tool.MotorController.read_all_positions`
        omits it.

        Args:
            register: A key of :data:`READABLE_REGISTERS`.
            num_retry: Extra attempts per motor before giving up on it.

        Returns:
            Motor name -> value. ``Present_Position`` is in the joint's own
            unit (degrees, or percent for ``gripper``); the other registers are
            raw signed counts. Motors that did not answer are absent.

        Raises:
            ValueError: When ``register`` is not readable.
            RuntimeError: When the bus is not open.
        """
        if register not in READABLE_REGISTERS:
            raise ValueError(
                f"FeetechBus: cannot read {register!r}; readable registers are {sorted(READABLE_REGISTERS)}"
            )
        conn = self._require_open(f"reading {register}")
        address, sign_bit = READABLE_REGISTERS[register]
        out: dict[str, float] = {}
        for name, spec in self.motors.items():
            raw = self._read_one(conn, spec.motor_id, address, num_retry)
            if raw is None:
                logger.warning("no verified %s reply from %s (id %d)", register, name, spec.motor_id)
                continue
            value = _decode(raw, sign_bit)
            out[name] = spec.to_value(value) if register == "Present_Position" else float(value)
        return out

    def _read_one(self, conn: Any, motor_id: int, address: int, num_retry: int) -> bytes | None:
        """Read one register from one motor, or ``None`` if it never verified."""
        request = read_packet(motor_id, address, _REGISTER_WIDTH)
        for _ in range(max(1, num_retry + 1)):
            conn.write(request)
            time.sleep(_REPLY_SETTLE_S)
            reply = conn.read(_READ_BUFFER)
            if not reply:
                continue
            try:
                _error, params = parse_status_packet(reply, motor_id, _REGISTER_WIDTH)
            except ProtocolError:
                # A frame that did not verify is not a measurement. Retry.
                continue
            return params
        return None

    # ------------------------------------------------------------------ #
    # Writes.                                                             #
    # ------------------------------------------------------------------ #

    def write_goal_positions(self, targets: dict[str, float]) -> None:
        """Command joint positions, all in one SYNC_WRITE frame.

        One frame for the whole arm rather than one per joint: the servos then
        latch their goals from the same packet, so a six-joint move starts
        together instead of smearing over six write latencies.

        Args:
            targets: Motor name -> target, in the joint's own unit.

        Raises:
            ValueError: When a name is not on this bus, a value is not finite,
                or a value is outside the joint's range.
            RuntimeError: When the bus is not open.
        """
        if not targets:
            raise ValueError("FeetechBus: no targets to write")
        conn = self._require_open("writing goal positions")
        motor_data: list[tuple[int, bytes]] = []
        for name, value in targets.items():
            spec = self.motors.get(name)
            if spec is None:
                raise ValueError(f"FeetechBus: unknown motor {name!r}; this bus carries {sorted(self.motors)}")
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(
                    f"FeetechBus: {name} target must be a finite number, got {type(value).__name__}: {value!r}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"FeetechBus: {name} target must be finite, got {value!r}")
            counts = spec.to_counts(number)
            motor_data.append((spec.motor_id, encode_word(counts)))
        conn.write(sync_write_packet(Register.GOAL_POSITION, _REGISTER_WIDTH, motor_data))

    def set_torque(self, enabled: bool) -> list[str]:
        """Energize or release every motor, returning the ones that failed.

        Every motor is attempted even after one fails: a release that gave up
        part-way would report the arm safe while some joints are still driven.

        Args:
            enabled: ``True`` to energize, ``False`` to release.

        Returns:
            Names of motors whose write failed; empty when all succeeded. A
            non-empty list after ``enabled=False`` means the arm is NOT fully
            de-energized.

        Raises:
            RuntimeError: When the bus is not open.
        """
        conn = self._require_open("setting torque")
        failed: list[str] = []
        for name, spec in self.motors.items():
            packet = build_packet(
                spec.motor_id,
                Instruction.WRITE,
                bytes([Register.TORQUE_ENABLE, 1 if enabled else 0]),
            )
            try:
                conn.write(packet)
            except OSError as e:
                logger.error("failed to set torque on %s (id %d): %s", name, spec.motor_id, e)
                failed.append(name)
        return failed
