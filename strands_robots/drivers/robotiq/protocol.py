"""Robotiq 2F-85 register map and Modbus TCP wire format. Pure, no I/O.

The 2F-85 is commanded through three 16-bit registers and reports through
three more. Everything a caller can ask of the gripper - activate, go to a
position, how hard, how fast - is six bytes wide, and everything it reports
back is another six. That map is the whole protocol, and it is shared with the
2F-140 and the Hand-E; only the stroke differs.

Wire frame (Modbus TCP, MBAP header then PDU):

.. code-block:: text

    TRANSACTION  PROTOCOL  LENGTH  UNIT  FUNCTION  ...
    <2 bytes>    0x0000    <2>     <1>   <1>       <function payload>

- ``LENGTH`` counts the bytes that follow it, i.e. ``UNIT`` + the PDU.
- There is **no checksum**. Modbus TCP delegates integrity to TCP; the CRC-16
  in :rfc:`Modbus RTU <0>` framing applies to the serial transport only, which
  is why none is computed here. A caller reaching the same gripper over RS-485
  needs RTU framing around the *same* register payloads
  (:func:`command_registers` / :func:`parse_status`), not a different map.
- A server reporting a problem answers with the function code ORed with
  ``0x80`` and a one-byte exception code, so a refusal is a *parse* outcome
  rather than a timeout. :func:`parse_response` raises
  :class:`ProtocolError` naming the code.

Register layout, from Robotiq's *2F-85 & 2F-140 Instruction Manual*, section
"Modbus RTU/TCP". The robot writes three registers at
:data:`OUTPUT_BASE` and reads three at :data:`INPUT_BASE`; each register is a
big-endian pair of the byte-oriented map:

.. code-block:: text

    write (0x03E8..0x03EA)        read (0x07D0..0x07D2)
    byte 0  ACTION REQUEST        byte 0  GRIPPER STATUS
    byte 1  reserved              byte 1  reserved
    byte 2  reserved              byte 2  FAULT STATUS
    byte 3  rPR position          byte 3  rPR echo
    byte 4  rSP speed             byte 4  gPO actual position
    byte 5  rFR force             byte 5  gCU motor current

Position is a *count*, not a length: ``0`` is fully open and ``255`` fully
closed, so it runs backwards to the aperture a caller measures in millimetres.
:func:`counts_to_aperture_mm` and :func:`aperture_mm_to_counts` are the only
place that inversion is written down, because a sign error here closes a
gripper that was asked to open.

Nothing here opens a socket. The transport lives in
:mod:`strands_robots.drivers.robotiq.driver`, so this module imports on any
host and the codec is gradeable without a gripper.
"""

from __future__ import annotations

import enum
import math
import struct
from typing import Any, Final

# --------------------------------------------------------------------------- #
# Framing and addressing.                                                     #
# --------------------------------------------------------------------------- #

MBAP_SIZE: Final[int] = 7
"""Bytes before the function code: transaction (2), protocol (2), length (2), unit (1)."""

PROTOCOL_ID: Final[int] = 0
"""The only protocol id Modbus TCP defines. A frame carrying anything else is not Modbus."""

DEFAULT_UNIT_ID: Final[int] = 9
"""Slave id Robotiq ships the gripper with (0x09), documented in the manual's
Modbus section. A gripper behind a Universal Robots controller commonly answers
on ``0`` instead, so the driver keeps this a parameter."""

DEFAULT_TCP_PORT: Final[int] = 502
"""The registered Modbus TCP port."""

OUTPUT_BASE: Final[int] = 0x03E8
"""First register the robot writes: the action request."""

INPUT_BASE: Final[int] = 0x07D0
"""First register the robot reads: the gripper status."""

REGISTER_COUNT: Final[int] = 3
"""Registers in each direction. Six bytes, and the map defines no more."""

PAYLOAD_SIZE: Final[int] = REGISTER_COUNT * 2
"""Bytes in one command or one status payload."""

MAX_COUNTS: Final[int] = 0xFF
"""Widest value the position, speed and force fields hold."""

STROKE_MM: Final[float] = 85.0
"""2F-85 stroke: the aperture at position count ``0``. The 2F-140 shares this
codec with a 140 mm stroke, which is why the conversions take it as a default
rather than baking it in."""


class FunctionCode(enum.IntEnum):
    """The two Modbus functions the gripper needs.

    Reading uses input registers (``0x04``) and writing uses the multiple-register
    write (``0x10``), which is what the manual specifies; the single-register
    write is not used because the three output registers must land together for
    ``rGTO`` to act on the position in the same frame.
    """

    READ_INPUT_REGISTERS = 0x04
    WRITE_MULTIPLE_REGISTERS = 0x10


class ActivationStatus(enum.IntEnum):
    """``gSTA``: where the gripper is in its activation sequence.

    There is no ``2``. The manual defines three states and leaves the fourth
    encoding unused, so a gripper reporting ``2`` is not answering this map.
    """

    RESET = 0
    ACTIVATING = 1
    ACTIVE = 3


class ObjectStatus(enum.IntEnum):
    """``gOBJ``: why the fingers stopped, which is how a grasp is detected.

    The distinction that matters to a caller: :attr:`AT_REQUEST` means the
    fingers reached the commanded position and therefore hold *nothing*, while
    :attr:`CONTACT_CLOSING` means they stopped early because something is in
    the way - a successful grasp. Reading "stopped" without reading which kind
    reports every empty close as a pick.
    """

    MOVING = 0
    CONTACT_OPENING = 1
    CONTACT_CLOSING = 2
    AT_REQUEST = 3


class Fault(enum.IntEnum):
    """``gFLT`` values the manual names. Only the codes it documents appear here."""

    NONE = 0x00
    ACTION_DELAYED_REACTIVATION_NEEDED = 0x05
    ACTIVATION_BIT_NOT_SET = 0x07
    OVER_TEMPERATURE = 0x08
    NO_COMMUNICATION = 0x09
    UNDER_VOLTAGE = 0x0A
    AUTOMATIC_RELEASE_IN_PROGRESS = 0x0B
    INTERNAL_FAULT = 0x0C
    ACTIVATION_FAULT = 0x0D
    OVERCURRENT = 0x0E
    AUTOMATIC_RELEASE_COMPLETE = 0x0F


class ProtocolError(ValueError):
    """A frame does not answer this map, or the gripper reported an exception.

    A :class:`ValueError` because a frame is a value that fails to parse. The
    driver catches it to build a refusal envelope, so a malformed reply is
    reported rather than mistaken for a position.
    """


# --------------------------------------------------------------------------- #
# Bounds. Written once: every field in the map is a byte, and a caller handing
# a float or a numpy integer to struct.pack gets an opaque struct.error that
# names neither the field nor this gripper.
# --------------------------------------------------------------------------- #
def _validate_byte(value: Any, field: str) -> int:
    """Return ``value`` as a byte, or raise naming the field and its range.

    Args:
        value: The candidate. A float is refused rather than truncated, because
            ``0.9`` of a position count is a caller error and rounding it hides
            which end of the stroke they meant. A ``bool`` is refused for the
            same reason: ``True`` is not a position.
        field: Field name to quote in the reason.

    Returns:
        The value as an :class:`int`.

    Raises:
        ProtocolError: If ``value`` is not an integer in ``0..255``.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(f"{field} must be an int in 0..{MAX_COUNTS}, got {value!r}")
    if not 0 <= value <= MAX_COUNTS:
        raise ProtocolError(f"{field} must be in 0..{MAX_COUNTS}, got {value}")
    return int(value)


def _validate_finite(value: Any, field: str) -> float:
    """Return ``value`` as a finite float, or raise naming the field.

    Args:
        value: The candidate. A ``bool`` is refused: ``True`` is not an
            aperture, and accepting it as ``1.0`` mm would move the fingers
            almost shut for a caller who meant "open".
        field: Field name to quote in the reason.

    Returns:
        The value as a :class:`float`.

    Raises:
        ProtocolError: If ``value`` is not a finite real number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{field} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ProtocolError(f"{field} must be finite, got {value!r}")
    return float(value)


def command_payload(
    *,
    activate: bool = True,
    go_to: bool = False,
    auto_release: bool = False,
    position: int = 0,
    speed: int = MAX_COUNTS,
    force: int = MAX_COUNTS,
) -> bytes:
    """Build the six output bytes.

    Args:
        activate: ``rACT``. Held ``True`` for every normal command: clearing it
            resets the gripper, which drops the activation and requires the
            whole sequence again.
        go_to: ``rGTO``. The gripper only moves on a frame that sets it, so a
            position written with ``go_to=False`` is staged and not executed.
        auto_release: ``rATR``. Emergency release; the manual requires the
            gripper be reactivated afterwards, so it is not a stop.
        position: ``rPR`` count, ``0`` fully open to ``255`` fully closed.
        speed: ``rSP`` count. Full speed by default.
        force: ``rFR`` count. Full force by default.

    Returns:
        Exactly :data:`PAYLOAD_SIZE` bytes, in the manual's byte order.

    Raises:
        ProtocolError: If a byte-valued field is out of range.
    """
    action = (0x01 if activate else 0x00) | (0x08 if go_to else 0x00) | (0x10 if auto_release else 0x00)
    return struct.pack(
        ">BBBBBB",
        action,
        0,
        0,
        _validate_byte(position, "position"),
        _validate_byte(speed, "speed"),
        _validate_byte(force, "force"),
    )


def command_registers(**kwargs: Any) -> tuple[int, ...]:
    """Build the three register values a write frame carries.

    The registers are the :func:`command_payload` bytes paired big-endian,
    which is the one place that packing is written down.

    Args:
        **kwargs: Passed through to :func:`command_payload`.

    Returns:
        :data:`REGISTER_COUNT` register values.
    """
    payload = command_payload(**kwargs)
    return tuple(int(value) for value in struct.unpack(">HHH", payload))


def parse_status(payload: bytes) -> dict[str, Any]:
    """Decode the six input bytes into named fields.

    Args:
        payload: Exactly :data:`PAYLOAD_SIZE` bytes, as read from
            :data:`INPUT_BASE`.

    Returns:
        The decoded status. ``activation`` and ``object`` are the
        :class:`ActivationStatus` / :class:`ObjectStatus` members; ``fault`` is
        a :class:`Fault` when the code is one the manual names and the raw
        integer otherwise, because an undocumented fault is still worth
        reporting verbatim rather than dropping. ``position`` and
        ``position_request`` are counts; ``aperture_mm`` is the position
        converted for a caller who thinks in millimetres.

    Raises:
        ProtocolError: If ``payload`` is the wrong length, or reports a
            ``gSTA`` outside the three states the map defines.
    """
    if len(payload) != PAYLOAD_SIZE:
        raise ProtocolError(f"status payload must be {PAYLOAD_SIZE} bytes, got {len(payload)}")
    status, _reserved, fault_byte, request, position, current = struct.unpack(">BBBBBB", payload)
    activation_bits = (status >> 4) & 0x03
    try:
        activation = ActivationStatus(activation_bits)
    except ValueError as exc:
        raise ProtocolError(
            f"gSTA={activation_bits} is not one of "
            f"{[member.value for member in ActivationStatus]} - this is not a 2F status word"
        ) from exc
    fault_code = fault_byte & 0x0F
    return {
        "activated": bool(status & 0x01),
        "go_to": bool(status & 0x08),
        "activation": activation,
        "object": ObjectStatus((status >> 6) & 0x03),
        "fault": Fault(fault_code) if fault_code in _FAULT_VALUES else fault_code,
        "position_request": request,
        "position": position,
        "aperture_mm": counts_to_aperture_mm(position),
        "current_ma": current * 10,
    }


_FAULT_VALUES: Final[frozenset[int]] = frozenset(member.value for member in Fault)
"""Codes :class:`Fault` names, so :func:`parse_status` can pass an undocumented
one through as an integer rather than raising on a gripper firmware that grew a
new code."""


# --------------------------------------------------------------------------- #
# Position conversions. The one inversion in the protocol.                     #
# --------------------------------------------------------------------------- #
def counts_to_aperture_mm(counts: int, stroke_mm: float = STROKE_MM) -> float:
    """Convert a position count to the aperture between the fingertips.

    Args:
        counts: ``0`` (fully open) to ``255`` (fully closed).
        stroke_mm: Aperture at count ``0``. Defaults to the 2F-85's.

    Returns:
        The aperture in millimetres.

    Raises:
        ProtocolError: If ``counts`` is not a byte.
    """
    return stroke_mm * (MAX_COUNTS - _validate_byte(counts, "counts")) / MAX_COUNTS


def aperture_mm_to_counts(aperture_mm: float, stroke_mm: float = STROKE_MM) -> int:
    """Convert a fingertip aperture to the position count that commands it.

    The inverse of :func:`counts_to_aperture_mm`, clamped to the stroke: a
    caller asking for 200 mm on an 85 mm gripper means "all the way open", and
    refusing that is less useful than opening.

    Args:
        aperture_mm: Requested aperture in millimetres.
        stroke_mm: Aperture at count ``0``. Defaults to the 2F-85's.

    Returns:
        The position count, ``0..255``.

    Raises:
        ProtocolError: If ``aperture_mm`` is not a finite number, or
            ``stroke_mm`` is not positive.
    """
    _validate_finite(aperture_mm, "aperture_mm")
    if stroke_mm <= 0:
        raise ProtocolError(f"stroke_mm must be positive, got {stroke_mm!r}")
    clamped = min(max(float(aperture_mm), 0.0), stroke_mm)
    return int(round(MAX_COUNTS * (stroke_mm - clamped) / stroke_mm))


def closed_fraction_to_counts(fraction: float) -> int:
    """Convert a normalised close command to a position count.

    Args:
        fraction: ``0.0`` fully open to ``1.0`` fully closed. Clamped, for the
            same reason :func:`aperture_mm_to_counts` clamps.

    Returns:
        The position count, ``0..255``.

    Raises:
        ProtocolError: If ``fraction`` is not a finite number.
    """
    _validate_finite(fraction, "fraction")
    return int(round(MAX_COUNTS * min(max(float(fraction), 0.0), 1.0)))


# --------------------------------------------------------------------------- #
# Modbus TCP framing.                                                          #
# --------------------------------------------------------------------------- #
def _validate_transaction_id(transaction_id: int) -> int:
    if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
        raise ProtocolError(f"transaction_id must be an int, got {transaction_id!r}")
    if not 0 <= transaction_id <= 0xFFFF:
        raise ProtocolError(f"transaction_id must be in 0..65535, got {transaction_id}")
    return transaction_id


def _frame(transaction_id: int, unit_id: int, pdu: bytes) -> bytes:
    """Wrap ``pdu`` in an MBAP header."""
    header = struct.pack(
        ">HHHB",
        _validate_transaction_id(transaction_id),
        PROTOCOL_ID,
        len(pdu) + 1,  # the unit id travels inside the counted length
        _validate_byte(unit_id, "unit_id"),
    )
    return header + pdu


def write_registers_frame(
    transaction_id: int,
    unit_id: int,
    address: int,
    values: tuple[int, ...],
) -> bytes:
    """Build a function-16 frame writing ``values`` at ``address``.

    Args:
        transaction_id: Echoed by the server, so a reply can be matched to its
            request.
        unit_id: Modbus slave id; see :data:`DEFAULT_UNIT_ID`.
        address: First register to write, e.g. :data:`OUTPUT_BASE`.
        values: Register values, each ``0..65535``.

    Returns:
        The complete frame, MBAP header included.

    Raises:
        ProtocolError: If ``values`` is empty or a value is out of range.
    """
    if not values:
        raise ProtocolError("write_registers_frame: values must not be empty")
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFF:
            raise ProtocolError(f"register value must be an int in 0..65535, got {value!r}")
    pdu = struct.pack(
        ">BHHB",
        FunctionCode.WRITE_MULTIPLE_REGISTERS,
        address,
        len(values),
        len(values) * 2,
    ) + struct.pack(f">{len(values)}H", *values)
    return _frame(transaction_id, unit_id, pdu)


def read_input_registers_frame(transaction_id: int, unit_id: int, address: int, count: int) -> bytes:
    """Build a function-4 frame reading ``count`` registers at ``address``.

    Args:
        transaction_id: Echoed by the server.
        unit_id: Modbus slave id.
        address: First register to read, e.g. :data:`INPUT_BASE`.
        count: How many registers to read.

    Returns:
        The complete frame, MBAP header included.

    Raises:
        ProtocolError: If ``count`` is not positive.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ProtocolError(f"count must be a positive int, got {count!r}")
    pdu = struct.pack(">BHH", FunctionCode.READ_INPUT_REGISTERS, address, count)
    return _frame(transaction_id, unit_id, pdu)


def parse_response(raw: bytes, expected_transaction_id: int, expected_function: int) -> bytes:
    """Validate a reply frame and return its PDU body after the function code.

    Args:
        raw: The whole frame, MBAP header included.
        expected_transaction_id: The id the request carried. A mismatch is a
            reply to a *different* request, which on a shared connection would
            otherwise be read as this one's answer.
        expected_function: The function code the request used.

    Returns:
        Everything after the function code. For a read that is the byte count
        followed by the data; for a write, the echoed address and count.

    Raises:
        ProtocolError: If the frame is truncated, carries the wrong protocol
            id or transaction id, or is an exception response.
    """
    if len(raw) < MBAP_SIZE + 1:
        raise ProtocolError(f"response must be at least {MBAP_SIZE + 1} bytes, got {len(raw)}")
    transaction_id, protocol_id, length, _unit_id = struct.unpack(">HHHB", raw[:MBAP_SIZE])
    if protocol_id != PROTOCOL_ID:
        raise ProtocolError(f"protocol id must be {PROTOCOL_ID}, got {protocol_id} - this is not Modbus TCP")
    if transaction_id != expected_transaction_id:
        raise ProtocolError(f"response is for transaction {transaction_id}, expected {expected_transaction_id}")
    body = raw[MBAP_SIZE:]
    if len(body) + 1 != length:
        raise ProtocolError(f"MBAP length says {length} bytes follow, got {len(body) + 1}")
    function = body[0]
    if function == (expected_function | 0x80):
        code = body[1] if len(body) > 1 else 0
        raise ProtocolError(f"gripper refused function {expected_function:#04x} with Modbus exception {code:#04x}")
    if function != expected_function:
        raise ProtocolError(f"response function is {function:#04x}, expected {expected_function:#04x}")
    return body[1:]


def read_registers_payload(raw: bytes, expected_transaction_id: int, count: int) -> bytes:
    """Validate a function-4 reply and return the register bytes it carries.

    Args:
        raw: The whole reply frame.
        expected_transaction_id: The id the request carried.
        count: Registers the request asked for.

    Returns:
        ``count * 2`` data bytes.

    Raises:
        ProtocolError: If the frame is malformed or carries the wrong amount of
            data. A short read is raised rather than padded: half a status word
            decodes to a position the gripper never reported.
    """
    body = parse_response(raw, expected_transaction_id, FunctionCode.READ_INPUT_REGISTERS)
    if not body:
        raise ProtocolError("read response carries no byte count")
    byte_count = body[0]
    data = body[1:]
    if byte_count != count * 2:
        raise ProtocolError(f"read response declares {byte_count} bytes, expected {count * 2}")
    if len(data) != byte_count:
        raise ProtocolError(f"read response declares {byte_count} bytes but carries {len(data)}")
    return data
