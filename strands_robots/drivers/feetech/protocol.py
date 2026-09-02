"""Feetech STS/SMS-series wire format. Pure, no I/O.

The STS3215 and its SCS/SMS siblings share a Protocol 1-shaped bus that is
close to Dynamixel's older Protocol 1 but not identical to Protocol 2. This
codec is verifiable against ``scservo_sdk`` byte-for-byte (the vendor SDK on
PyPI); as with :mod:`~strands_robots.drivers.dynamixel.protocol`, the point of
extracting it is not dependency avoidance but grading, since the SDK's own
parser lives inside :meth:`PacketHandler.readTxRx` and cannot be exercised
without a serial port.

**Framing is shared across the family; the two-byte word order is not.** Feetech
publishes one framing document for the whole family (`SCS Communication`_ below,
which is where the frame above comes from), and the vendor SDK is one package for
all of it - but that SDK carries a global end-ness its ``PacketHandler`` sets
from a per-model protocol number, and the two orders are the reverse of each
other:

.. code-block:: text

    protocol 0  STS3215, STS3250, SM8512BL   low byte first
    protocol 1  SCS0009 and the SCS series   high byte first

So a two-byte register value framed for one series is a *different value* on the
other, not a mis-scaled one: 1023 written low-byte-first is read as 65283 by an
SCS-series servo. This module implements protocol 0 - :func:`encode_word` and
:func:`decode_word` are that order, named so the whole package reads it from one
place, and :data:`MAX_GOAL_POSITION` is the STS/SMS full scale (the SCS series is
10-bit, a quarter of it). Supporting an SCS-series servo is therefore a second
word order and a second full scale rather than a parameter, and nothing here
claims to do it; see :issue:`2812`.

Wire frame (from the Feetech STS3215 datasheet, `SCS Communication`_):

.. code-block:: text

    HEADER1  HEADER2  ID    LEN     INSTR   P1..PN   CHECKSUM
    0xFF     0xFF     <id>  <n+2>   <op>    <n bytes> <sum>

- ``ID`` is 0..253 for a specific servo, ``0xFE`` for a broadcast (no reply).
- ``LEN`` counts ``INSTR`` + params + checksum, i.e. ``n + 2``.
- ``CHECKSUM = (~(ID + LEN + INSTR + sum(params))) & 0xFF``.
  This is the *additive* checksum Feetech uses. Protocol 2's CRC-16 does not
  apply here. :mod:`~strands_robots.tools.pose_tool` already uses this
  formula inline; the codec here reuses the *same* bit pattern so a frame
  built by :func:`build_packet` is byte-identical to one the pose tool would
  emit, and :func:`parse_status_packet` accepts what
  :func:`~strands_robots.tools.pose_tool._parse_status_packet` accepts.
- No byte-stuffing. Protocol 2 stuffs to break the ``FF FF FD`` prefix out of
  payloads; Feetech's Protocol 1 tolerates ``FF`` inside a param because the
  parser resyncs on the ``FF FF`` header pair and a following byte cannot be
  ``FF`` (LEN is at most 0xFC on any real read, and the ID byte is 0..0xFE).

.. _SCS Communication: https://www.feetechrc.com/PDF/EN%20SCS%20Communication%20Protocol.pdf

The status packet a servo returns is the same shape with an ERR byte in front
of the parameters (``FF FF ID LEN ERR P1..PN CHECKSUM``), so
:func:`parse_status_packet` shares the framing check with :func:`build_packet`
and only differs in what it consumes after ``INSTR``.

Nothing here opens a serial port. The one function that would want the port
- ``read_register`` say - is deliberately in the bus module (see :issue:`360`
scope 1) so this module can be imported on any host in CI.

Register addresses are declared in :class:`Register`. The STS3215 EEPROM/SRAM
map spans 0x00..0x53; only the registers the driver actually reads or writes
are named here. Adding a register is a one-line entry with a comment naming
the datasheet page it comes from.
"""

from __future__ import annotations

import enum
from typing import Final, Literal

# ---------------------------------------------------------------------------
# Framing constants. Two bytes named because the manual talks about them by
# name, and because a caller reading the source should not have to remember
# whether Feetech's header is one 0xFF or two.
# ---------------------------------------------------------------------------
HEADER: Final[bytes] = b"\xff\xff"
"""The two-byte start-of-frame every Feetech packet begins with."""

BROADCAST_ID: Final[int] = 0xFE
"""ID a controller writes to when every servo on the bus should receive the
packet and none should reply. :func:`sync_write_packet` uses this; a
broadcast that expects replies (``SYNC_READ``, ``BULK_READ``) needs a
distinct primitive that reads N status packets back."""

MAX_UNICAST_ID: Final[int] = 0xFD
"""Highest ID a specific servo may hold. 0xFE is the broadcast."""

_MAX_PARAM_COUNT: Final[int] = 0xFA
"""``LEN`` is one byte, and it must carry ``params + 2``. Anything above
``0xFA`` (250) params would overflow ``LEN`` past ``0xFC`` and collide with
the reserved range; the SDK caps writes below this and so does this codec."""

MAX_GOAL_POSITION: Final[int] = 4095
"""Highest index ``Goal_Position`` addresses on the STS/SMS series, and so the
divisor that turns a count into a fraction of a full turn.

12-bit, i.e. 4096 counts. This is a property of the *series*, not of the
register: lerobot's ``MODEL_RESOLUTION`` gives 4096 counts for ``sts3215``,
``sts3250`` and ``sm8512bl`` and 1024 for ``scs0009``, so on an SCS-series servo
the ceiling and the divisor are both 1023. Every surface in this package that
bounds or reports a Feetech position reads this name, so the series the number
belongs to is stated once."""

WORD_LENGTH: Final[int] = 2
"""Bytes a two-byte register field occupies on the wire."""

_MAX_WORD: Final[int] = (1 << (8 * WORD_LENGTH)) - 1

_WORD_ORDER: Final[Literal["little"]] = "little"
"""The STS/SMS (protocol 0) byte order, named rather than spelled as shifts so
the one place it is decided reads as the property it is. The SCS series is
``"big"``; nothing here emits that order."""


def encode_word(value: int) -> bytes:
    """Encode ``value`` as the two bytes an STS/SMS-series servo reads.

    Low byte first - the vendor SDK's protocol 0 order, which its
    ``SCS_LOBYTE`` / ``SCS_HIBYTE`` pair produces while its global end-ness is
    0. The SCS series sets that global to 1 and reads the same two bytes in the
    opposite order, so this encoder is series-specific by construction and is
    not portable to it (see the module docstring).

    Args:
        value: The register value, ``0..0xFFFF``. A field narrower than the word
            - ``Goal_Position`` at :data:`MAX_GOAL_POSITION`, ``Goal_Velocity``
            at its direction bit - is bounded by the caller that owns the field's
            meaning; this refuses only what the two bytes cannot carry at all.

    Returns:
        Exactly :data:`WORD_LENGTH` bytes, low byte first.

    Raises:
        TypeError: If ``value`` is not an :class:`int`. A :class:`bool` is
            refused with it, since ``True`` would silently encode as 1.
        ValueError: If ``value`` does not fit two bytes. Masking it instead
            would put a different, reachable command on the wire and report the
            number the caller asked for.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"word value must be int, got {type(value).__name__}")
    if not 0 <= value <= _MAX_WORD:
        raise ValueError(f"word value out of range 0..{_MAX_WORD}: {value}")
    return value.to_bytes(WORD_LENGTH, _WORD_ORDER)


def decode_word(raw: bytes) -> int:
    """Read the two bytes an STS/SMS-series servo replied with.

    The inverse of :func:`encode_word`, and the same protocol 0 order. Sign
    handling is deliberately absent: which bit carries direction is a
    per-register property (bit 15 on the goal and present pairs, bit 10 on
    ``Present_Load``, and nothing at all on the SCS series), so it belongs to
    the caller that knows which register it read.

    Args:
        raw: Exactly :data:`WORD_LENGTH` bytes, low byte first.

    Returns:
        The unsigned register value.

    Raises:
        TypeError: If ``raw`` is not bytes-like.
        ValueError: If ``raw`` is not exactly :data:`WORD_LENGTH` bytes - a
            short read decoded anyway reports a position the servo never sent.
    """
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError(f"raw must be bytes, got {type(raw).__name__}")
    if len(raw) != WORD_LENGTH:
        raise ValueError(f"a register word is {WORD_LENGTH} bytes, got {len(raw)}")
    return int.from_bytes(raw, _WORD_ORDER)


class Instruction(enum.IntEnum):
    """The instruction bytes Feetech's Protocol 1 defines.

    Only the members the driver actually issues appear here; ``FACTORY_RESET``
    and friends will land as they are wired.
    """

    PING = 0x01
    READ = 0x02
    WRITE = 0x03
    REG_WRITE = 0x04
    ACTION = 0x05
    RESET = 0x06
    SYNC_WRITE = 0x83


class Register(enum.IntEnum):
    """STS3215 register addresses the driver reads or writes.

    Addresses match the STS3215 datasheet's control table (rev. 2024-02, pp.
    12-15). Registers below 0x28 are EEPROM (persist across power); 0x28 and
    up are SRAM (volatile). The driver does not write EEPROM unless the
    caller opts into it, so both regions are named here but the write-path
    validators refuse EEPROM addresses by default (bus PR scope, not codec).
    """

    # EEPROM
    ID = 0x05
    BAUD_RATE = 0x06
    RETURN_DELAY = 0x07
    STATUS_RETURN_LEVEL = 0x08
    MIN_POSITION_LIMIT = 0x09  # 2 bytes
    MAX_POSITION_LIMIT = 0x0B  # 2 bytes
    MAX_TORQUE = 0x10  # 2 bytes

    # SRAM
    TORQUE_ENABLE = 0x28
    ACCELERATION = 0x29
    GOAL_POSITION = 0x2A  # 2 bytes, 0..MAX_GOAL_POSITION on the STS/SMS series
    GOAL_TIME = 0x2C  # 2 bytes
    GOAL_VELOCITY = 0x2E  # 2 bytes, sign-magnitude on bit 15
    LOCK = 0x37
    PRESENT_POSITION = 0x38  # 2 bytes, read-only
    PRESENT_VELOCITY = 0x3A  # 2 bytes, sign-magnitude on bit 15
    PRESENT_LOAD = 0x3C  # 2 bytes, sign-magnitude on bit 10
    PRESENT_VOLTAGE = 0x3E
    PRESENT_TEMPERATURE = 0x3F
    MOVING = 0x42
    PRESENT_CURRENT = 0x45  # 2 bytes


# ---------------------------------------------------------------------------
# Building. Every builder produces a byte string that a servo bus would put on
# the wire verbatim - no framing, checksum, or field is inferred.
# ---------------------------------------------------------------------------


def _validate_id(motor_id: int, allow_broadcast: bool) -> None:
    """Refuse an ID the wire cannot carry, or a broadcast where a reply is expected."""
    if not isinstance(motor_id, int) or isinstance(motor_id, bool):
        raise TypeError(f"motor_id must be int, got {type(motor_id).__name__}")
    if not 0 <= motor_id <= 0xFE:
        raise ValueError(f"motor_id out of range 0..0xFE: {motor_id:#x}")
    if motor_id == BROADCAST_ID and not allow_broadcast:
        raise ValueError(f"motor_id={BROADCAST_ID:#x} is the broadcast; this instruction expects a reply")


def _validate_params(params: bytes) -> None:
    """Refuse a parameter block that will not fit in a one-byte LEN."""
    if len(params) > _MAX_PARAM_COUNT:
        raise ValueError(f"params length {len(params)} exceeds LEN capacity ({_MAX_PARAM_COUNT})")


def _checksum(payload: bytes) -> int:
    """Feetech's additive checksum: bitwise-NOT of the low byte of the running sum.

    ``payload`` is the frame's ``ID + LEN + INSTR + params`` block - everything
    the header does not cover and the checksum does. This is the identity
    :mod:`~strands_robots.tools.pose_tool` uses (``~sum(packet[2:]) & 0xFF``);
    stated here as a function so :func:`parse_status_packet` grades the same
    bit pattern against inbound frames.
    """
    return (~sum(payload)) & 0xFF


def build_packet(motor_id: int, instruction: int, params: bytes = b"", *, allow_broadcast: bool = False) -> bytes:
    """Frame one Feetech Protocol 1 packet.

    Args:
        motor_id: Target servo ID (0..0xFD unicast, 0xFE broadcast).
        instruction: One of :class:`Instruction` values.
        params: Parameter bytes for the instruction, in the order the register
            map declares.
        allow_broadcast: Set to ``True`` for :attr:`Instruction.SYNC_WRITE` and
            other reply-less instructions that legitimately target the
            broadcast ID. Every other caller passes ``False`` so a slip that
            addresses a broadcast where a reply is required is refused here
            rather than by a servo that never answers.

    Returns:
        The complete frame ``FF FF ID LEN INSTR params CHECKSUM``.

    Raises:
        TypeError: If ``motor_id`` is not an :class:`int`. A :class:`bool` is
            refused with it, since ``True`` would silently address motor 1.
        ValueError: If ``motor_id``, ``instruction`` or the length of
            ``params`` is outside the range the frame can carry, or if
            ``motor_id`` is the broadcast while ``allow_broadcast`` is
            ``False``. The type and range refusals come from
            :func:`_validate_id`, so a caller sees them from here.
    """
    _validate_id(motor_id, allow_broadcast=allow_broadcast)
    _validate_params(params)
    if not 0 <= instruction <= 0xFF:
        raise ValueError(f"instruction byte out of range: {instruction:#x}")

    length = len(params) + 2  # INSTR + params + CHECKSUM
    payload = bytes([motor_id, length, instruction]) + params
    return HEADER + payload + bytes([_checksum(payload)])


def ping_packet(motor_id: int) -> bytes:
    """Frame a ``PING`` to ``motor_id``.

    The servo replies with an empty status packet whose error byte reports
    its state; that reply is what the bus module reads back.

    Raises:
        TypeError: If ``motor_id`` is not an :class:`int`.
        ValueError: If ``motor_id`` is outside ``0x00..0xFE``, or is the
            broadcast - a ``PING`` is answered, so it is never reply-less.
    """
    return build_packet(motor_id, Instruction.PING)


def read_packet(motor_id: int, address: int, length: int) -> bytes:
    """Frame a ``READ`` of ``length`` bytes starting at ``address``.

    ``address`` and ``length`` are one byte each. The reply is an
    :func:`parse_status_packet`-shaped frame carrying ``length`` param bytes.

    Raises:
        TypeError: If ``motor_id`` is not an :class:`int`.
        ValueError: If ``motor_id``, ``address`` or ``length`` is outside its
            one-byte range, or if ``motor_id`` is the broadcast - a ``READ``
            is answered, so it is never reply-less.
    """
    if not 0 <= address <= 0xFF:
        raise ValueError(f"address out of range 0..0xFF: {address:#x}")
    if not 1 <= length <= 0xFA:
        raise ValueError(f"read length out of range 1..0xFA: {length}")
    return build_packet(motor_id, Instruction.READ, bytes([address, length]))


def write_packet(motor_id: int, address: int, data: bytes, *, allow_broadcast: bool = False) -> bytes:
    """Frame a ``WRITE`` of ``data`` to ``address`` on ``motor_id``.

    The servo replies with an empty status packet (unless ``motor_id`` is the
    broadcast). ``allow_broadcast`` lets a caller who genuinely wants a
    reply-less write opt in - the default refuses a broadcast because a
    caller who addresses one by mistake would otherwise silently wait for a
    reply that never comes.

    Raises:
        TypeError: If ``motor_id`` is not an :class:`int`.
        ValueError: If ``motor_id`` or ``address`` is out of range, if
            ``data`` is empty, or if ``motor_id`` is the broadcast while
            ``allow_broadcast`` is ``False``.
    """
    if not 0 <= address <= 0xFF:
        raise ValueError(f"address out of range 0..0xFF: {address:#x}")
    if not data:
        raise ValueError("write data is empty; nothing to send")
    return build_packet(motor_id, Instruction.WRITE, bytes([address]) + data, allow_broadcast=allow_broadcast)


def sync_write_packet(address: int, per_motor_length: int, motor_data: list[tuple[int, bytes]]) -> bytes:
    """Frame a ``SYNC_WRITE`` addressing the broadcast ID.

    ``motor_data`` is ``(motor_id, data)`` pairs; each ``data`` block must be
    exactly ``per_motor_length`` bytes so the servo's parser can carve the
    packet into per-servo slices without a per-servo length field.

    Raises:
        TypeError: If any ``motor_id`` in ``motor_data`` is not an
            :class:`int`.
        ValueError: If ``address`` or ``per_motor_length`` is out of range, if
            ``motor_data`` is empty, if it names one ``motor_id`` twice, if a
            ``data`` block is not exactly ``per_motor_length`` bytes, or if a
            ``motor_id`` inside ``motor_data`` is the broadcast.
    """
    if not 0 <= address <= 0xFF:
        raise ValueError(f"address out of range 0..0xFF: {address:#x}")
    if not 1 <= per_motor_length <= 0xFA:
        raise ValueError(f"per_motor_length out of range 1..0xFA: {per_motor_length}")
    if not motor_data:
        raise ValueError("sync_write with no motors is a no-op; refused")

    params = bytearray([address, per_motor_length])
    seen: set[int] = set()
    for motor_id, data in motor_data:
        _validate_id(motor_id, allow_broadcast=False)  # a broadcast INSIDE sync_write is a bug
        if motor_id in seen:
            raise ValueError(f"sync_write lists motor_id {motor_id:#x} twice")
        if len(data) != per_motor_length:
            raise ValueError(f"motor {motor_id:#x} data length {len(data)} != per_motor_length {per_motor_length}")
        seen.add(motor_id)
        params.append(motor_id)
        params += data

    return build_packet(BROADCAST_ID, Instruction.SYNC_WRITE, bytes(params), allow_broadcast=True)


# ---------------------------------------------------------------------------
# Parsing. :func:`parse_status_packet` accepts one framed status packet and
# returns the error byte and param bytes it carries, or raises with a message
# naming which field failed which check.
# ---------------------------------------------------------------------------


class ProtocolError(ValueError):
    """A byte string is not a valid Feetech status packet.

    Distinguished from :class:`ValueError` so callers - the bus module - can
    tell a wire-corruption from a caller bug at the codec boundary.
    """


def parse_status_packet(raw: bytes, expected_id: int, expected_param_count: int) -> tuple[int, bytes]:
    """Read one status packet's error byte and param block.

    Args:
        raw: The exact bytes read back from the bus. Leading noise up to the
            first ``FF FF`` header is skipped (a half-duplex bus can carry an
            echoed byte from the host's own write); trailing bytes are
            refused because they may belong to the next packet and the bus
            module - not this codec - is responsible for framing the stream.
        expected_id: The ID the packet must carry. A mismatch is
            :class:`ProtocolError` because a bus with multiple motors may
            reply out of order (the pose tool's status framing test names
            the same shape).
        expected_param_count: How many param bytes the READ or PING asked
            for. The frame's LEN is checked against ``expected_param_count +
            2`` so a servo that answers with the wrong param count is
            refused here rather than by the caller reading past the buffer.

    Returns:
        ``(error_byte, params)`` where ``params`` is exactly
        ``expected_param_count`` bytes long.

    Raises:
        TypeError: If ``raw`` is not bytes-like, refused before any framing is
            attempted.
        ValueError: If ``expected_id`` or ``expected_param_count`` is outside
            the domain the protocol allows. Named separately from
            :class:`ProtocolError` even though that is itself a
            :class:`ValueError`, because these two mark a caller bug rather
            than wire corruption - the distinction the bus module reads this
            boundary for, and one ``except ProtocolError`` alone would not
            catch them.
        ProtocolError: If ``raw`` is not one well-formed status packet
            addressed to ``expected_id`` carrying ``expected_param_count``
            param bytes.
    """
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError(f"raw must be bytes, got {type(raw).__name__}")
    if not 0 <= expected_id <= 0xFD:
        raise ValueError(f"expected_id out of range 0..0xFD: {expected_id:#x}")
    if expected_param_count < 0 or expected_param_count > _MAX_PARAM_COUNT:
        raise ValueError(f"expected_param_count out of range: {expected_param_count}")

    # Resync on the header. A half-duplex bus can put the host's own echo
    # (a single 0xFF) in front of the reply; :func:`_parse_status_packet` in
    # pose_tool.py handles that by scanning, and the codec does the same so
    # the two agree on what a valid frame looks like.
    start = _find_header(raw)
    if start < 0:
        raise ProtocolError(f"no FF FF header found in {len(raw)} bytes")

    frame = bytes(raw[start:])
    # Minimum: header(2) + id + len + err + checksum = 6 bytes for zero params.
    min_len = 6 + expected_param_count
    if len(frame) < min_len:
        raise ProtocolError(f"status packet truncated: got {len(frame)} bytes after header, need {min_len}")

    # LEN is checked before trailing-bytes so a servo that answered with the
    # wrong param count is refused as a protocol violation rather than as a
    # buffer-size mismatch. The two shapes shade into each other on the wire
    # but the caller wants to know which one the servo caused.
    motor_id = frame[2]
    length = frame[3]
    expected_length = expected_param_count + 2  # ERR + params + CHECKSUM
    if length != expected_length:
        raise ProtocolError(f"status packet LEN mismatch: got {length}, expected {expected_length}")

    if len(frame) > min_len:
        raise ProtocolError(f"status packet has trailing bytes: got {len(frame)}, exactly {min_len} expected")

    error = frame[4]
    params = frame[5 : 5 + expected_param_count]
    got_checksum = frame[5 + expected_param_count]

    if motor_id != expected_id:
        raise ProtocolError(f"status packet ID mismatch: got {motor_id:#x}, expected {expected_id:#x}")

    payload = frame[2 : 5 + expected_param_count]  # ID..last-param
    want_checksum = _checksum(payload)
    if got_checksum != want_checksum:
        raise ProtocolError(f"status packet checksum mismatch: got {got_checksum:#x}, computed {want_checksum:#x}")

    return error, params


def _find_header(raw: bytes) -> int:
    """Index of the first ``FF FF`` in ``raw`` that starts a real frame, or ``-1``.

    A half-duplex bus can put the host's own echoed 0xFF in front of the
    reply, so a naive search for ``FF FF`` in ``\\xff\\xff\\xff...`` lands on
    offset 0 and reads the echo byte as ID. Skip past the run of leading
    ``0xFF`` bytes until exactly two remain in front of a non-``0xFF``: that
    pair is the frame header.

    Two 0xFF bytes together anywhere else in a Feetech frame are impossible
    (LEN <= 0xFC, error and params are bounded), so the first such pair whose
    third byte is not ``0xFF`` is the frame start.
    """
    i = 0
    n = len(raw)
    while i < n - 1:
        if raw[i] != 0xFF:
            i += 1
            continue
        # Found a run of 0xFFs starting at i. Advance to the last two.
        j = i
        while j < n and raw[j] == 0xFF:
            j += 1
        # The frame's header is the two 0xFF right before the non-0xFF byte.
        if j - i >= 2 and j < n:
            return j - 2
        # Otherwise the run reaches the end without a body; nothing to parse.
        return -1
    return -1
