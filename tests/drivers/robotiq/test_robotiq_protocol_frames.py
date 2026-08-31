"""The Robotiq register map and Modbus TCP framing, byte for byte.

The frames are pinned as literal bytes rather than rebuilt from the codec's own
constants: a golden frame derived from the thing under test moves whenever the
thing under test moves, which is the one case a wire-format test exists to
catch. Every literal below is the layout in Robotiq's *2F-85 & 2F-140
Instruction Manual*, Modbus RTU/TCP section.
"""

from __future__ import annotations

import struct

import pytest

from strands_robots.drivers.robotiq.protocol import (
    INPUT_BASE,
    MAX_COUNTS,
    OUTPUT_BASE,
    REGISTER_COUNT,
    STROKE_MM,
    ActivationStatus,
    Fault,
    FunctionCode,
    ObjectStatus,
    ProtocolError,
    aperture_mm_to_counts,
    closed_fraction_to_counts,
    command_payload,
    command_registers,
    counts_to_aperture_mm,
    parse_response,
    parse_status,
    read_input_registers_frame,
    read_registers_payload,
    write_registers_frame,
)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # rACT alone: the activation frame.
        ({"activate": True}, "01 00 00 00 ff ff"),
        # rACT cleared: the reset that must precede a fresh activation.
        ({"activate": False}, "00 00 00 00 ff ff"),
        # rACT|rGTO with a position: the frame that actually moves the fingers.
        ({"activate": True, "go_to": True, "position": 255}, "09 00 00 ff ff ff"),
        ({"activate": True, "go_to": True, "position": 0}, "09 00 00 00 ff ff"),
        # rATR is bit 4, and sits alongside rACT.
        ({"activate": True, "auto_release": True}, "11 00 00 00 ff ff"),
        # Speed and force are the last two bytes, in that order.
        ({"activate": True, "speed": 0x10, "force": 0x20}, "01 00 00 00 10 20"),
    ],
)
def test_the_command_payload_is_the_manual_byte_layout(kwargs: dict[str, object], expected: str) -> None:
    """Each output byte carries the field the manual assigns it."""
    assert command_payload(**kwargs).hex(" ") == expected  # type: ignore[arg-type]


def test_the_command_registers_pair_the_payload_big_endian() -> None:
    """Three registers carry six bytes, high byte first."""
    assert command_registers(activate=True, go_to=True, position=0xFF) == (0x0900, 0x00FF, 0xFFFF)


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (
            write_registers_frame(1, 9, OUTPUT_BASE, command_registers(activate=True, go_to=True, position=255)),
            # transaction 0001 | protocol 0000 | length 000d | unit 09 |
            # fc 10 | addr 03e8 | 3 registers | 6 bytes | payload
            "00 01 00 00 00 0d 09 10 03 e8 00 03 06 09 00 00 ff ff ff",
        ),
        (
            read_input_registers_frame(1, 9, INPUT_BASE, REGISTER_COUNT),
            # transaction 0001 | protocol 0000 | length 0006 | unit 09 |
            # fc 04 | addr 07d0 | 3 registers
            "00 01 00 00 00 06 09 04 07 d0 00 03",
        ),
    ],
)
def test_a_request_frame_is_byte_exact(frame: bytes, expected: str) -> None:
    """The MBAP header and PDU match the wire, including the counted length."""
    assert frame.hex(" ") == expected


def test_the_mbap_length_counts_the_unit_id_and_the_pdu() -> None:
    """The one field a hand-built frame gets wrong, so it is pinned on its own."""
    frame = write_registers_frame(7, 2, OUTPUT_BASE, (1, 2, 3))
    (length,) = struct.unpack(">H", frame[4:6])

    assert length == len(frame) - 6, "length must count every byte after itself"


def test_a_status_word_decodes_the_documented_bit_fields() -> None:
    """``gACT``/``gGTO``/``gSTA``/``gOBJ`` are packed in one byte; read each."""
    # 0xb9 = rACT | rGTO | gSTA=ACTIVE(3)<<4 | gOBJ=CONTACT_CLOSING(2)<<6
    status = parse_status(bytes.fromhex("b9 00 00 c8 be 05"))

    assert status["activated"] is True
    assert status["go_to"] is True
    assert status["activation"] is ActivationStatus.ACTIVE
    assert status["object"] is ObjectStatus.CONTACT_CLOSING
    assert status["fault"] is Fault.NONE
    assert status["position_request"] == 200
    assert status["position"] == 190
    assert status["current_ma"] == 50, "gCU is reported in units of 10 mA"


def test_an_undocumented_fault_is_reported_verbatim_not_dropped() -> None:
    """A firmware that grew a new code should still surface it to an operator."""
    status = parse_status(bytes.fromhex("31 00 03 00 00 00"))

    assert status["fault"] == 0x03
    assert not isinstance(status["fault"], Fault)


def test_a_status_word_that_is_not_a_2f_is_refused() -> None:
    """``gSTA=2`` is undefined, so the payload is not this gripper's."""
    with pytest.raises(ProtocolError, match="not a 2F status word"):
        parse_status(bytes.fromhex("21 00 00 00 00 00"))


@pytest.mark.parametrize("payload", [b"", b"\x00" * 5, b"\x00" * 7])
def test_a_short_status_payload_is_refused_rather_than_padded(payload: bytes) -> None:
    """Half a status word decodes to a position the gripper never reported."""
    with pytest.raises(ProtocolError, match="must be 6 bytes"):
        parse_status(payload)


@pytest.mark.parametrize(
    ("counts", "aperture"),
    [(0, STROKE_MM), (MAX_COUNTS, 0.0), (128, pytest.approx(42.35, abs=0.05))],
)
def test_position_counts_run_backwards_to_aperture(counts: int, aperture: float) -> None:
    """Count 0 is fully OPEN. The inversion is the protocol's one sign trap."""
    assert counts_to_aperture_mm(counts) == aperture


@pytest.mark.parametrize("aperture", [0.0, 10.0, 42.5, 85.0])
def test_the_aperture_conversions_round_trip(aperture: float) -> None:
    """Millimetres to counts and back lands within one count of the stroke."""
    counts = aperture_mm_to_counts(aperture)

    assert counts_to_aperture_mm(counts) == pytest.approx(aperture, abs=STROKE_MM / MAX_COUNTS)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-5.0, 0), (0.0, 0), (1.0, MAX_COUNTS), (99.0, MAX_COUNTS)],
)
def test_a_closed_fraction_outside_the_unit_range_clamps(value: float, expected: int) -> None:
    """ "More than fully closed" means fully closed, not a refusal."""
    assert closed_fraction_to_counts(value) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "half", None, True])
def test_a_position_that_is_not_a_finite_number_is_refused(value: object) -> None:
    """These reach ``struct.pack``, which cannot say which field was wrong."""
    with pytest.raises(ProtocolError):
        aperture_mm_to_counts(value)  # type: ignore[arg-type]
    with pytest.raises(ProtocolError):
        closed_fraction_to_counts(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("position", [-1, 256, 1.5, True, "255"])
def test_a_position_count_outside_a_byte_is_refused(position: object) -> None:
    """A float count hides which end of the stroke the caller meant."""
    with pytest.raises(ProtocolError, match="position"):
        command_payload(position=position)  # type: ignore[arg-type]


def test_an_exception_reply_is_raised_with_its_code() -> None:
    """The gripper answers a refusal with the function ORed with 0x80."""
    exception = struct.pack(">HHHBBB", 1, 0, 3, 9, FunctionCode.READ_INPUT_REGISTERS | 0x80, 0x02)

    with pytest.raises(ProtocolError, match="Modbus exception 0x02"):
        parse_response(exception, 1, FunctionCode.READ_INPUT_REGISTERS)


def test_a_reply_to_a_different_request_is_not_read_as_this_one() -> None:
    """On a shared connection a stale reply would otherwise become the answer."""
    reply = read_input_registers_frame(4, 9, INPUT_BASE, 3)

    with pytest.raises(ProtocolError, match="response is for transaction 4, expected 5"):
        parse_response(reply, 5, FunctionCode.READ_INPUT_REGISTERS)


def test_a_non_modbus_protocol_id_is_refused() -> None:
    """Protocol id 0 is the only one Modbus TCP defines."""
    frame = bytearray(read_input_registers_frame(1, 9, INPUT_BASE, 3))
    frame[2:4] = b"\x00\x07"

    with pytest.raises(ProtocolError, match="not Modbus TCP"):
        parse_response(bytes(frame), 1, FunctionCode.READ_INPUT_REGISTERS)


def test_a_read_reply_declaring_the_wrong_byte_count_is_refused() -> None:
    """A truncated register block must not be decoded as a whole status.

    The MBAP length is kept consistent with the body on purpose, so the frame
    passes the framing check and the byte-count check is the one under test.
    """
    body = struct.pack(">BB", FunctionCode.READ_INPUT_REGISTERS, 2) + b"\x00\x00"
    short = struct.pack(">HHHB", 1, 0, len(body) + 1, 9) + body

    with pytest.raises(ProtocolError, match="declares 2 bytes, expected 6"):
        read_registers_payload(short, 1, REGISTER_COUNT)
