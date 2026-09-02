# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every frame :mod:`~strands_robots.drivers.feetech.protocol` produces or
consumes must be byte-identical to what the vendor SDK - ``scservo_sdk`` on
PyPI - would put on the wire, and to what a real Feetech STS3215 would
reply.

Two audiences:

- **Any box, SDK or not.** 45 of this file's 47 cells build frames by hand
  from the datasheet-published shape and check them against the codec
  directly. No SDK, no serial, no host dependency. Upstream CI happens to
  carry ``scservo_sdk``, but only incidentally - it arrives transitively
  through ``lerobot[feetech]``, which the ``all`` extra the hatch default
  env installs pulls in - so these cells deliberately do not rely on it.
- **A host with the SDK**, where :class:`TestTheVendorAgreesOnFraming` runs
  the same builders through :class:`scservo_sdk.PacketHandler` and confirms
  the byte sequences match. That is the remaining 2 cells; they skip where
  the SDK is absent.

The bus / driver skeleton lands as a stacked PR (see :issue:`360` scope 2);
this suite grades only the codec.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from strands_robots.drivers.feetech.protocol import (
    BROADCAST_ID,
    HEADER,
    MAX_GOAL_POSITION,
    MAX_UNICAST_ID,
    WORD_LENGTH,
    Instruction,
    ProtocolError,
    build_packet,
    decode_word,
    encode_word,
    parse_status_packet,
    ping_packet,
    read_packet,
    sync_write_packet,
    write_packet,
)


# ---------------------------------------------------------------------------
# The additive-checksum identity, isolated from framing.
# ---------------------------------------------------------------------------
def _feetech_checksum(payload: bytes) -> int:
    """The formula the datasheet (rev. 2024-02, p. 8) publishes."""
    return (~sum(payload)) & 0xFF


class TestChecksum:
    """Grade the codec's checksum against the formula published by Feetech."""

    def test_ping_frame_carries_the_published_checksum(self) -> None:
        # From the STS3215 datasheet's PING example: ID=1, LEN=2, INSTR=0x01,
        # sum(1+2+1)=4, checksum = ~4 & 0xFF = 0xFB.
        frame = ping_packet(1)
        assert frame == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])

    def test_write_frame_matches_hand_computed_checksum(self) -> None:
        # WRITE 2 bytes 0x00 0x08 to address 0x2A on motor 5.
        # Payload after header: 05 05 03 2A 00 08
        # Sum = 5+5+3+42+0+8 = 63 -> ~63 & 0xFF = 0xC0.
        frame = write_packet(5, 0x2A, bytes([0x00, 0x08]))
        assert frame == bytes([0xFF, 0xFF, 0x05, 0x05, 0x03, 0x2A, 0x00, 0x08, 0xC0])

    def test_read_frame_matches_hand_computed_checksum(self) -> None:
        # READ 2 bytes from address 0x38 on motor 3.
        # Payload: 03 04 02 38 02 ; sum=3+4+2+56+2=67 -> ~67 & 0xFF = 0xBC.
        frame = read_packet(3, 0x38, 2)
        assert frame == bytes([0xFF, 0xFF, 0x03, 0x04, 0x02, 0x38, 0x02, 0xBC])


class TestFrameShape:
    """The header, ID, LEN, and instruction bytes go where the datasheet
    puts them - nothing here validates payload content."""

    def test_header_is_two_ff_bytes(self) -> None:
        assert HEADER == b"\xff\xff"
        assert ping_packet(1).startswith(HEADER)

    def test_len_field_counts_instr_params_checksum(self) -> None:
        # WRITE with 1 address byte + 3 data bytes = 4 params.
        # LEN = params(4) + INSTR + CHECKSUM = 6.
        frame = write_packet(1, 0x2A, bytes([0x11, 0x22, 0x33]))
        assert frame[3] == 6

    def test_len_field_for_ping_is_two(self) -> None:
        # PING has no params: LEN = 0 (params) + 2 (INSTR + CHECKSUM) = 2.
        assert ping_packet(1)[3] == 2

    def test_instruction_byte_is_the_enum_value(self) -> None:
        # READ carries Instruction.READ = 0x02 in the instruction slot.
        assert read_packet(1, 0x38, 2)[4] == Instruction.READ.value

    def test_broadcast_write_is_refused_by_default(self) -> None:
        with pytest.raises(ValueError, match="broadcast"):
            write_packet(BROADCAST_ID, 0x28, bytes([1]))

    def test_broadcast_write_admits_explicit_opt_in(self) -> None:
        # A caller who genuinely wants a reply-less write may opt in - the
        # default refuses only the mistake of addressing a broadcast where a
        # reply is expected.
        frame = write_packet(BROADCAST_ID, 0x28, bytes([1]), allow_broadcast=True)
        assert frame[2] == BROADCAST_ID


class TestIdDomain:
    """The wire lands an ID byte between 0 and 0xFE inclusive. Anything else
    is refused before it can reach the bus."""

    @pytest.mark.parametrize("motor_id", [0, 1, 0x40, MAX_UNICAST_ID, BROADCAST_ID])
    def test_accepted_ids(self, motor_id: int) -> None:
        # Broadcast is accepted for sync_write's opt-in shape; the ID-only
        # domain check here does not distinguish.
        build_packet(motor_id, Instruction.PING, allow_broadcast=True)

    @pytest.mark.parametrize("motor_id", [-1, 0xFF, 0x100, 1000])
    def test_out_of_range_ids_refused(self, motor_id: int) -> None:
        with pytest.raises(ValueError, match="motor_id"):
            build_packet(motor_id, Instruction.PING, allow_broadcast=True)

    @pytest.mark.parametrize("motor_id", [1.0, True, "1", None])
    def test_non_integer_ids_refused(self, motor_id) -> None:
        # ``bool`` is a subclass of int; refuse it too because a motor
        # addressed as ``True`` is a caller bug and the wire cannot tell.
        with pytest.raises(TypeError, match="motor_id"):
            build_packet(motor_id, Instruction.PING, allow_broadcast=True)


class TestReadPacketDomain:
    """READ addresses one byte and asks for at most 250 bytes back."""

    def test_length_zero_refused(self) -> None:
        with pytest.raises(ValueError, match="read length"):
            read_packet(1, 0x38, 0)

    def test_length_above_cap_refused(self) -> None:
        with pytest.raises(ValueError, match="read length"):
            read_packet(1, 0x38, 0xFB)

    @pytest.mark.parametrize("address", [-1, 0x100])
    def test_out_of_range_address_refused(self, address: int) -> None:
        with pytest.raises(ValueError, match="address"):
            read_packet(1, address, 2)


class TestWritePacketDomain:
    """WRITE refuses an empty payload; there is no way to say 'write nothing'
    on this bus, and the resulting frame would ping-shape instead of
    write-shape."""

    def test_empty_payload_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            write_packet(1, 0x28, b"")

    def test_over_cap_payload_refused(self) -> None:
        # LEN is one byte, so the params block + address byte + INSTR +
        # CHECKSUM has to fit under 0xFC. 250 params - 1 address byte = 249,
        # so a 250-byte payload overflows.
        with pytest.raises(ValueError, match="LEN capacity"):
            write_packet(1, 0x28, bytes(250))


class TestSyncWrite:
    """SYNC_WRITE addresses the broadcast and gives every listed servo the
    same block size, so the parser can carve the payload without a
    per-servo length."""

    def test_single_motor_frame_shape(self) -> None:
        # Two-byte goal position at 0x2A for motor 1, value 2048.
        frame = sync_write_packet(0x2A, 2, [(1, bytes([0x00, 0x08]))])
        # Header + BC ID + LEN + INSTR + addr + per_len + [id + 2 bytes] + CHK
        assert frame[:2] == HEADER
        assert frame[2] == BROADCAST_ID
        assert frame[4] == Instruction.SYNC_WRITE.value
        assert frame[5] == 0x2A  # address
        assert frame[6] == 2  # per-motor length
        assert frame[7] == 1  # first motor's ID
        assert frame[8:10] == bytes([0x00, 0x08])

    def test_multiple_motors_frame_shape(self) -> None:
        frame = sync_write_packet(0x2A, 2, [(1, b"\x11\x22"), (2, b"\x33\x44"), (3, b"\x55\x66")])
        # Body starts after HEADER + BC + LEN + INSTR = 6 bytes.
        body = frame[5:-1]
        assert body[0] == 0x2A
        assert body[1] == 2
        # Per-motor slices in the same order the caller listed.
        assert body[2:5] == bytes([1, 0x11, 0x22])
        assert body[5:8] == bytes([2, 0x33, 0x44])
        assert body[8:11] == bytes([3, 0x55, 0x66])

    def test_mismatched_per_motor_length_refused(self) -> None:
        with pytest.raises(ValueError, match="data length"):
            sync_write_packet(0x2A, 2, [(1, b"\x11\x22"), (2, b"\x33")])

    def test_duplicate_motor_id_refused(self) -> None:
        # A sync-write that lists the same motor twice is either a caller
        # bug (the loop wrote the same slot into two dict entries) or a race
        # (two producers merged). Either way the servo takes only one of
        # them, silently; refuse it here.
        with pytest.raises(ValueError, match="twice"):
            sync_write_packet(0x2A, 2, [(1, b"\x11\x22"), (1, b"\x33\x44")])

    def test_empty_motor_list_refused(self) -> None:
        with pytest.raises(ValueError, match="no motors"):
            sync_write_packet(0x2A, 2, [])

    def test_broadcast_id_inside_motor_list_refused(self) -> None:
        # A caller who listed the broadcast ID as one of the servos is
        # confused - sync_write is *already* addressed to the broadcast.
        with pytest.raises(ValueError, match="broadcast"):
            sync_write_packet(0x2A, 2, [(BROADCAST_ID, b"\x00\x00")])


class TestParseStatusPacket:
    """Parse a well-formed status packet, and refuse each corruption class."""

    def _make_status(self, motor_id: int, params: bytes, error: int = 0, *, corrupt_checksum: bool = False) -> bytes:
        """Build one status packet the way a servo would."""
        length = len(params) + 2  # ERR + params + CHECKSUM
        payload = bytes([motor_id, length, error]) + params
        checksum = _feetech_checksum(payload)
        if corrupt_checksum:
            checksum ^= 0xFF
        return HEADER + payload + bytes([checksum])

    def test_well_formed_position_reply(self) -> None:
        # Present_Position = 1024 (little-endian) from motor 1, no error.
        raw = self._make_status(1, bytes([0x00, 0x04]))
        error, params = parse_status_packet(raw, expected_id=1, expected_param_count=2)
        assert error == 0
        assert params == bytes([0x00, 0x04])

    def test_error_byte_returned(self) -> None:
        # Bit 5 = OVERLOAD_ERROR on the STS3215.
        raw = self._make_status(3, b"", error=0x20)
        error, params = parse_status_packet(raw, expected_id=3, expected_param_count=0)
        assert error == 0x20
        assert params == b""

    def test_leading_echo_byte_tolerated(self) -> None:
        # A half-duplex bus can put a single 0xFF (the host's own echo) in
        # front of the reply. The parser must resync to the first FF FF.
        raw = b"\xff" + self._make_status(1, bytes([0x00, 0x04]))
        error, params = parse_status_packet(raw, expected_id=1, expected_param_count=2)
        assert error == 0
        assert params == bytes([0x00, 0x04])

    def test_leading_garbage_tolerated_up_to_header(self) -> None:
        # A garbled prefix that does not itself contain FF FF is skipped.
        raw = b"\x12\x34\x56" + self._make_status(2, bytes([0xAA]))
        error, params = parse_status_packet(raw, expected_id=2, expected_param_count=1)
        assert params == bytes([0xAA])

    def test_no_header_refused(self) -> None:
        with pytest.raises(ProtocolError, match="no FF FF header"):
            parse_status_packet(b"\x00\x11\x22\x33", expected_id=1, expected_param_count=0)

    def test_truncated_after_header_refused(self) -> None:
        # Header + ID + LEN only - four bytes short.
        with pytest.raises(ProtocolError, match="truncated"):
            parse_status_packet(b"\xff\xff\x01\x02", expected_id=1, expected_param_count=0)

    def test_trailing_bytes_refused(self) -> None:
        # A well-formed frame plus one extra byte at the end. The bus module
        # is responsible for slicing the stream into whole frames; a caller
        # that hands us a fragment plus its neighbour has confused two
        # frames, and answering that with the first one's params would
        # silently drop the second.
        raw = self._make_status(1, bytes([0x00, 0x04])) + b"\x77"
        with pytest.raises(ProtocolError, match="trailing"):
            parse_status_packet(raw, expected_id=1, expected_param_count=2)

    def test_id_mismatch_refused(self) -> None:
        # Motor 2 answered a read that was addressed to motor 1. This
        # happens on a shared bus when a stale reply lingers; naming it as
        # the right joint's measurement would report a wrong angle.
        raw = self._make_status(2, bytes([0x00, 0x04]))
        with pytest.raises(ProtocolError, match="ID mismatch"):
            parse_status_packet(raw, expected_id=1, expected_param_count=2)

    def test_length_mismatch_refused(self) -> None:
        # The servo returned 4 bytes when the READ asked for 2.
        raw = self._make_status(1, bytes([0x00, 0x04, 0x00, 0x00]))
        with pytest.raises(ProtocolError, match="LEN mismatch"):
            parse_status_packet(raw, expected_id=1, expected_param_count=2)

    def test_corrupted_checksum_refused(self) -> None:
        raw = self._make_status(1, bytes([0x00, 0x04]), corrupt_checksum=True)
        with pytest.raises(ProtocolError, match="checksum mismatch"):
            parse_status_packet(raw, expected_id=1, expected_param_count=2)

    def test_expected_id_broadcast_refused(self) -> None:
        # Expecting a reply from the broadcast is a caller bug.
        raw = self._make_status(1, b"")
        with pytest.raises(ValueError, match="expected_id"):
            parse_status_packet(raw, expected_id=BROADCAST_ID, expected_param_count=0)


# ---------------------------------------------------------------------------
# The vendor SDK, when installed, must agree with the codec byte-for-byte.
# ---------------------------------------------------------------------------
# ``scservo_sdk`` is Feetech's optional vendor SDK. It is an import name, not a
# PyPI project name: the distribution that provides it is ``feetech-servo-sdk``
# (1.0.0 on PyPI), and it does reach upstream CI - transitively, via
# ``lerobot[feetech]`` in the ``all`` extra the hatch default env installs.
# That makes it an incidental dependency: nothing in this repository asks for it
# on its own behalf, so if lerobot ever drops or renames its ``feetech`` extra a
# file gated on the SDK goes quiet with no test failing. A module-level
# ``pytest.importorskip`` made this file exactly that - it skips the ENTIRE
# file, silently deselecting all 45 datasheet-driven codec cells above, which is
# the "silent skip" defect class AGENTS.md > Review Learnings (PR #85) > Testing
# names. Scope the skip to the vendor-agreement class instead: the fixture
# imports the SDK per-class, so a box without the SDK still runs all 45 and
# skips only this class's 2 cells.
_scservo_sdk_available = importlib.util.find_spec("scservo_sdk") is not None

#: Repository root, so the source-level cells read the shipped file rather than
#: an ``inspect.getsource`` of an already-imported module.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Values spanning the word: both bytes zero, low byte only, high byte only, the
#: STS/SMS position full scale, and the two-byte maximum.
_WORD_VALUES = (0, 1, 255, 256, 511, 1023, 2048, MAX_GOAL_POSITION, 0xFFFF)


def _sdk_word(sdk, value: int, *, protocol: int) -> list[int]:  # type: ignore[no-untyped-def]
    """The two bytes the vendor SDK frames ``value`` as under ``protocol``.

    ``SCS_LOBYTE`` / ``SCS_HIBYTE`` read a module-global end-ness rather than
    taking it as an argument, and ``PacketHandler`` is what sets it. Selecting
    protocol 0 again in a ``finally`` keeps that global from leaking into any
    later cell.
    """
    try:
        sdk.PacketHandler(protocol)
        return [sdk.SCS_LOBYTE(value), sdk.SCS_HIBYTE(value)]
    finally:
        sdk.PacketHandler(0)


def _sdk_make_word(sdk, low: int, high: int, *, protocol: int) -> int:  # type: ignore[no-untyped-def]
    """The value a servo speaking ``protocol`` reads from the pair ``low, high``."""
    try:
        sdk.PacketHandler(protocol)
        return int(sdk.SCS_MAKEWORD(low, high))
    finally:
        sdk.PacketHandler(0)


@pytest.mark.skipif(
    not _scservo_sdk_available,
    reason="scservo_sdk not installed on this box",
)
class TestTheVendorAgreesOnFraming:
    """Every builder produces the same bytes the SDK's ``PacketHandler`` would.

    Skipped where ``scservo_sdk`` is absent (via ``skipif`` on the class, not a
    module-level ``importorskip`` - the latter would silently skip the 45 codec
    cells above too). The codec's own frame-shape tests above do not
    need the SDK, and the frames they build were taken from the datasheet
    rather than from this suite's own output.
    """

    @pytest.fixture(scope="class")
    def sdk(self):  # type: ignore[no-untyped-def]
        # Import inside the fixture so the module still imports on a box
        # without the SDK - the class's ``skipif`` guards collection, and the
        # fixture never runs when the class is skipped.
        return pytest.importorskip("scservo_sdk", reason="scservo_sdk not installed on this box")

    @pytest.fixture(scope="class")
    def handler(self, sdk):  # type: ignore[no-untyped-def]
        # Feetech's PacketHandler is a pure codec too - no port is opened.
        return sdk.PacketHandler(0)

    def test_ping_matches_sdk(self, sdk, handler) -> None:  # type: ignore[no-untyped-def]
        # ``sdk`` exposes byte-offset constants (PKT_HEADER0, PKT_ID,
        # PKT_LENGTH, PKT_INSTRUCTION) plus INST_PING. Assemble the PING
        # frame from those primitives so the SDK's own view of the layout
        # grades our builder.
        # PING has zero params so LEN = 2 (INSTR + CHECKSUM).
        sdk_frame = bytearray(
            [
                0xFF,  # PKT_HEADER0
                0xFF,  # PKT_HEADER1
                1,  # PKT_ID
                2,  # PKT_LENGTH_L on Protocol 1: 2
                sdk.INST_PING,
            ]
        )
        # Feetech additive checksum over ID..INSTR.
        sdk_frame.append((~sum(sdk_frame[sdk.PKT_ID :])) & 0xFF)
        assert ping_packet(1) == bytes(sdk_frame)

    def test_write_matches_sdk_checksum_formula(self, sdk, handler) -> None:  # type: ignore[no-untyped-def]
        # WRITE 2 bytes at address 0x2A on motor 5.
        motor_id = 5
        address = 0x2A
        data = [0x00, 0x08]
        length = len(data) + 3  # address + INSTR + CHECKSUM
        payload = [motor_id, length, sdk.INST_WRITE, address] + data
        checksum = (~sum(payload)) & 0xFF
        expected = bytes([0xFF, 0xFF] + payload + [checksum])
        assert write_packet(motor_id, address, bytes(data)) == expected

    @pytest.mark.parametrize("value", _WORD_VALUES)
    def test_encode_word_matches_the_sdk_protocol_0_order(self, sdk, value: int) -> None:  # type: ignore[no-untyped-def]
        # The SDK's LOBYTE/HIBYTE pair reads a module-global end-ness that
        # ``PacketHandler`` sets, so select protocol 0 - the STS/SMS order - and
        # restore it afterwards rather than leaving a global behind for whichever
        # cell runs next.
        assert encode_word(value) == bytes(_sdk_word(sdk, value, protocol=0))

    @pytest.mark.parametrize("value", _WORD_VALUES)
    def test_the_scs_series_reads_the_same_bytes_as_a_different_value(self, sdk, value: int) -> None:  # type: ignore[no-untyped-def]
        """Why :func:`encode_word` is series-specific rather than mis-scaled.

        ``PacketHandler(1)`` - the SCS series - reverses the pair, so the bytes
        this package puts on the wire are read by an SCS-series servo as the
        byte-swapped value. For every value whose two bytes differ that is a
        different position, which is what makes covering the SCS series a second
        word order rather than a scale option.
        """
        low, high = _sdk_word(sdk, value, protocol=0)
        assert _sdk_word(sdk, value, protocol=1) == [high, low]

        as_scs_reads_it = _sdk_make_word(sdk, low, high, protocol=1)
        if low == high:
            assert as_scs_reads_it == value, "a palindrome word is the one case both series agree on"
        else:
            assert as_scs_reads_it != value, (
                f"position {value} is framed as {[low, high]} and read by an SCS-series servo as {as_scs_reads_it}"
            )


class TestTheRegisterWordIsTheStsSeriesOrder:
    """A two-byte register value is framed low byte first, and only there.

    The order is a property of the servo *series*, not of the register or of
    Feetech generally: the vendor SDK reverses it on a per-model protocol number.
    These cells state the protocol 0 order this codec implements from the
    datasheet's own examples, so the SDK-graded cells above confirm rather than
    define it.
    """

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0, b"\x00\x00"),
            (1, b"\x01\x00"),
            (255, b"\xff\x00"),
            (256, b"\x00\x01"),
            (1023, b"\xff\x03"),  # full scale on an SCS-series servo
            (2048, b"\x00\x08"),  # mid-travel on an STS3215
            (MAX_GOAL_POSITION, b"\xff\x0f"),
            (0xFFFF, b"\xff\xff"),
        ],
    )
    def test_a_word_is_framed_low_byte_first(self, value: int, expected: bytes) -> None:
        assert encode_word(value) == expected
        assert len(encode_word(value)) == WORD_LENGTH
        assert decode_word(expected) == value

    @pytest.mark.parametrize("value", _WORD_VALUES)
    def test_a_word_survives_the_round_trip(self, value: int) -> None:
        assert decode_word(encode_word(value)) == value

    @pytest.mark.parametrize("value", [-1, 0x10000, 70000])
    def test_a_value_the_word_cannot_carry_is_refused_not_masked(self, value: int) -> None:
        # Masking would put a different, reachable command on the wire while the
        # caller is told the number they asked for went out.
        with pytest.raises(ValueError, match="out of range"):
            encode_word(value)

    @pytest.mark.parametrize("value", [True, 2.0, "2048", None])
    def test_a_non_integer_word_is_refused(self, value: object) -> None:
        # ``True`` is refused with the rest: it would silently encode as 1.
        with pytest.raises(TypeError, match="must be int"):
            encode_word(value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("raw", [b"", b"\x01", b"\x01\x02\x03"])
    def test_a_reply_that_is_not_one_word_is_refused(self, raw: bytes) -> None:
        # A short read decoded anyway reports a position the servo never sent.
        with pytest.raises(ValueError, match="bytes"):
            decode_word(raw)

    def test_a_bytearray_reply_decodes(self) -> None:
        # The bus reads into whatever pyserial hands back; both shapes decode.
        assert decode_word(bytearray(b"\xff\x03")) == 1023

    def test_a_reply_that_is_not_bytes_is_refused(self) -> None:
        with pytest.raises(TypeError, match="must be bytes"):
            decode_word([0xFF, 0x03])  # type: ignore[arg-type]


class TestTheWireFormatIsDecidedInOnePlace:
    """Every Feetech write path reads the word order and the full scale from here.

    Both are series properties, and the package spelled each of them several
    times: the two-byte split appeared as ``value & 0xFF, (value >> 8) & 0xFF``
    in the bus, in ``serial_tool`` (twice) and in ``pose_tool``, with the joining
    shift in two more places, and the STS/SMS full scale appeared as eight
    integer literals. A second spelling is what lets one of them be corrected for
    a different series while the others stay behind, which is the arrangement
    :issue:`2812` reports.
    """

    #: The modules that frame or read a Feetech register word.
    _WRITE_PATH = (
        "strands_robots/drivers/feetech/bus.py",
        "strands_robots/drivers/feetech/driver.py",
        "strands_robots/tools/serial_tool.py",
        "strands_robots/tools/pose_tool.py",
    )

    @staticmethod
    def _shift_by_eight_sites(source: str) -> list[int]:
        """Lines spelling a byte-order shift, i.e. ``<< 8`` or ``>> 8``."""
        tree = ast.parse(source)
        return sorted(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, (ast.LShift, ast.RShift))
            and isinstance(node.right, ast.Constant)
            and node.right.value == 8
        )

    @pytest.mark.parametrize("module_path", _WRITE_PATH)
    def test_no_consumer_spells_the_byte_order_itself(self, module_path: str) -> None:
        source = (_REPO_ROOT / module_path).read_text(encoding="utf-8")

        assert self._shift_by_eight_sites(source) == [], (
            f"{module_path} shifts a byte into place on lines "
            f"{self._shift_by_eight_sites(source)}; the order belongs to "
            "encode_word / decode_word so a servo series with the opposite order "
            "is one edit rather than six"
        )

    def test_the_codec_names_the_order_rather_than_shifting(self) -> None:
        source = (_REPO_ROOT / "strands_robots/drivers/feetech/protocol.py").read_text(encoding="utf-8")

        assert self._shift_by_eight_sites(source) == []
        assert '"little"' in source, "the order the codec implements is named, not encoded in shifts"

    @pytest.mark.parametrize("module_path", _WRITE_PATH)
    def test_no_consumer_restates_the_full_scale(self, module_path: str) -> None:
        tree = ast.parse((_REPO_ROOT / module_path).read_text(encoding="utf-8"))
        literals = sorted(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value.__class__ is int and node.value == MAX_GOAL_POSITION
        )

        assert literals == [], (
            f"{module_path} spells the STS/SMS full scale {MAX_GOAL_POSITION} on lines "
            f"{literals}; it is a series property and MAX_GOAL_POSITION is where it is decided"
        )


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="lerobot not installed on this box",
)
class TestLerobotAgreesOnWhichSeriesTheFullScaleBelongsTo:
    """lerobot's motor tables are the oracle for the resolution and the protocol.

    Skipped where lerobot is absent, like the vendor-SDK class above: the codec's
    own cells do not need it, and its tables are what let this package state that
    4096 counts is the STS/SMS number rather than the Feetech number.
    """

    def test_the_full_scale_is_the_sts_series_resolution(self) -> None:
        tables = pytest.importorskip("lerobot.motors.feetech.tables")

        for model in ("sts3215", "sts3250", "sm8512bl"):
            assert tables.MODEL_RESOLUTION[model] - 1 == MAX_GOAL_POSITION, model
            assert tables.MODEL_PROTOCOL[model] == 0, f"{model} is the low-byte-first order"

    def test_the_scs_series_is_a_different_scale_and_a_different_order(self) -> None:
        tables = pytest.importorskip("lerobot.motors.feetech.tables")

        assert tables.MODEL_RESOLUTION["scs0009"] - 1 != MAX_GOAL_POSITION
        assert tables.MODEL_RESOLUTION["scs0009"] == 1024, "10-bit, a quarter of the STS/SMS scale"
        assert tables.MODEL_PROTOCOL["scs0009"] == 1, "the high-byte-first order this codec does not emit"
