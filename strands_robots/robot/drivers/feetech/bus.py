"""Feetech STS/SMS-series servo bus wire protocol and half-duplex transport.

Wire: ``FF FF ID LEN INSTR params CHK``, ``CHK = ~sum(ID..params) & 0xFF``, 1 Mbaud, little-endian words,
sign-magnitude values (bit 15 for positions/velocities, bit 10 for load, bit 11 for homing offset).
``pyserial`` is imported lazily; any object with ``write``/``read``/``reset_input_buffer`` can be injected.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from ...._extras import hint

HEADER = b"\xff\xff"
BROADCAST = 0xFE
RESOLUTION = 4096

PING, READ, WRITE, SYNC_READ, SYNC_WRITE = 0x01, 0x02, 0x03, 0x82, 0x83
REG_HOMING_OFFSET = 0x1F  # EEPROM, 2 bytes, sign bit 11
REG_TORQUE_ENABLE = 0x28
REG_GOAL_POSITION = 0x2A  # 2 bytes
REG_LOCK = 0x37
REG_PRESENT_POSITION = 0x38  # 2 bytes, then velocity 0x3A, load 0x3C (contiguous: one 6-byte read)


class ProtocolError(ValueError):
    pass


def checksum(payload: bytes) -> int:
    return (~sum(payload)) & 0xFF


def frame(motor_id: int, instr: int, params: bytes = b"") -> bytes:
    payload = bytes([motor_id, len(params) + 2, instr]) + params
    return HEADER + payload + bytes([checksum(payload)])


def word(value: int) -> bytes:
    return int(value).to_bytes(2, "little")


def decode_sm(raw: bytes, sign_bit: int) -> int:
    """Sign-magnitude word: the named bit is direction, the bits below it magnitude."""
    v = int.from_bytes(raw, "little")
    mag = v & ((1 << sign_bit) - 1)
    return -mag if v & (1 << sign_bit) else mag


def encode_sm(value: int, sign_bit: int) -> bytes:
    mag = min(abs(int(value)), (1 << sign_bit) - 1)
    return word(mag | (1 << sign_bit) if value < 0 else mag)


def parse_status(raw: bytes, expect_id: int, nparams: int) -> tuple[int, bytes]:
    i = raw.find(HEADER)
    if i < 0:
        raise ProtocolError(f"no FF FF header in {len(raw)} bytes from servo {expect_id}")
    f = raw[i:]
    if len(f) < 6 + nparams:
        raise ProtocolError(f"servo {expect_id}: truncated reply ({len(f)} bytes, need {6 + nparams})")
    mid, length, err = f[2], f[3], f[4]
    if mid != expect_id:
        raise ProtocolError(f"reply from servo {mid}, expected {expect_id}")
    if length != nparams + 2:
        raise ProtocolError(f"servo {expect_id}: LEN {length} != {nparams + 2}")
    params = bytes(f[5 : 5 + nparams])
    if f[5 + nparams] != checksum(f[2 : 5 + nparams]):
        raise ProtocolError(f"servo {expect_id}: bad checksum")
    return err, params


MAX_SLIDE = 64  # bytes skipped looking for FF FF before a reply is refused: a UART break streams 0x00 forever


class FeetechBus:
    """Half-duplex serial bus. ``serial`` may be injected (tests); otherwise pyserial opens ``port``."""

    def __init__(self, port: str, *, baud: int = 1_000_000, timeout: float = 0.02, serial: Any = None) -> None:
        self.port = port
        if serial is None:
            try:
                import serial as pyserial
            except ImportError as exc:
                raise ImportError(f"the feetech driver needs pyserial: {hint('feetech')}") from exc
            serial = pyserial.Serial(port, baudrate=baud, timeout=timeout)
        self.serial = serial
        self.wire = threading.RLock()  # one transaction at a time: an e-stop from any thread never collides
        self.echoes: bool | None = None  # learned from the wire: does this adapter hear its own request?

    def _txrx(self, packet: bytes, expect_id: int, nparams: int) -> tuple[int, bytes]:
        with self.wire:
            self.serial.reset_input_buffer()
            self.serial.write(packet)
            raw = self._echoed(packet, self._frame())
        return parse_status(raw, expect_id, nparams)

    def _echoed(self, packet: bytes, raw: bytes) -> bytes:
        """Drop the request an echoing adapter hears; learn which kind of adapter this is.

        A frame equal to the request is the echo unless the adapter is known not to echo (a reply can be
        byte-identical: ``ERR`` equal to the instruction byte). Echo then silence is an unanswered request.
        """
        if raw != packet:
            if raw.startswith(HEADER) and self.echoes is None:
                self.echoes = False
            return raw
        if self.echoes is False:
            return raw
        reply = self._frame()
        if reply:
            self.echoes = True
        return reply

    def _frame(self) -> bytes:
        """One frame off the wire: find ``FF FF``, read ``ID LEN``, then exactly ``LEN`` bytes, never a guess.

        A healthy reply is under a millisecond; asking for a byte that never comes costs the whole ``timeout``.
        """
        head = bytes(self.serial.read(4))
        for _ in range(MAX_SLIDE):  # stray bytes ahead of the header; a line streaming no header stops here
            if len(head) < 4 or head[:2] == HEADER:
                break
            head = head[1:] + bytes(self.serial.read(1))
        return head + bytes(self.serial.read(head[3])) if len(head) == 4 and head[:2] == HEADER else head

    def ping(self, motor_id: int) -> bool:
        try:
            self._txrx(frame(motor_id, PING), motor_id, 0)
            return True
        except ProtocolError:
            return False

    def read(self, motor_id: int, reg: int, n: int) -> bytes:
        _, params = self._txrx(frame(motor_id, READ, bytes([reg, n])), motor_id, n)
        return params

    def write(self, motor_id: int, reg: int, data: bytes) -> None:
        self._txrx(frame(motor_id, WRITE, bytes([reg]) + data), motor_id, 0)

    def sync_read(self, reg: int, n: int, ids: list[int], retries: int = 0) -> dict[int, bytes]:
        """One reply per servo; a garbled or missing reply is asked for again, up to ``retries`` more times."""
        for _ in range(retries):
            try:
                return self._sync_read(reg, n, ids)
            except ProtocolError:
                continue
        return self._sync_read(reg, n, ids)

    def _sync_read(self, reg: int, n: int, ids: list[int]) -> dict[int, bytes]:
        packet = frame(BROADCAST, SYNC_READ, bytes([reg, n]) + bytes(ids))
        out: dict[int, bytes] = {}
        with self.wire:
            self.serial.reset_input_buffer()
            self.serial.write(packet)
            for mid in ids:  # the servos answer in id order, one frame each
                raw = self._echoed(packet, self._frame()) if mid == ids[0] else self._frame()
                if not raw.startswith(HEADER):
                    raise ProtocolError(f"sync_read: no reply from servo {mid} (got {len(out)} of {len(ids)})")
                out[mid] = parse_status(raw, mid, n)[1]
        return out

    def sync_write(self, reg: int, per: int, items: Mapping[int, bytes]) -> None:
        params = bytes([reg, per]) + b"".join(bytes([mid]) + data for mid, data in items.items())
        with self.wire:
            self.serial.write(frame(BROADCAST, SYNC_WRITE, params))

    def torque(self, ids: list[int], on: bool) -> None:
        self.sync_write(REG_TORQUE_ENABLE, 1, {i: bytes([1 if on else 0]) for i in ids})

    def close(self) -> None:
        self.serial.close()
