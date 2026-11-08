"""Unitree's LowCmd CRC: their bitwise CRC-32 (poly 0x04C11DB7) over the packed message, table-driven.

Unitree feeds the packed struct as little-endian 32-bit words, MSB first, and folds each data bit in
AFTER the shift. That update is linear over GF(2), so one byte step is ``(crc << 8) ^ T[crc >> 24] ^ U[b]``:
the same result as the SDK's bit loop (checked against it in the tests) at about 1/30 of the cost,
which matters at 500 Hz in Python.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

_POLY, _MASK = 0x04C11DB7, 0xFFFFFFFF
HG_LOWCMD_FMT = "<2B2x" + "B3x5fI" * 35 + "5I"  # mode_pr, mode_machine, 35 x MotorCmd, reserve[4], crc


def _bits(crc: int, byte: int) -> int:
    for i in range(8):
        crc = ((crc << 1) & _MASK) ^ (_POLY if crc & 0x80000000 else 0)
        if byte & (0x80 >> i):
            crc ^= _POLY
    return crc


_T = tuple(_bits(i << 24, 0) for i in range(256))
_U = tuple(_bits(0, b) for b in range(256))


def crc32_words(words: Sequence[int]) -> int:
    crc = 0xFFFFFFFF
    for w in words:
        for shift in (24, 16, 8, 0):
            crc = ((crc << 8) & _MASK) ^ _T[crc >> 24] ^ _U[(w >> shift) & 0xFF]
    return crc


def hg_lowcmd_crc(
    mode_pr: int, mode_machine: int, motors: Sequence[Sequence[float]], reserve: Sequence[int] = (0, 0, 0, 0)
) -> int:
    """CRC for a ``unitree_hg`` LowCmd. ``motors``: 35 rows of (mode, q, dq, tau, kp, kd, reserve)."""
    if len(motors) != 35:
        raise ValueError(f"unitree_hg LowCmd has 35 motor slots, got {len(motors)}")
    flat: list[float | int] = [int(mode_pr), int(mode_machine)]
    for m in motors:
        flat += [int(m[0]), float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5]), int(m[6])]
    flat += [int(x) for x in reserve] + [0]
    packed = struct.pack(HG_LOWCMD_FMT, *flat)
    n = (len(packed) >> 2) - 1  # the crc word itself is excluded
    return crc32_words(struct.unpack(f"<{n}I", packed[: n * 4]))
