"""What the page sees: a world camera's picture (``sim.render``, what a policy sees), sent as PNG."""

from __future__ import annotations

import struct
import zlib

import numpy as np


def png(img: np.ndarray) -> bytes:
    """An HxWx3 uint8 frame as PNG bytes (stdlib only: the dashboard extra carries no imaging library)."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in np.ascontiguousarray(img, dtype=np.uint8))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    head = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
