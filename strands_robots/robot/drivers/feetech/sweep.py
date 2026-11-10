"""The range sweep of ``FeetechDriver.calibrate()``: each joint's min/max Present_Position while a human moves it."""

from __future__ import annotations

import time
from collections.abc import Sequence

from .bus import REG_PRESENT_POSITION, RESOLUTION, FeetechBus, decode_sm


def sweep_ranges(
    bus: FeetechBus, ids: Sequence[int], names: Sequence[str], seconds: float
) -> tuple[dict[int, int], dict[int, int]]:
    """``(range_min, range_max)`` per motor id, read for ``seconds``. wrist_roll is never swept: LeRobot records it
    as the full turn. A joint nobody moved has ``range_min >= range_max``."""
    full = {i for n, i in zip(names, ids, strict=True) if n == "wrist_roll"}
    lo = {i: 0 if i in full else RESOLUTION for i in ids}
    hi = {i: RESOLUTION - 1 if i in full else 0 for i in ids}
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        raw = bus.sync_read(REG_PRESENT_POSITION, 2, list(ids))
        for i in set(ids) - full:
            v = decode_sm(raw[i], 15)
            lo[i], hi[i] = min(lo[i], v), max(hi[i], v)
        time.sleep(0.01)
    return lo, hi
