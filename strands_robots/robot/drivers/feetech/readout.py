"""One sync_read reply per servo -> a JointState: Present_Position alone, or with velocity and load.

The three registers are contiguous from ``REG_PRESENT_POSITION``: 2 bytes is position only, 6 bytes adds velocity and
load. A position-only read reports velocity and effort as NaN, the reading every body gives for a value it did not
measure (a ROS position-only joint does the same; the mesh sends it as ``null``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ....types import JointState
from .bus import decode_sm

if TYPE_CHECKING:
    from .driver import FeetechDriver

POSITION_BYTES, MOTION_BYTES = 2, 6


def joint_state(drv: FeetechDriver, raw: Mapping[int, bytes]) -> JointState:
    """Radians (and rad/s, signed fraction of max torque) in ``drv.names`` order from per-servo register bytes."""
    pairs = list(zip(drv.names, drv.ids, strict=True))
    pos = np.array([drv.counts_to_rad(name, decode_sm(raw[mid][0:2], 15)) for name, mid in pairs])
    if all(len(raw[mid]) < MOTION_BYTES for _, mid in pairs):
        unmeasured = np.full(len(pairs), np.nan)
        return JointState(drv.names, pos, unmeasured, unmeasured.copy())
    vel, load = [], []
    for name, mid in pairs:
        cmin, cmax, lo, hi, rev = drv._span(name)
        vel.append(decode_sm(raw[mid][2:4], 15) * (hi - lo) / (cmax - cmin) * (-1 if rev else 1))
        load.append(decode_sm(raw[mid][4:6], 10) / 1000.0)
    return JointState(drv.names, pos, np.array(vel), np.array(load))
