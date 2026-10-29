"""A bus with no wire: ``Robot("so101", driver="memory")`` is a hardware body with no arm attached.

Writes land at once and the next read returns them, so the operator gate (``confirm=``), ``approve=``, the
e-stop and the watchdog's real-body rule run exactly as on an arm: for docs, notebooks and CI.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...description import Description
from ...types import JointState, Pose
from .calibration import Calibration


class MemoryDriver:
    """``writes`` logs every target written, in order; ``holds`` and ``estops`` count those verbs."""

    supports_position_hold = True
    calibration: Calibration | None = None
    resolution: int | None = None

    def __init__(self, description: Description, *, port: str | None = None, cameras: Mapping[str, Any] | None = None):
        if port is not None or cameras:
            raise TypeError("the memory driver has no wire: it takes no port= and no cameras=")
        self.names: tuple[str, ...] = tuple(description.action_keys)
        self.q: np.ndarray = np.zeros(len(self.names))
        self.writes: list[np.ndarray] = []
        self.holds = self.estops = 0

    def read(self) -> JointState:
        z = np.zeros(len(self.names))
        return JointState(self.names, self.q.copy(), z, z.copy())

    def write(self, targets: np.ndarray) -> None:
        self.q = np.asarray(targets, dtype=float).copy()
        self.writes.append(self.q.copy())

    def pose(self) -> Pose | None:
        return None

    def images(self) -> Mapping[str, np.ndarray]:
        return {}

    def extras(self) -> Mapping[str, Any]:
        return {"writes": len(self.writes), "holds": self.holds, "estops": self.estops}

    def hold(self) -> None:
        self.holds += 1

    def estop(self) -> None:
        self.estops += 1

    def calibrate(self, *, interactive: bool = True) -> Calibration:
        raise RuntimeError("memory: there is no bus to calibrate")

    def close(self) -> None:
        """Nothing to release."""
