"""Short-range hand proximity sensing + grasp decision (bundled copy).

This is the issue-#2 sensor/decision layer (``examples/isaac_bed_making/grasp.py``)
bundled per the repo's self-contained-demo convention, with distances in **cm** to
match the Newton cloth stack's units. It is representation-agnostic: it only needs
the hand position and the live cloth points.

The sensor models a real Inspire/BrainCo fingertip proximity+tactile sensor: it
reports a distance ONLY within a few centimetres (a real grab is a near-contact
event) and feels nothing at arm's length — so no privileged sim state ever drives
the behaviour. Ground-truth distance is exposed separately for diagnostics only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def _min_gap(hand_xyz: Sequence[float], cloth_pts) -> float:
    if cloth_pts is None or len(cloth_pts) == 0:
        return math.inf
    h = np.asarray(hand_xyz, dtype=float)
    return float(np.min(np.linalg.norm(np.asarray(cloth_pts, dtype=float) - h, axis=1)))


@dataclass
class HandClothSensor:
    """Short-range proximity/tactile sensor (cm)."""

    sensing_range: float = 15.0   # cm

    def read(self, hand_xyz: Sequence[float], cloth_pts) -> Optional[float]:
        gap = _min_gap(hand_xyz, cloth_pts)
        return gap if gap <= self.sensing_range else None

    @staticmethod
    def true_gap(hand_xyz: Sequence[float], cloth_pts) -> float:
        """DIAGNOSTIC ONLY — never fed to the behaviour."""
        return _min_gap(hand_xyz, cloth_pts)


class GraspDecision:
    """Per-hand grasp state machine (see issue-#2 grasp.py for the rationale).

    SEEKING — close the moment the sensor reports cloth within ``contact_gap``.
    GRIPPED — while pulling, sensor losing the cloth past ``slip_gap`` for
              ``slip_patience`` consecutive reads = the grip slipped → release
              (requirement E: let go rather than fight and topple).
    """

    def __init__(self, contact_gap: float = 10.0, slip_gap: float = 30.0,
                 slip_patience: int = 10):
        self.contact_gap = contact_gap
        self.slip_gap = slip_gap
        self.slip_patience = slip_patience
        self.gripped = False
        self.slipped = False
        self.grabbed_at: Optional[float] = None
        self.min_true_gap = math.inf
        self._lost = 0

    def on_reach(self, sensed: Optional[float], true_gap: float) -> bool:
        self.min_true_gap = min(self.min_true_gap, true_gap)
        if self.gripped or sensed is None:
            return False
        if sensed <= self.contact_gap:
            self.gripped = True
            self.grabbed_at = sensed
            return True
        return False

    def on_pull(self, sensed: Optional[float], true_gap: float) -> bool:
        self.min_true_gap = min(self.min_true_gap, true_gap)
        if not self.gripped or self.slipped:
            return False
        lost = (sensed is None) or (sensed > self.slip_gap)
        self._lost = self._lost + 1 if lost else 0
        if self._lost >= self.slip_patience:
            self.slipped = True
            return True
        return False

    @property
    def holding(self) -> bool:
        return self.gripped and not self.slipped
