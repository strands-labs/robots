"""Sensor-driven grasp DECISION layer for the bed-making G1s (issue #2).

Split of responsibility (the field-standard loco-manipulation architecture, and what
keeps this sim-to-real):

* The **whole-body reach policy is trusted with the MOTION** — it balances the robot on
  its own two feet and moves the arm/waist to a commanded hand target. We never move the
  robot (or the cloth) kinematically.
* This **decision layer owns the GRASP** — *when* to close the physical cloth grip and
  *when* to let go. It does NOT reposition the robot; it only reads a hand sensor and
  decides to attach / hold / release the PhysX cloth attachment.

The sensor is a **short-range hand proximity/tactile sensor**: a real G1 hand
(Inspire / BrainCo) carries fingertip proximity + tactile sensing that registers an
object only within a few centimetres — it sees *nothing* at arm's length. We model that
honestly: the decision only acts on the cloth once the hand is within ``sensing_range``
(a real grab is a near-contact event), so the behaviour transfers to a real hand sensor.
The exact distance to the nearest cloth particle (read from the PhysX cloth view) is the
sensor's stand-in signal; the *true* min distance is logged separately for diagnostics
only and is never used to drive the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


def _min_gap(hand_xyz: Sequence[float], cloth_pts) -> float:
    """Smallest distance from the hand to any cloth particle (m). inf if no cloth."""
    if cloth_pts is None or len(cloth_pts) == 0:
        return math.inf
    import numpy as np

    h = np.asarray(hand_xyz, dtype=float)
    return float(np.min(np.linalg.norm(np.asarray(cloth_pts, dtype=float) - h, axis=1)))


@dataclass
class HandClothSensor:
    """Short-range hand proximity/tactile sensor. ``read`` returns the sensed distance to
    the cloth ONLY when it is within physical sensing range (else ``None`` = "feels
    nothing"), so the decision layer behaves like it would on a real fingertip sensor."""

    sensing_range: float = 0.15   # m — a real hand sensor sees nothing beyond a few cm

    def read(self, hand_xyz: Sequence[float], cloth_pts) -> Optional[float]:
        gap = _min_gap(hand_xyz, cloth_pts)
        return gap if gap <= self.sensing_range else None

    @staticmethod
    def true_gap(hand_xyz: Sequence[float], cloth_pts) -> float:
        """Ground-truth min distance — DIAGNOSTIC ONLY (logging / closest-approach), never
        fed to the robot. A real robot does not have this; the policy never sees it."""
        return _min_gap(hand_xyz, cloth_pts)


class GraspDecision:
    """Per-hand grasp state machine driven by :class:`HandClothSensor`.

    SEEKING — the policy is reaching; grab the instant the sensor reports contact.
    GRIPPED — cloth attached; while pulling, if the sensor loses the cloth (gap grows past
              ``slip_gap``) the grip has slipped → release (requirement E: let go on a bad
              hold rather than fight it and topple).
    """

    def __init__(self, contact_gap: float = 0.10, slip_gap: float = 0.30, slip_patience: int = 10):
        self.contact_gap = contact_gap    # within this the cloth is in the hand → grab
        self.slip_gap = slip_gap          # gripped but sensor lost it this far → maybe slipped
        self.slip_patience = slip_patience  # consecutive lost reads before we call it a real slip
        self.gripped = False
        self.slipped = False
        self.grabbed_at: Optional[float] = None   # sensed gap at the moment of grab
        self.min_true_gap = math.inf      # diagnostic: closest the hand ever came
        self._lost = 0                    # consecutive control steps the sensor has lost the cloth

    def on_reach(self, sensed: Optional[float], true_gap: float) -> bool:
        """Call each control step while reaching. Returns True if the grip should close now
        (sensor reports cloth within contact range)."""
        self.min_true_gap = min(self.min_true_gap, true_gap)
        if self.gripped or sensed is None:
            return False
        if sensed <= self.contact_gap:
            self.gripped = True
            self.grabbed_at = sensed
            return True
        return False

    def on_pull(self, sensed: Optional[float], true_gap: float) -> bool:
        """Call each control step while pulling. Returns True if the grip has slipped and
        should be released. The sensor losing the cloth (no reading, or a reading beyond
        ``slip_gap``) means the hand no longer holds it."""
        self.min_true_gap = min(self.min_true_gap, true_gap)
        if not self.gripped or self.slipped:
            return False
        lost = (sensed is None) or (sensed > self.slip_gap)
        self._lost = self._lost + 1 if lost else 0
        if self._lost >= self.slip_patience:   # debounce: a transient stretch isn't a slip
            self.slipped = True
            return True
        return False

    @property
    def holding(self) -> bool:
        return self.gripped and not self.slipped
