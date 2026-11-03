"""What a world made of: a contact, and the engine's time policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Contact:
    """One contact between two bodies (world frame)."""

    body_a: str
    body_b: str
    position: np.ndarray
    normal: np.ndarray
    force: float


@dataclass(frozen=True)
class Physics:
    """Which engine steps the world, how finely, and how many physics steps make one control step."""

    name: str
    timestep: float
    substeps: int

    @property
    def control_dt(self) -> float:
        return self.timestep * self.substeps
