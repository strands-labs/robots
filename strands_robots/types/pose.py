"""A rigid pose in the world frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .joints import _eq, _key, _ro


@dataclass(frozen=True, eq=False)
class Pose:
    """A rigid pose in the world frame: position (m) and orientation as a unit quaternion (w, x, y, z).

    A floating body also reports its twist: ``linear_velocity`` (world frame, m/s) and ``angular_velocity``
    (body frame, rad/s — what a gyro mounted on the base reads). Fixed bases and drivers leave both ``None``.
    """

    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None

    __eq__ = _eq

    def __hash__(self) -> int:
        return hash(
            tuple(_key(getattr(self, f)) for f in ("position", "orientation", "linear_velocity", "angular_velocity"))
        )

    def __post_init__(self) -> None:
        p, q = _ro(self.position), _ro(self.orientation)
        if p.shape != (3,):
            raise ValueError(f"Pose.position has shape {p.shape}, expected (3,)")
        if q.shape != (4,):
            raise ValueError(f"Pose.orientation has shape {q.shape}, expected (4,) as (w, x, y, z)")
        object.__setattr__(self, "position", p)
        object.__setattr__(self, "orientation", q)
        for f in ("linear_velocity", "angular_velocity"):
            v = getattr(self, f)
            if v is not None:
                arr = _ro(v)
                if arr.shape != (3,):
                    raise ValueError(f"Pose.{f} has shape {arr.shape}, expected (3,)")
                object.__setattr__(self, f, arr)

    @classmethod
    def identity(cls, position: Any = (0.0, 0.0, 0.0)) -> Pose:
        return cls(np.asarray(position, dtype=float), np.array([1.0, 0.0, 0.0, 0.0]))
