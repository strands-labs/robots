"""Joint readings and the Action a policy returns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np


def _ro(a: Any, dtype: Any = np.float64) -> np.ndarray:
    """A read-only float array (frozen dataclasses deserve frozen arrays)."""
    out = np.array(a, dtype=dtype)
    out.flags.writeable = False
    return out


def _same(a: Any, b: Any) -> bool:
    """Value equality through arrays and mappings: what ``==`` means on every type here."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return bool(a == b)


def _eq(self: Any, other: Any) -> bool:
    if type(other) is not type(self):
        return NotImplemented
    return all(_same(getattr(self, f.name), getattr(other, f.name)) for f in fields(self))


def _key(a: np.ndarray | None) -> bytes | None:
    """A hash key that agrees with ``np.array_equal`` (``+ 0.0`` folds -0.0 into 0.0)."""
    return None if a is None else (a + 0.0).tobytes()


@dataclass(frozen=True, eq=False)
class JointState:
    """Positions/velocities/efforts for a named set of joints (SI: rad, rad/s, N·m). A value: ``==``, ``hash``."""

    names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.names)
        for f in ("positions", "velocities", "efforts"):
            arr = _ro(getattr(self, f))
            if arr.shape != (n,):
                raise ValueError(f"JointState.{f} has shape {arr.shape}, expected ({n},) for {n} joints")
            object.__setattr__(self, f, arr)

    __eq__ = _eq

    def __hash__(self) -> int:
        return hash((self.names, _key(self.positions), _key(self.velocities), _key(self.efforts)))

    def __getitem__(self, joint: str) -> float:
        try:
            return float(self.positions[self.names.index(joint)])
        except ValueError:
            raise KeyError(f"unknown joint {joint!r}; joints are {list(self.names)}") from None

    def __len__(self) -> int:
        return len(self.names)

    def as_dict(self) -> dict[str, float]:
        return {n: float(p) for n, p in zip(self.names, self.positions, strict=True)}

    @classmethod
    def zeros(cls, names: tuple[str, ...] | list[str]) -> JointState:
        z = np.zeros(len(names))
        return cls(tuple(names), z, z.copy(), z.copy())


Action = np.ndarray | Mapping[str, float] | JointState
"""Joint targets, either a vector in ``Description.action_keys`` order, a ``{joint: target}`` mapping
(missing joints keep their last commanded target) or the ``JointState`` an observation measured (the
measured pose as the target: once, a hold; every step, it follows gravity down)."""
