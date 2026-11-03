"""Quaternion math and rigid-frame helpers, numpy only (no engine).

A unit quaternion is ``(w, x, y, z)``. These are the maps a world uses to move a
spawn frame when a body teleports; the MuJoCo backend uses the same functions so
there is one definition of the algebra, not one per engine.
"""

from __future__ import annotations

import numpy as np

from .types import Pose


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The Hamilton product ``a * b`` of two ``(w, x, y, z)`` quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """The conjugate (inverse, for a unit quaternion) of ``q``."""
    return np.asarray(q, dtype=float) * [1, -1, -1, -1]


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q``."""
    return quat_mul(quat_mul(q, np.r_[0.0, v]), quat_conj(q))[1:]


def moved_frame(frame: Pose, old: Pose, new: Pose) -> Pose:
    """The spawn frame that puts a body at ``new`` when it was at ``old`` under ``frame``: new * old^-1 * frame."""
    q = quat_mul(new.orientation, quat_conj(old.orientation))
    q = q / np.linalg.norm(q)
    return Pose(new.position + quat_rotate(q, frame.position - old.position), quat_mul(q, frame.orientation))


def vec(value: object, n: int, what: str) -> tuple[float, ...]:
    """``value`` as ``n`` floats; a string or a wrong length is refused by name, never read character by character."""
    if isinstance(value, str | bytes) or not hasattr(value, "__iter__"):
        raise TypeError(f"{what} must be {n} numbers, got {value!r}")
    out = tuple(float(x) for x in value)
    if len(out) != n:
        raise ValueError(f"{what} must be {n} numbers, got {len(out)}")
    return out
