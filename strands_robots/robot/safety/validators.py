"""Validators: pure checks ``(device, request) -> None`` that raise ``RequestRefused`` naming the bad input.

Validators are advisory command checks for agents and dashboards, not functional safety: a real body
keeps its own firmware limits and an operator keeps the e-stop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..robot import Robot

_EPS = 1e-9


class RequestRefused(PermissionError):
    """A command the device will not carry out; the message names the input, its value and the bound."""


class NonFiniteTarget(ValueError):
    """A NaN or infinite target: refused before it reaches a body; a run ends with ``stopped_by="fault"``."""


@dataclass(frozen=True)
class Request:
    """A control write on its way through the safety chain: full targets in ``action_keys`` order.

    ``measured`` (same order, or None) is where the body actually sits: a target is never refused for a
    limit the body already rests past, only for driving a joint further out (``JointLimitValidator``).
    """

    keys: tuple[str, ...]
    targets: np.ndarray
    confirmed: bool = False
    source: str = "agent"
    measured: np.ndarray | None = None

    def items(self) -> list[tuple[str, float]]:
        return [(k, float(v)) for k, v in zip(self.keys, self.targets, strict=True)]


Validator = Callable[["Robot", Request], Any]


def _outside(v: float, bound: tuple[float, float] | None) -> bool:
    return bound is not None and not (bound[0] - _EPS <= v <= bound[1] + _EPS)


class JointLimitValidator:
    """Refuse a target that drives a joint further outside its hard ``range`` than the body already sits.

    A joint can rest a hair past a soft (MuJoCo) or calibration (hardware: ``counts_to_rad`` has no clamp)
    limit; holding there, or moving back toward the range, is always allowed. Only a target beyond the
    body's measured position on the outside is refused, so an agent run that holds a saturated pose, or a
    ``confirm_act`` of the measured joints, is not refused for where the body already is.
    """

    def __call__(self, robot: Robot, request: Request) -> None:
        manifest = robot.manifest
        measured = request.measured
        for i, (key, v) in enumerate(request.items()):
            rng = manifest.control(f"{key}.pos").range
            if rng is None:
                continue
            lo, hi = rng
            if measured is not None:  # the body may read past a soft/calibration limit: it can hold there
                m = float(measured[i])
                lo, hi = min(lo, m), max(hi, m)
            if _outside(v, (lo, hi)):
                raise RequestRefused(f"{robot.id}: {key}={v:.4f} rad is outside its range [{rng[0]:.4f}, {rng[1]:.4f}]")


class WorkspaceBoundsValidator:
    """Refuse targets whose tool point ``fk(targets) -> xyz`` (metres, world frame) leaves the box ``lo..hi``."""

    def __init__(self, fk: Callable[[np.ndarray], Any], *, lo: Sequence[float], hi: Sequence[float]) -> None:
        self.fk, self.lo, self.hi = fk, np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def __call__(self, robot: Robot, request: Request) -> None:
        p = np.asarray(self.fk(request.targets), dtype=float)
        for axis, v, a, b in zip("xyz", p, self.lo, self.hi, strict=True):
            if _outside(float(v), (float(a), float(b))):
                raise RequestRefused(f"{robot.id}: tool {axis}={v:.4f} m is outside the workspace [{a:.4f}, {b:.4f}]")


def outside_safe(robot: Robot, request: Request) -> list[str]:
    """Each target inside ``range`` but outside the soft ``safe`` bound, as a readable phrase."""
    out = []
    for key, v in request.items():
        safe = robot.manifest.control(f"{key}.pos").safe
        if _outside(v, safe):
            assert safe is not None
            out.append(f"{key}={v:.4f} rad outside safe [{safe[0]:.4f}, {safe[1]:.4f}]")
    return out


def compose(*validators: Validator | Sequence[Validator]) -> Validator:
    """One validator that runs the given ones in order and stops at the first refusal."""
    flat: list[Validator] = []
    for v in validators:
        flat.extend(v if isinstance(v, Sequence) else [v])

    def run_all(robot: Robot, request: Request) -> None:
        for v in flat:
            v(robot, request)

    return run_all
