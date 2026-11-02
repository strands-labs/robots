"""The unit map and the scheme registry: a policy declares what it speaks, the robot converts.

The robot speaks radians; a policy speaks whatever its dataset stored. A ``Units`` is the affine map
between the two, built for a given robot by a named *scheme*. Schemes register themselves
(:func:`register`), so adding one is adding a module, not editing an ``if``-chain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..description import Description

RAD = "rad"
SCHEMES: dict[str, Callable[[Description], Units]] = {}


def register(name: str) -> Callable[[Callable[[Description], Units]], Callable[[Description], Units]]:
    """Register a scheme builder under ``name`` (used as a decorator in ``rad.py`` / ``lerobot.py``)."""

    def deco(fn: Callable[[Description], Units]) -> Callable[[Description], Units]:
        SCHEMES[name] = fn
        return fn

    return deco


@dataclass(frozen=True)
class Units:
    """An affine map rad -> policy units per action key: ``u = lo_u + (q - lo) / (hi - lo) * (hi_u - lo_u)``."""

    name: str
    keys: tuple[str, ...]
    lo: np.ndarray  # rad
    hi: np.ndarray  # rad
    lo_u: np.ndarray  # policy units
    hi_u: np.ndarray  # policy units
    modes: tuple[str, ...] = ()  # per key, when the scheme has modes (LeRobot's MotorNormMode)

    @classmethod
    def for_robot(cls, description: Description, name: str) -> Units:
        """The converter between this description's radians and the named unit system."""
        try:
            builder = SCHEMES[name]
        except KeyError:
            raise ValueError(f"unknown units {name!r}; known: {', '.join(sorted(SCHEMES))}") from None
        return builder(description)

    @property
    def is_identity(self) -> bool:
        return self.name == RAD

    @property
    def bounded(self) -> np.ndarray:
        """Per key: does this unit system clamp to its range (LeRobot's ``range_*`` modes do, ``degrees`` not)."""
        return np.array([m.startswith("range_") for m in self.modes] or [False] * len(self.keys), dtype=bool)

    def from_rad(self, q: np.ndarray) -> np.ndarray:
        """rad -> policy units; a bounded key clamps into its range exactly as LeRobot's ``_normalize`` does (K9)."""
        q = np.asarray(q, dtype=float)
        frac = (q - self.lo) / (self.hi - self.lo)
        frac = np.where(self.bounded, np.clip(frac, 0.0, 1.0), frac)
        return self.lo_u + frac * (self.hi_u - self.lo_u)

    def to_rad(self, u: np.ndarray) -> np.ndarray:
        """policy units -> rad, unclamped: an agent run's safety chain refuses a target past the range (K6)."""
        u = np.asarray(u, dtype=float)
        frac = (u - self.lo_u) / (self.hi_u - self.lo_u)
        return self.lo + frac * (self.hi - self.lo)


def declared_units(policy: Any) -> str | None:
    """What a policy says it speaks: ``policy.units`` -> a registered scheme name, or None when it cannot say."""
    u = getattr(policy, "units", None)
    if u is None:
        return None
    if u not in SCHEMES:
        raise ValueError(f"policy declares units {u!r}; known: {', '.join(sorted(SCHEMES))}")
    return str(u)


def resolve_units(policy: Any, requested: str | None, description: Description) -> Units | None:
    """``requested``: "auto" (``policy.units_for(d)``, its declaration, or radians), "rad", a scheme, or None (rad).

    A scheme says what a policy that cannot say speaks. One that declares its units keeps them: a scheme that agrees
    gets the policy's own map, in its norm modes, and any other ("rad" and None too) is refused, naming both.
    """
    if requested == "auto":
        return _own(policy, description)[1]
    want = None if requested is None or requested == RAD else Units.for_robot(description, requested)
    said, own = _own(policy, description)
    if not said:
        return want
    said_as = RAD if own is None else own.name
    if said_as != (RAD if want is None else want.name):
        raise ValueError(
            f"{policy!r} declares units {said_as!r}, but units={requested!r} was passed; drop "
            "units= (the policy says what it speaks)"
        )
    return own


def _own(policy: Any, description: Description) -> tuple[bool, Units | None]:
    """Whether the policy says what it speaks on this robot, and the map it says (None: radians)."""
    units_for = getattr(policy, "units_for", None)
    if callable(units_for):  # a policy whose units depend on the robot (a LeRobot checkpoint's semantics)
        u = units_for(description)
        return True, None if u is None or u.is_identity else u
    name = declared_units(policy)
    if name is None:
        return False, None
    return True, None if name == RAD else Units.for_robot(description, name)
