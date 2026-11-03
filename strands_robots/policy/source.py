"""A LeRobot Teleoperator (or Robot) as the source of a run: ``arm.run(leader, hz=30, seconds=60)`` (D1/D2).

The source declares its own units: each motor's ``MotorNormMode`` on its bus (``leader.bus.motors[k].norm_mode``).
Its ``get_action()`` (a Teleoperator) or ``get_observation()`` (a Robot being mirrored) is a ``{"<key>.pos": v}``
dict in those units; this adapter orders it by the follower's action keys and declares the exact unit map, so the
run loop converts it with the follower's own calibration -- the same arithmetic LeRobot's teleoperate does (the leader
normalises with its calibration, the follower un-normalises with its own). Nothing is guessed: a source without a bus
that declares norm modes is refused with the fix.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import Action, Observation
from ..units import Units, lerobot_units

_MODES = {"RANGE_M100_100": "range_m100_100", "RANGE_0_100": "range_0_100", "DEGREES": "degrees"}


def _pos(key: str) -> str:
    """LeRobot's one feature-naming rule for a joint: ``"<action key>.pos"``."""
    return f"{key}.pos"


def is_lerobot_source(obj: Any) -> bool:
    """A LeRobot Teleoperator (``get_action``) or Robot (``get_observation``) with a motor bus."""
    return hasattr(getattr(obj, "bus", None), "motors") and (
        callable(getattr(obj, "get_action", None)) or callable(getattr(obj, "get_observation", None))
    )


def norm_of(source: Any) -> dict[str, str]:
    """The source's per-motor norm modes, as strands unit-scheme names."""
    motors = getattr(getattr(source, "bus", None), "motors", None)
    if not motors:
        raise TypeError(
            f"{type(source).__name__} declares no motor norm modes (no .bus.motors): its action's units are unknown; "
            "pass units= to run(), or a LeRobot Teleoperator/Robot"
        )
    out = {}
    for k, m in motors.items():
        name = getattr(m.norm_mode, "name", str(m.norm_mode))
        if name not in _MODES:
            raise ValueError(f"{type(source).__name__}.{k}: norm mode {name!r} has no strands unit scheme")
        out[k] = _MODES[name]
    return out


class SourcePolicy:
    """Adapts a LeRobot Teleoperator/Robot to the run loop (``as_policy`` does this for you)."""

    realtime = True  # a human at a leader arm: a watched world runs in wall time too

    def __init__(self, source: Any) -> None:
        self.source = source
        self.norm = norm_of(source)
        self._read = getattr(source, "get_action", None) or source.get_observation
        self._keys: tuple[str, ...] = ()

    def units_for(self, description: Any) -> Units:
        missing = [k for k in description.action_keys if k not in self.norm]
        if missing:
            raise ValueError(f"{type(self.source).__name__} has no motor(s) {missing} for {description.name}")
        self._keys = tuple(description.action_keys)
        return lerobot_units(description, self.norm)

    def needs_observation(self, *, instruction: str | None = None) -> bool:
        return False  # the leader is the input; the follower's cameras are not needed to decide

    def reset(self) -> None:
        pass

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        a = self._read()
        keys = self._keys or tuple(k[: -len(".pos")] for k in a if k.endswith(".pos"))
        missing = [_pos(k) for k in keys if _pos(k) not in a]
        if missing:
            raise KeyError(f"{type(self.source).__name__} did not report {missing}; it reported {sorted(a)}")
        return np.array([float(a[_pos(k)]) for k in keys])

    def __repr__(self) -> str:
        return f"SourcePolicy({type(self.source).__name__}({getattr(self.source, 'id', '')!r}))"
