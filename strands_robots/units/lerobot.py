"""LeRobot's motor normalisation, as a unit scheme.

A LeRobot checkpoint un-normalises to the units its dataset stored. Each joint uses one of LeRobot's
``MotorNormMode`` values: ``range_m100_100`` (calibrated travel onto -100..100), ``range_0_100`` (onto
0..100, grippers) or ``degrees``. The per-joint mode is data: ``hardware.lerobot_norm`` in the registry
is the robot's default (a ``"*"`` entry plus per-key overrides), and a checkpoint's own semantics
override it (``lerobot_units(description, norm)``, see ``units/semantics.json``).

The convention, once: the feetech driver maps a joint's calibrated ``range_min..range_max`` counts
linearly onto the model's joint limits ``lo..hi``, and this scheme maps the same ``lo..hi`` onto
-100..100 (or 0..100). Because both use the same ``lo``/``hi``, counts -> rad -> lerobot equals
LeRobot's own normalisation ``(counts - range_min) / (range_max - range_min) * 200 - 100``.

``degrees`` is LeRobot's angle about the calibrated travel's midpoint. Under the same convention (the
calibrated travel spans the model's ``lo..hi``) that is ``rad2deg(q - (lo + hi) / 2)``: still affine,
so it is one more pair of end points, ``rad2deg(lo - mid)`` and ``rad2deg(hi - mid)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from ..description import Description
from .scheme import Units, register

_RANGE = {"range_m100_100": (-100.0, 100.0), "range_0_100": (0.0, 100.0)}
MODES = (*_RANGE, "degrees")


def lerobot_units(description: Description, norm: Mapping[str, str] | None = None) -> Units:
    """The LeRobot map for this robot; ``norm`` (key or ``"*"`` -> mode) overrides the registry default."""
    norm = norm if norm is not None else description.hardware.lerobot_norm
    if norm is None:
        raise ValueError(
            f"{description.name} declares no hardware.lerobot_norm in the registry; the lerobot units need one"
        )
    keys = description.action_keys
    missing = [k for k in keys if k not in description.limits]
    if missing:
        raise ValueError(f"'lerobot' needs joint limits for every action key; {description.name} lacks {missing}")
    lo = np.array([description.limits[k][0] for k in keys])
    hi = np.array([description.limits[k][1] for k in keys])
    lo_u, hi_u, modes = [], [], []
    for k, a, b in zip(keys, lo, hi, strict=True):
        mode = norm.get(k, norm.get("*"))
        modes.append(str(mode))
        if mode == "degrees":
            mid = (a + b) / 2.0
            lo_u.append(float(np.rad2deg(a - mid)))
            hi_u.append(float(np.rad2deg(b - mid)))
        elif mode in _RANGE:
            lo_u.append(_RANGE[mode][0])
            hi_u.append(_RANGE[mode][1])
        else:
            raise ValueError(f"{description.name}: joint {k!r} has unknown lerobot_norm mode {mode!r}; known: {MODES}")
    return Units("lerobot", keys, lo, hi, np.array(lo_u), np.array(hi_u), tuple(modes))


def _named(description: Description) -> dict[str, str]:
    """The joints the registry gives a mode of their own (``hardware.lerobot_norm`` but ``"*"``): an SO arm's gripper
    is ``range_0_100`` whether LeRobot recorded its body in degrees or in range."""
    return {k: m for k, m in (description.hardware.lerobot_norm or {}).items() if k != "*"}


def unnamed(description: Description, norm: Mapping[str, str]) -> dict[str, str]:
    """The joints ``norm`` leaves to a ``"*"`` other than the mode the registry names for them: joint -> that mode."""
    star, named = norm.get("*"), _named(description)
    left = [k for k in description.action_keys if k not in norm and k in named]
    return {k: named[k] for k in left if star is not None and named[k] != star}


def norm_hint(description: Description) -> str:
    """The ``norm=`` a real body's refusal asks for, whole: each body mode with the joints the registry names."""
    return " or ".join("norm=" + str({"*": body, **_named(description)}) for body in ("degrees", "range_m100_100"))


def calibrated(units: Units, calibration: Any, *, resolution: int) -> Units:
    """Exact hardware ``degrees``: LeRobot's ``(counts - mid) * 360 / (resolution - 1)`` at the calibrated ends.

    ``calibration`` is a bus calibration (``.joints[name]`` with ``range_min``/``range_max``/``drive_mode``),
    whose ``range_min..range_max`` the driver maps onto ``lo..hi``. LeRobot's degrees ignore ``drive_mode``,
    so a reversed joint's end points swap. Range modes are already exact and are left as they are.
    """
    joints = getattr(calibration, "joints", None) or {}
    if "degrees" not in units.modes or not joints:
        return units
    lo_u, hi_u = units.lo_u.copy(), units.hi_u.copy()
    for i, (k, mode) in enumerate(zip(units.keys, units.modes, strict=True)):
        c = joints.get(k)
        if mode == "degrees" and c is not None:
            half = (c.range_max - c.range_min) / 2 * 360 / (resolution - 1)
            lo_u[i], hi_u[i] = (half, -half) if c.drive_mode else (-half, half)
    return replace(units, lo_u=lo_u, hi_u=hi_u)


@register("lerobot")
def lerobot(description: Description) -> Units:
    return lerobot_units(description)
