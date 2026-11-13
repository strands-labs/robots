"""Action -> full target vector: the one coercion every write path uses (``act``, ``run``, the safety chain)."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..description import Description
from ..types import Action, JointState
from .safety import NonFiniteTarget

if TYPE_CHECKING:
    from .robot import Robot


def hardware_limits(description: Description) -> Description:
    """``description`` as a real body drives it: a joint with no range in the model (one that turns without end)
    takes the ``hardware.safe`` bound its registry entry declares as its hard limits, so the range check refuses a
    target past it even with the operator's yes, and ``clip_to_limits`` holds it inside. ``ValueError`` names any
    driven joint with neither: the safety chain would read it as unconstrained."""
    safe = description.hardware.safe or {}
    unbounded = [k for k in description.action_keys if not _bounded(description.limits.get(k))]
    if missing := [k for k in unbounded if k not in safe]:
        raise ValueError(
            f"{description.name}: joint(s) {missing} have no range in the model, so a real robot would accept any "
            'goal for them. Give each a range in the model (limited="true" range="lo hi"), or, for a joint that '
            "turns without end, a hardware.safe bound in its registry entry, which is then its hard limit on "
            "hardware; without the mujoco extra the model is read by an XML walk that reads ranges as MuJoCo does"
        )
    folded = {k: safe[k] for k in unbounded}
    return replace(description, limits={**description.limits, **folded}) if folded else description


def _bounded(lim: tuple[float, float] | None) -> bool:
    """A range a check can hold a joint inside: two finite ends, low below high (``-inf inf`` bounds nothing)."""
    return lim is not None and bool(np.isfinite(lim).all()) and lim[0] < lim[1]


_BOUNDS: dict[int, tuple[weakref.ref[Description], np.ndarray, np.ndarray]] = {}


def _bounds(description: Description) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) per action key, computed once per Description (frozen: ``replace`` makes a new one). A joint
    with no limits, or a NaN end, bounds nothing on that side, as the per-joint ``min``/``max`` it replaces did."""
    hit = _BOUNDS.get(id(description))
    if hit is not None and hit[0]() is description:
        return hit[1], hit[2]
    lims = [description.limits.get(k) or (-np.inf, np.inf) for k in description.action_keys]
    lo = np.nan_to_num(np.array([lim[0] for lim in lims], dtype=float), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    hi = np.nan_to_num(np.array([lim[1] for lim in lims], dtype=float), nan=np.inf, posinf=np.inf, neginf=-np.inf)
    key = id(description)
    _BOUNDS[key] = (weakref.ref(description, lambda _: _BOUNDS.pop(key, None)), lo, hi)
    return lo, hi


def clip_to_limits(description: Description, targets: np.ndarray) -> np.ndarray:
    """A copy of ``targets`` with every joint that has limits clipped into them (a NaN target stays NaN)."""
    lo, hi = _bounds(description)
    return np.minimum(np.maximum(np.array(targets, dtype=float), lo), hi)


def coerce(robot: Robot, action: Action, *, clip: bool = True, measured: np.ndarray | None = None) -> np.ndarray:
    """Any accepted Action -> a full target vector in ``action_keys`` order, clipped to limits.

    A joint a mapping does not name keeps its last commanded target (``robot._held``, set by every write), so
    a partial action holds the others against gravity instead of re-targeting wherever they sagged to. Before
    the first write, and after ``hold()``, an e-stop or ``sim.reset()``, it is the measured position
    (``measured=``, the observation the caller already holds; else one read). Either is clipped into its
    limits even when ``clip=False``: only the caller's own values reach the safety chain unclipped. NaN/inf is
    refused (``NonFiniteTarget``), and so is an action vector or a ``measured`` that is not one value per action
    key (``ValueError``).
    """
    desc = robot.description
    keys = desc.action_keys
    if isinstance(action, JointState):
        action = action.as_dict()
    if isinstance(action, Mapping):
        unknown = [k for k in action if k not in keys]
        if unknown:
            if any(k.endswith(".pos") for k in unknown):  # a LeRobot-format dict: its units are not act()'s to guess
                raise KeyError(
                    f"{robot.id}: {unknown} are LeRobot '<joint>.pos' names in the policy's units; act() takes a "
                    f"'<joint>' target in radians. Pass a LeRobot Teleoperator/Robot to run(), or drop the '.pos' "
                    f"suffix and the units. action_keys are {list(keys)}"
                )
            raise KeyError(f"unknown joint(s) {unknown} for {robot.id}; action_keys are {list(keys)}")
        base = robot._held
        if base is None:
            if measured is None:
                with robot._lock:
                    measured = np.array(robot.body.read().positions, dtype=float)
            base = measured
        q = np.asarray(base, dtype=float).reshape(-1)
        if q.shape != (len(keys),):
            raise ValueError(f"measured has {q.size} values; {robot.id} has {len(keys)} action keys {list(keys)}")
        targets = clip_to_limits(desc, q) if not clip else q.copy()  # clip=True: the final clip covers the base
        for k, v in action.items():
            targets[keys.index(k)] = float(v)
    else:
        targets = np.asarray(action, dtype=float).reshape(-1)
        if targets.shape != (len(keys),):
            raise ValueError(f"action has {targets.size} values; {robot.id} has {len(keys)} action keys {list(keys)}")
    if not np.isfinite(targets).all():  # NaN survives min/max: a policy's NaN would reach the bus
        raise NonFiniteTarget(f"{robot.id}: non-finite target {dict(zip(keys, targets.tolist(), strict=True))}")
    return clip_to_limits(desc, targets) if clip else targets
