"""Action -> full target vector: the one coercion every write path uses (``act``, ``run``, the safety chain)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ..description import Description
from ..types import Action, JointState
from .safety import NonFiniteTarget

if TYPE_CHECKING:
    from .robot import Robot


def clip_to_limits(description: Description, targets: np.ndarray) -> np.ndarray:
    """A copy of ``targets`` with every joint that has limits clipped into them."""
    out = np.array(targets, dtype=float)
    for i, k in enumerate(description.action_keys):
        lim = description.limits.get(k)
        if lim is not None:
            out[i] = min(max(out[i], lim[0]), lim[1])
    return out


def coerce(robot: Robot, action: Action, *, clip: bool = True, measured: np.ndarray | None = None) -> np.ndarray:
    """Any accepted Action -> a full target vector in ``action_keys`` order, clipped to limits.

    A joint a mapping does not name holds its measured position (``measured=``, the observation the caller
    already holds; else one read), clipped into its limits even when ``clip=False``: only the caller's own
    values reach the safety chain unclipped. NaN/inf is refused (``NonFiniteTarget``), and so is an action
    vector or a ``measured`` that is not one value per action key (``ValueError``).
    """
    desc = robot.description
    keys = desc.action_keys
    if isinstance(action, JointState):
        action = action.as_dict()
    if isinstance(action, Mapping):
        unknown = [k for k in action if k not in keys]
        if unknown:
            raise KeyError(f"unknown joint(s) {unknown} for {robot.id}; action_keys are {list(keys)}")
        if measured is None:
            with robot._lock:
                measured = np.array(robot.body.read().positions, dtype=float)
        q = np.asarray(measured, dtype=float).reshape(-1)
        if q.shape != (len(keys),):
            raise ValueError(f"measured has {q.size} values; {robot.id} has {len(keys)} action keys {list(keys)}")
        targets = clip_to_limits(desc, q)
        for k, v in action.items():
            targets[keys.index(k)] = float(v)
    else:
        targets = np.asarray(action, dtype=float).reshape(-1)
        if targets.shape != (len(keys),):
            raise ValueError(f"action has {targets.size} values; {robot.id} has {len(keys)} action keys {list(keys)}")
    if not np.isfinite(targets).all():  # NaN survives min/max: a policy's NaN would reach the bus
        raise NonFiniteTarget(f"{robot.id}: non-finite target {dict(zip(keys, targets.tolist(), strict=True))}")
    return clip_to_limits(desc, targets) if clip else targets
