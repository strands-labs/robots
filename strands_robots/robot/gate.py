"""The operator gate: a real body never moves because of a default.

A command from the developer's own code acts directly. A command from an agent tool or a dashboard
goes through ``gate``: a simulated body passes, a real body passes only when an operator already said
yes or the ``Robot(..., confirm=callable)`` answers True for these targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..types import Action, Observation

if TYPE_CHECKING:
    from .robot import Robot


def gate(
    robot: Robot,
    action: Action,
    *,
    confirmed: bool = False,
    approved_run: bool = False,
    measured: np.ndarray | None = None,
) -> np.ndarray:
    """The whole safety chain for ``action``; the operator rule below is its last step.

    ``approved_run``: a step of an agent run the operator already approved as a whole (``RunPlan``, K6).
    """
    from .safety import SafetyChain

    return SafetyChain().check(robot, action, confirmed=confirmed, approved_run=approved_run, measured=measured)


def operator_gate(robot: Robot, targets: np.ndarray, *, confirmed: bool = False) -> None:
    """A real body needs the operator's yes; raise ``RequestRefused`` (a ``PermissionError``) otherwise."""
    from .safety import RequestRefused

    if not robot.is_real or confirmed:
        return
    if robot.confirm is None:
        raise RequestRefused(
            f"{robot.id} is a real robot and this command came through an agent or dashboard: pass "
            f"confirm=<callable(action) -> bool> to Robot(...) to allow it"
        )
    if not robot.confirm(targets):
        raise RequestRefused(f"{robot.id}: the operator gate refused this action")


def confirm_act(
    robot: Robot, action: Action, *, steps: int | None = None, seconds: float | None = None, confirmed: bool = False
) -> Observation:
    """``act`` through the operator gate: what an agent tool and the dashboard call.

    The gate is asked once, for the target; a ``seconds`` ramp from the measured pose stays inside the box of the
    two, so every point of it passes the limits the target passed.
    """
    return robot.act(gate(robot, action, confirmed=confirmed), steps=steps, seconds=seconds)
