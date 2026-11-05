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
    """``act`` through the operator gate: what an agent tool, the dashboard and a mesh peer call.

    The operator is asked once, for the target. A ``seconds`` ramp then passes every point it writes through the
    rest of the chain again (interlocks, validators, the workspace box): a joint-space line between two allowed
    poses can leave a Cartesian box, and an interlock reading a door or a light curtain can change mid-move. A
    refused point is never written; the ramp stops there and the body holds the last pose it reached.
    """
    targets = gate(robot, action, confirmed=confirmed)
    if seconds is None:
        return robot.act(targets, steps=steps)
    if steps is not None:
        raise ValueError("act takes steps= or seconds=, not both")
    from .reach import reach

    q = np.asarray(robot.observe(images=False).joints.positions, dtype=float)  # raw: a joint past its limit may hold
    reach(robot, targets, seconds, gate=lambda t: gate(robot, t, confirmed=True, measured=q))
    return robot.observe(images=False)
