"""The operator gate: a real body never moves because of a default.

A command from the developer's own code acts directly. A command from an agent tool or a dashboard
goes through ``gate``: a simulated body passes, a real body passes only when an operator already said
yes or the ``Robot(..., operator=callable)`` answers True for the ``Move`` naming these targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..types import Action, Observation

if TYPE_CHECKING:
    from collections.abc import Callable

    from .robot import Robot


def gate(
    robot: Robot,
    action: Action,
    *,
    confirmed: bool = False,
    approved_run: bool = False,
    measured: np.ndarray | None = None,
    source: str = "agent",
    seconds: float | None = None,
) -> np.ndarray:
    """The whole safety chain for ``action``; the operator rule below is its last step.

    ``approved_run``: a step of an agent run the operator already approved as a whole (``RunPlan``, K6).
    ``source`` and ``seconds`` are what the operator reads on the ``Move`` it is asked with.
    """
    from .safety import SafetyChain

    return SafetyChain().check(
        robot, action, confirmed=confirmed, approved_run=approved_run, measured=measured, source=source, seconds=seconds
    )


def operator_gate(
    robot: Robot, targets: np.ndarray, *, confirmed: bool = False, source: str = "agent", seconds: float | None = None
) -> None:
    """A real body needs the operator's yes; raise ``RequestRefused`` (a ``PermissionError``) otherwise."""
    if not robot.is_real or confirmed:
        return
    ask(robot, targets, source=source, seconds=seconds)


def ask(robot: Robot, targets: np.ndarray, *, source: str, seconds: float | None, reason: str | None = None) -> None:
    """Ask ``robot.operator`` with the ``Move`` for ``targets``; ``RequestRefused`` if there is none or it says no."""
    from .safety import RequestRefused
    from .safety.move import HINT, Move

    if robot.operator is None:
        if reason is not None:
            raise RequestRefused(f"{robot.id}: {reason}: needs the operator's yes: {HINT}")
        raise RequestRefused(
            f"{robot.id} is a real robot and this command came from an agent, the dashboard or the mesh: {HINT}"
        )
    move = Move.of(robot.id, robot.action_keys, targets, source=source, seconds=seconds, reason=reason)
    if not robot.operator(move):
        raise RequestRefused(f"{robot.id}: the operator gate refused {move}")


def confirm_act(
    robot: Robot,
    action: Action,
    *,
    steps: int | None = None,
    seconds: float | None = None,
    confirmed: bool = False,
    source: str = "agent",
    ramp: Callable[..., None],
) -> Observation:
    """``act`` through the operator gate: what an agent tool, the dashboard and a mesh peer call.

    The operator is asked once, for the target. A ``seconds`` ramp then passes every point it writes through the
    rest of the chain again (interlocks, validators, the workspace box): a joint-space line between two allowed
    poses can leave a Cartesian box, and an interlock reading a door or a light curtain can change mid-move. A
    refused point is never written; the ramp stops there and the body holds the last pose it reached. ``ramp`` is
    ``reach.reach``, passed in by ``act_through_gate`` (a test can pass its own).
    """
    if seconds is not None and steps is not None:
        raise ValueError("act takes steps= or seconds=, not both")
    targets = gate(robot, action, confirmed=confirmed, source=source, seconds=seconds)
    if seconds is None:
        return robot.act(targets, steps=steps)
    q = np.asarray(robot.observe(images=False).joints.positions, dtype=float)  # raw: a joint past its limit may hold
    ramp(robot, targets, seconds, gate=lambda t: gate(robot, t, confirmed=True, measured=q, source=source))
    return robot.observe(images=False)


def act_through_gate(
    robot: Robot,
    action: Action,
    *,
    steps: int | None = None,
    seconds: float | None = None,
    confirmed: bool = False,
    source: str = "agent",
) -> Observation:
    """``Robot.confirm_act``: ``act`` through the operator gate, asked once with the target's ``Move``. What an agent
    tool, the dashboard, a mesh peer and a ROS topic call; ``source`` is which, as the operator reads it."""
    from .reach import reach

    return confirm_act(robot, action, steps=steps, seconds=seconds, confirmed=confirmed, source=source, ramp=reach)
