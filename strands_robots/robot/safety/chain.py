"""The safety chain every control write from an agent or a dashboard passes: range -> safe -> interlocks
-> validators -> operator gate. The developer's own ``robot.act`` skips it (they are the operator); the
e-stop bypasses it entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...types import Action
from .validators import JointLimitValidator, Request, RequestRefused, Validator, outside_safe

if TYPE_CHECKING:
    from ..robot import Robot


class SafetyChain:
    """``check(robot, action)`` returns the targets to write, or raises ``RequestRefused``."""

    def __init__(self, validators: tuple[Validator, ...] = ()) -> None:
        self.validators = validators

    def check(
        self,
        robot: Robot,
        action: Action,
        *,
        confirmed: bool = False,
        source: str = "agent",
        approved_run: bool = False,
        measured: np.ndarray | None = None,
    ) -> np.ndarray:
        """``approved_run``: a step of an agent run the operator approved as a whole (``RunPlan``, K6): the operator
        gate is not asked again, and a target outside ``safe`` is REFUSED instead of asked per tick."""
        from ..gate import operator_gate

        robot.estop_latch.check(robot.id)
        targets = robot.coerce(action, clip=False, measured=measured)
        request = Request(robot.action_keys, targets, confirmed, source, measured=measured)
        try:
            JointLimitValidator()(robot, request)
        except RequestRefused:
            # A joint can read a hair past a soft/calibration limit; re-grade against where the body
            # actually sits so a hold (or a return toward range) passes and only driving further out is
            # refused. ``measured`` is read once, here, only when the plain check would refuse.
            if measured is not None:
                raise
            with robot._lock:
                measured = np.array(robot.body.read().positions, dtype=float)
            request = Request(robot.action_keys, targets, confirmed, source, measured=measured)
            JointLimitValidator()(robot, request)
        soft = outside_safe(robot, request)
        if soft and approved_run:
            raise RequestRefused(f"{robot.id}: {'; '.join(soft)}: outside the approved run's safe bounds")
        if approved_run:
            confirmed = True  # the operator approved this run once, for its whole plan
        if soft and not confirmed:
            if robot.confirm is None or not robot.confirm(request.targets):
                raise RequestRefused(f"{robot.id}: {'; '.join(soft)}: needs the operator's yes (confirm=)")
            confirmed = True  # the operator was asked once, for exactly these targets
        for v in (*robot.interlocks, *self.validators, *robot.validators):
            v(robot, request)
        operator_gate(robot, request.targets, confirmed=confirmed)
        return request.targets
