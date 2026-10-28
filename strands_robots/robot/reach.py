"""``robot.act(action, seconds=s)``: reach a pose, not one step toward it.

One write of a far target leaves a servo arm mid-swing when ``act`` reads back, and a simulated arm one control step
along. With ``seconds``, the targets ramp from the measured pose to ``action`` at ``HZ`` (smoothstep, so it starts and
ends at rest) and the last one is held until ``seconds`` are up: the read-back is the pose reached, gated once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .clock import MONOTONIC

if TYPE_CHECKING:
    from .robot import Robot

HZ = 30.0
RAMP = 0.8  # of ``seconds`` spent moving; the rest settles at the target


def reach(
    robot: Robot, targets: np.ndarray, seconds: float, *, gate: Callable[[np.ndarray], np.ndarray] | None = None
) -> None:
    """``gate``: what every ramp target passes through before the write (an approved agent run's safety chain)."""
    if not seconds > 0:
        raise ValueError(f"seconds must be > 0, got {seconds}")
    start = robot.coerce(robot.observe(images=False).joints)
    ticks = max(1, round(seconds * HZ))
    moving = max(1, round(ticks * RAMP))
    steps = max(1, round(1.0 / HZ / robot.sim.physics.timestep)) if robot.sim is not None else None
    for i in range(1, ticks + 1):
        u = min(1.0, i / moving)
        t = start + (targets - start) * (3 * u * u - 2 * u * u * u)
        robot.command(t if gate is None else gate(t), steps=steps)
        if robot.sim is None:  # hardware: one write per period on the body's clock; an e-stop cuts the wait short
            (robot.estop_latch.wait if robot.clock is MONOTONIC else robot.clock.sleep)(1.0 / HZ)
