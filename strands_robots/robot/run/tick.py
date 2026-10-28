"""THE step, shared by ``run`` and ``lockstep``: one read, one decision, one write."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ...policy.base import PolicyFault
from ...types import Observation
from ..safety import EStopped, NonFiniteTarget, RequestRefused
from .bind import Bound, decide
from .source import is_confirmed

if TYPE_CHECKING:
    from ..robot import Robot


@dataclass
class Tick:
    """What one tick did: the observation it acted on and the targets it wrote, or why it ended the run."""

    obs: Observation
    targets: np.ndarray | None = None
    ended: Literal["stop_when", "fault", "estop"] | None = None
    fault: str | None = None
    held: bool = False  # the policy returned None: the body kept its last target, nothing was written


def tick(
    robot: Robot,
    bound: Bound,
    instruction: str | None,
    *,
    images: bool,
    gated: bool,
    stop_when: Callable[[Observation], bool] | None = None,
    steps: int | None = None,
    before_write: Callable[[], None] | None = None,
) -> Tick:
    """observe (the only body read) -> stop_when -> policy (joints it leaves out hold that read) -> write.

    A policy that returns ``None`` holds: no gate, no write (``Tick.held``); a world steps its period anyway.

    ``gated`` (an agent's run) passes the targets through the safety chain and the operator gate first; under
    ``from_agent(confirmed=True)`` the operator step is pre-satisfied, every other step still runs.
    ``steps=None`` in lockstep: the world steps once for everyone after every robot wrote.
    """
    obs = robot.observe(images=images)
    if stop_when is not None and stop_when(obs):
        return Tick(obs, ended="stop_when")
    try:
        targets = decide(
            robot, bound, obs, instruction, clip=not gated
        )  # an agent run is refused out of range, not clipped
        if targets is None:  # a hold: the body keeps its last target; a world still advances its period
            idle(robot, steps)
            return Tick(obs, held=True)
        if gated and is_confirmed():  # a dashboard run: the operator said yes once, for these targets
            targets = robot.gate(targets, confirmed=True)
        elif gated:  # an agent run the operator approved as a whole (K6): the chain every step, no per-tick ask
            targets = robot.gate(targets, approved_run=True, measured=obs.joints.positions)
        if before_write is not None:  # a paced body: at least one period since the previous write
            before_write()
        robot.command(targets, steps=steps)
    except EStopped:
        return Tick(obs, ended="estop")
    except (PolicyFault, NonFiniteTarget) as e:
        return Tick(obs, ended="fault", fault=str(e))
    except RequestRefused as e:
        if gated and not is_confirmed():  # an approved agent run: a refused step ends the run as a fault (K6)
            return Tick(obs, ended="fault", fault=str(e))
        raise  # a dashboard run (confirmed once) re-raises loudly, so the page sees the refusal as an error
    return Tick(obs, targets)


def idle(robot: Robot, steps: int | None) -> None:
    """Advance the world one period with NO write (a hold): simulated actuators keep their last target."""
    if robot.sim is not None:
        with robot._lock:
            if steps is None:
                robot.sim.control_step()
            else:
                robot.sim.step(steps)
    elif robot.twin is not None and robot.twin.sim is not None:
        robot.twin.sim.control_step()


def stopped(robot: Robot, stops0: int) -> bool:
    """An e-stop is engaged, or was engaged (and maybe resumed) since the run began: it outranks ``stop()``."""
    return robot.estopped or robot.estop_latch.count != stops0
