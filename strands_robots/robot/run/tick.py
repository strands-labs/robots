"""THE step, shared by ``run`` and ``lockstep``: one read, one decision, one write (``decided``, then ``commit``)."""

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
    """What one tick did: the observation it acted on and the targets it decided (``commit`` writes them), or why it
    ended the run."""

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
    stops0: int,
    stop_when: Callable[[Observation], bool] | None = None,
    steps: int | None = None,
    before_write: Callable[[], None] | None = None,
) -> Tick:
    """observe (the only body read) -> stop_when -> policy (joints it leaves out hold that read) -> write.

    A policy that returns ``None`` holds: no gate, no write (``Tick.held``); a world steps its period anyway.

    ``gated`` (an agent's run) passes the targets through the safety chain and the operator gate first; under
    ``from_agent(confirmed=True)`` the operator step is pre-satisfied, every other step still runs.
    ``stops0`` is the e-stop latch's count when the loop began (after any ``start_pose`` ramp): a press since then,
    even one resumed, ends the tick before its write.
    """
    step = decided(robot, bound, instruction, images=images, gated=gated, stop_when=stop_when)
    if step.ended is not None:
        return step
    return commit(robot, step, gated=gated, stops0=stops0, steps=steps, before_write=before_write)


def decided(
    robot: Robot,
    bound: Bound,
    instruction: str | None,
    *,
    images: bool,
    gated: bool,
    stop_when: Callable[[Observation], bool] | None = None,
) -> Tick:
    """The tick up to its write (``commit``): lockstep decides every robot's tick before any robot writes."""
    obs = robot.observe(images=images)
    if stop_when is not None and stop_when(obs):
        return Tick(obs, ended="stop_when")
    try:
        targets = decide(
            robot, bound, obs, instruction, clip=not gated
        )  # an agent run is refused out of range, not clipped
        if targets is None:  # a hold: nothing to gate or write; ``commit`` still advances a world its period
            return Tick(obs, held=True)
        if gated and is_confirmed():  # a dashboard run: the operator said yes once, for these targets
            targets = robot.gate(targets, confirmed=True)
        elif gated:  # an agent run the operator approved as a whole (K6): the chain every step, no per-tick ask
            targets = robot.gate(targets, approved_run=True, measured=obs.joints.positions)
    except (RequestRefused, PolicyFault, NonFiniteTarget) as e:
        if (ended := _ended(obs, e, gated)) is None:
            raise  # a dashboard run (confirmed once) re-raises loudly, so the page sees the refusal as an error
        return ended
    return Tick(obs, targets)


def commit(
    robot: Robot,
    step: Tick,
    *,
    gated: bool,
    stops0: int,
    steps: int | None = None,
    before_write: Callable[[], None] | None = None,
) -> Tick:
    """A decided tick's one write; a hold writes nothing and a world still advances its period. ``steps=None`` in
    lockstep: the world steps once for everyone after every robot wrote. A press since ``stops0`` (see ``tick``),
    even one resumed, ends the tick unwritten."""
    if step.held:  # the body keeps its last target; simulated actuators hold it through the period
        idle(robot, steps)
        return step
    assert step.targets is not None, "commit writes a decided tick"
    try:
        if before_write is not None:  # a paced body: at least one period since the previous write
            before_write()
        # guarded_write refuses it, under the bus lock, once the latch has moved since stops0
        robot.command(step.targets, steps=steps, stops0=stops0)
    except (RequestRefused, PolicyFault, NonFiniteTarget) as e:
        if (ended := _ended(step.obs, e, gated)) is None:
            raise
        return ended
    return step


def _ended(obs: Observation, e: Exception, gated: bool) -> Tick | None:
    """How a tick ends on what its decision or its write raised; None: the caller raises it again."""
    if isinstance(e, EStopped):
        return Tick(obs, ended="estop")
    if isinstance(e, PolicyFault | NonFiniteTarget):
        return Tick(obs, ended="fault", fault=str(e))
    if gated and not is_confirmed():  # an approved agent run: a refused step ends the run as a fault (K6)
        return Tick(obs, ended="fault", fault=str(e))
    return None


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
