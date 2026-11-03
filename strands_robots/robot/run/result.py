"""What a run measured."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...types import Observation

StoppedBy = Literal["steps", "seconds", "stop_when", "stopped", "fault", "estop", "deadline"]


@dataclass(frozen=True)
class RunResult:
    """What ``Robot.run`` measured: how many policy steps ran, over how long, and why it stopped.

    ``stopped_by`` names the limit that ended it: ``"seconds"`` / ``"steps"`` for the one the caller gave.
    ``seconds`` is on the body's clock: in a world, the sim time its steps advanced; on hardware, the commanded
    span (first write to one period after the last, a write held to its period counted on it), so with
    ``deadline_misses == 0`` it is ``steps / hz_target`` to within one period, however late the host's sleeps
    wake, plus the time a hold waited past its period. A miss is a paced write that landed after the next tick
    should have started (re-phased, never caught up); ``worst_late_s`` is the worst. ``wall_seconds`` is the
    host's time for the run.
    A tick whose policy returned ``None`` is a ``hold``: the body kept its last target, nothing was written, the grid
    moved on — a policy waiting for its server (``execution="sequential"``) holds on purpose, and that is not a miss.
    """

    steps: int
    seconds: float
    hz_target: float
    hz_achieved: float
    stopped_by: StoppedBy
    recorded: str | None = None
    final: Observation | None = None
    fault: str | None = None  # why the run could not go on (stopped_by == "fault" | "deadline")
    deadline_misses: int = 0
    worst_late_s: float = 0.0
    wall_seconds: float = 0.0
    holds: int = 0  # ticks the policy chose to hold (``act`` returned None): no write, never a miss
    held_seconds: float = 0.0  # those ticks' share of the commanded span (holds / hz_target)
