"""What the operator is asked: a ``Move`` (one command) or a ``RunPlan`` (one agent run), through one hook.

``Robot(..., operator=callable(Move | RunPlan) -> bool)`` is the operator's yes for a real body. A command from an
agent tool, the dashboard, a mesh peer or a ROS topic asks it with a ``Move`` naming exactly the targets it would
write; an agent's run asks it once with its ``RunPlan`` before anything is acquired. ``None`` refuses both.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from ..run.source import RunPlan

HINT = "pass operator=<callable(Move | RunPlan) -> bool> to Robot(...)"
OLD = ("confirm", "approve")


@dataclass(frozen=True, eq=False)
class Move:
    """One command on its way to a real body: ``targets`` by joint in the robot's units (``action_keys`` order),
    ``vector`` the same values as written, ``source`` who sent it (``agent``, ``dashboard``, ``mesh``, ``ros``),
    ``seconds`` the ramp it reaches them over (None: at once), ``reason`` why it is asked beyond the plain gate."""

    robot: str
    targets: dict[str, float]
    vector: np.ndarray = field(repr=False)
    source: str = "agent"
    seconds: float | None = None
    reason: str | None = None

    @classmethod
    def of(cls, robot: str, keys: Sequence[str], vector: Any, **kw: Any) -> Move:
        v = np.array(vector, dtype=float)
        v.setflags(write=False)  # the operator reads it; the chain writes its own copy
        return cls(robot, {k: float(x) for k, x in zip(keys, v, strict=True)}, v, **kw)

    def __str__(self) -> str:
        joints = ", ".join(f"{k}={v:.3f}" for k, v in self.targets.items())
        ramp = "" if self.seconds is None else f" over {self.seconds:g} s"
        why = "" if self.reason is None else f" ({self.reason})"
        return f"move {self.robot} from {self.source}{ramp}: {joints}{why}"


Operator: TypeAlias = "Callable[[Move | RunPlan], bool]"


def refuse_old_names(where: str, given: dict[str, Any]) -> None:
    """``confirm=`` / ``approve=`` were two hooks for one question; name the one that replaced them."""
    old = [n for n in OLD if n in given]
    if old:
        raise TypeError(
            f"{where} takes no {', '.join(o + '=' for o in old)}: use operator=<callable(Move | RunPlan) "
            "-> bool>, asked with a Move for one command and a RunPlan for one agent run"
        )
