"""A policy that plays a fixed list of actions (handy for tests and scripted motions)."""

from __future__ import annotations

from collections.abc import Sequence

from ..types import Action, Observation
from .base import Policy


class SequencePolicy(Policy):
    """Plays a fixed list of actions, then holds the last one (handy for tests and scripted motions)."""

    def __init__(self, actions: Sequence[Action]) -> None:
        if not actions:
            raise ValueError("SequencePolicy needs at least one action")
        self.actions = list(actions)
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a
