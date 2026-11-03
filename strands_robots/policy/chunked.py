"""Hold an action chunk and replay it, one action per control step."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..types import Action, Observation
from .base import Policy


class ChunkedPolicy(Policy):
    """Wraps a policy with ``act_chunk(obs, instruction=) -> [H, n]``; asks again when the chunk runs out.

    ``replan_every`` (steps) asks earlier; ``None`` plays the whole chunk. A new instruction drops the
    rest of the chunk. Lives here, never in the sim or the run loop. Every other attribute (``units``, ``attach``,
    ``units_for``, ``requirements``, ``hz`` ...) is the inner policy's.
    """

    def __init__(self, inner: Any, *, replan_every: int | None = None) -> None:
        if not callable(getattr(inner, "act_chunk", None)):
            raise TypeError(f"{type(inner).__name__} has no act_chunk(obs, instruction=) -> [H, n]")
        if replan_every is not None and replan_every < 1:
            raise ValueError(f"replan_every must be >= 1, got {replan_every}")
        self.inner = inner
        self.units = getattr(inner, "units", None)  # Policy.units on the class would hide the inner's
        self.replan_every = replan_every
        self._queue: deque[np.ndarray] = deque()
        self._played = 0
        self._instruction: str | None = None

    def __getattr__(self, name: str) -> Any:
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def requirements(self) -> Any:
        req = getattr(self.inner, "requirements", None)
        return req() if callable(req) else super().requirements()

    def reset(self) -> None:
        self._queue.clear()
        self._played = 0
        self.inner.reset()

    def needs_observation(self, *, instruction: str | None = None) -> bool:
        """Only the step that asks the inner policy looks; the queued steps replay without a frame."""
        stale = self.replan_every is not None and self._played >= self.replan_every
        return not self._queue or stale or instruction != self._instruction

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        if self.needs_observation(instruction=instruction):
            chunk = np.asarray(self.inner.act_chunk(obs, instruction=instruction), dtype=float)
            if chunk.ndim != 2 or len(chunk) == 0:
                raise ValueError(f"act_chunk returned shape {chunk.shape}; expected [H, n] with H >= 1")
            self._queue = deque(chunk)
            self._played = 0
            self._instruction = instruction
        self._played += 1
        return self._queue.popleft()

    def __repr__(self) -> str:
        return f"ChunkedPolicy({self.inner!r})"
