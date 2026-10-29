"""Adapt a bare ``(obs) -> action`` callable, or anything with act/reset, to a Policy."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from os import PathLike
from typing import Any

from ..types import Action, Observation
from .base import Policy, PolicyLike


class CallablePolicy(Policy):
    """Adapts ``f(obs) -> action`` (optionally ``f(obs, instruction=...)``)."""

    def __init__(self, fn: Callable[..., Any], name: str | None = None) -> None:
        self.fn = fn
        self.name = name or str(getattr(fn, "__name__", fn))  # what an operator reads: the handle, not a repr
        try:
            self._takes_instruction = "instruction" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            self._takes_instruction = False

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        return self.fn(obs, instruction=instruction) if self._takes_instruction else self.fn(obs)

    def __repr__(self) -> str:
        return f"Policy({self.name!r})"


def as_policy(p: Any) -> PolicyLike:
    """Accept a Policy, anything with act/reset, a bare callable, or a policy spec string."""
    if isinstance(p, PolicyLike):
        return p
    if isinstance(p, str | PathLike):  # a policy spec: the factory picks the provider from its form
        from .factory import make

        return as_policy(make(p))
    if callable(p):
        return CallablePolicy(p)
    from .source import SourcePolicy, is_lerobot_source

    if is_lerobot_source(p):  # a LeRobot Teleoperator/Robot drives the run directly (its bus declares the units)
        return SourcePolicy(p)
    raise TypeError(f"not a policy: {type(p).__name__} (need .act(obs) or a callable)")
