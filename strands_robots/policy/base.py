"""A Policy turns an Observation into an Action. That is the whole contract."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..types import Action, Observation
from ..units import SCHEMES
from .requirements import Requirements


@runtime_checkable
class PolicyLike(Protocol):
    def act(self, obs: Observation, *, instruction: str | None = None) -> Action: ...

    def reset(self) -> None: ...


class PolicyFault(RuntimeError):
    """A policy can no longer produce safe actions (deadline, stale data, server lost): the run ends here."""


class _Factory(type):
    """``Policy("lerobot/smolvla_base")`` builds a provider; ``SomePolicy(...)`` is a plain constructor."""

    def __call__(cls, *args: Any, **kw: Any) -> Any:
        if cls is Policy and (args or kw):
            from .factory import make

            return make(*args, **kw)
        return super().__call__(*args, **kw)


class Policy(metaclass=_Factory):
    """Base class: override ``act``; ``units`` and ``requirements`` say what the policy speaks and needs.

    Optional hooks the run loop honours: ``attach(robot)`` before step one (connect, admit, load),
    ``detach()`` after the last step, ``realtime`` (pace in wall time even in a sim), ``hz`` and
    ``needs_observation`` (the loop renders the cameras only for a step that answers True), ``gains``
    ``(kp, kd)``, ``start_pose`` (where its episodes begin, in the policy's units — a mapping or a full
    vector — reached over ``start_seconds``, default 3, before step one). The loop reads them once before
    step one; a hook of the wrong type is a ``TypeError``.
    """

    units: str | None = None  # a scheme name in strands_robots.units (e.g. "rad", "lerobot") or None = cannot say

    def __init_subclass__(cls, **kw: object) -> None:
        super().__init_subclass__(**kw)
        declared = cls.__dict__.get("units")
        if declared is not None and declared not in SCHEMES:
            raise ValueError(f"policy {cls.__name__} declares units {declared!r}; known: {', '.join(sorted(SCHEMES))}")

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def needs_observation(self, *, instruction: str | None = None) -> bool:
        """Whether the next ``act`` looks at the observation's images; the default looks on every step."""
        return True

    def requirements(self) -> Requirements:
        """Cameras/state this policy needs; the default needs nothing specific."""
        return Requirements()
