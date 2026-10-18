"""Bind a policy to a robot (units, requirements, reset; its gains go on last), then decide one action per step."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from ...policy import PolicyLike, as_policy
from ...policy.requirements import Requirements
from ...types import Observation
from ...units import Units, calibrated, resolve_units
from .source import refuse_remote_spec

if TYPE_CHECKING:
    from ..robot import Robot


@dataclass(frozen=True)
class Bound:
    """A policy made ready for a robot: the adapted policy, the unit converter (or None for radians), its hooks."""

    policy: PolicyLike
    units: Units | None
    hz: float | None = None
    realtime: bool = False
    gains: tuple[Any, Any] | None = None  # the policy's (kp, kd) for a body in a world; install_gains puts them on


def _hook(pol: Any, name: str, ok: Callable[[Any], bool], want: str, default: Any = None) -> Any:
    """One optional policy attribute, read once: absent is the default, a wrong type is refused by name."""
    value = getattr(pol, name, default)
    if value is not default and not ok(value):
        raise TypeError(f"{pol!r}.{name} must be {want}, got {value!r}")
    return value


def _rate(v: Any) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0


def _flag(v: Any) -> bool:
    return isinstance(v, bool | np.bool_)


def _pair(v: Any) -> bool:
    return isinstance(v, tuple | list) and len(v) == 2


def _gain(v: Any) -> bool:
    g = np.asarray(v, dtype=float)
    return bool(np.isfinite(g).all() and (g >= 0).all())


def bind(robot: Robot, policy: PolicyLike | Callable[[Observation], Any], units: str | None) -> Bound:
    """What every loop does before step one: the policy's units, its requirements, a check of its gains, a reset.

    The policy's gains are only carried in the ``Bound``: ``install_gains`` puts them on after the loop's last checks.
    """
    refuse_remote_spec(policy)  # an agent's run, by any entry point: no policy at a network address
    pol = as_policy(policy)
    for name in ("attach", "detach", "requirements", "needs_observation"):
        _hook(pol, name, callable, "a method")
    if _hook(pol, "requires_position_hold", _flag, "True or False", False) and not robot.supports_position_hold:
        from ..safety import RequestRefused

        raise RequestRefused(f"{robot.id} cannot hold position between ticks, and {pol!r} requires a body that can")
    attach = getattr(pol, "attach", None)
    if attach is not None:  # connect / load / admit against THIS robot, before units and requirements
        attach(robot)
    hz = _hook(pol, "hz", _rate, "a number > 0")  # after attach: a remote policy learns its rate from the server
    realtime = _hook(pol, "realtime", _flag, "True or False", False)
    gains = _hook(pol, "gains", _pair, "a (kp, kd) pair")  # carried in the Bound: install_gains puts them on last
    conv = resolve_units(pol, units, robot.description)
    cal, res = getattr(robot.body, "calibration", None), getattr(robot.body, "resolution", None)
    if conv is not None and cal is not None and res is not None:  # a calibrated bus: exact hardware degrees
        conv = calibrated(conv, cal, resolution=res)
    req = getattr(pol, "requirements", None)
    if req is not None:
        need = req()
        if not isinstance(need, Requirements):
            raise TypeError(f"{pol!r}.requirements() must return Requirements, got {need!r}")
        need.check(robot.observe(), who=repr(pol))
    n = len(robot.action_keys)
    if robot.sim is None:  # a servo bus keeps its own gains: the policy's are for a body in a world
        gains = None
    elif gains is not None and any(np.size(g) != n for g in gains):  # refused here: in lock-step, before any go on
        raise ValueError(f"{robot.id}: {pol!r} carries kp/kd for {[np.size(g) for g in gains]} actuators, not {n}")
    elif gains is not None and not all(map(_gain, gains)):  # as set_gains would, but before a partner's go on
        raise ValueError(f"{robot.id}: {pol!r} carries kp/kd that are not all finite and >= 0: {gains!r}")
    pol.reset()
    return Bound(pol, conv, None if hz is None else float(hz), bool(realtime), gains)


def install_gains(bound: Mapping[Robot, Bound]) -> list[Robot]:
    """Put each policy's gains on its body, after the loop's last checks (the rate, the step count), before step one.

    A latched e-stop on any of these robots ends the run at step 0, so nothing is installed: an e-stopped body keeps
    the gains it holds with. Returns those robots, and the loop ends the run on them: one read of the latches decides
    both, so a ``resume()`` landing before step one cannot start the run without its policy's gains.
    """
    held = [r for r in bound if r.estopped]
    if not held:
        for robot, b in bound.items():
            if b.gains is not None:  # the policy's PD is part of the policy, not of the world
                robot.set_gains(*b.gains)
    return held


def release(bound: Bound) -> None:
    """After the last step: let the policy close what ``attach`` opened (a session, a server)."""
    detach = getattr(bound.policy, "detach", None)
    if detach is not None:  # bind() refused a detach that is not a method
        detach()


def decide(robot: Robot, bound: Bound, obs: Observation, instruction: str | None) -> Any:
    """One inference in the policy's units, returned as an action in the robot's radians."""
    pol, conv = bound.policy, bound.units
    if conv is None:
        return robot.coerce(pol.act(obs, instruction=instruction))
    seen = obs.with_joints(conv.from_rad(obs.joints.positions))
    return robot.coerce(conv.to_rad(np.asarray(pol.act(seen, instruction=instruction), dtype=float)))


def looks(bound: Bound, instruction: str | None) -> bool:
    """Whether this step renders the cameras: only when the policy looks (a chunk replays without a frame)."""
    ask = getattr(bound.policy, "needs_observation", None)
    return ask is None or bool(ask(instruction=instruction))
