"""Bind a policy to a robot (gains, units, requirements, reset), then decide one action per step."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from ...policy import PolicyLike, as_policy
from ...policy.requirements import Requirements
from ...types import JointState, Observation
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


def bind(robot: Robot, policy: PolicyLike | Callable[[Observation], Any], units: str | None) -> Bound:
    """What every loop does before step one: the policy's gains, its units, its requirements, a reset."""
    refuse_remote_spec(policy)  # an agent's run, by any entry point: no policy at a network address
    pol = as_policy(policy)
    for name in ("attach", "detach", "requirements", "needs_observation"):
        _hook(pol, name, callable, "a method")
    if _hook(pol, "requires_position_hold", _flag, "True or False", False) and not robot.supports_position_hold:
        from ..safety import RequestRefused

        raise RequestRefused(f"{robot.id} cannot hold position between ticks, and {pol!r} requires a body that can")
    if units != "auto":  # a named scheme needs no session: an unknown one is refused before attach opens one
        resolve_units(pol, units, robot.description)
    attach = getattr(pol, "attach", None)
    if attach is not None:  # connect / load / admit against THIS robot, before units and requirements
        attach(robot)
    try:
        return _ready(robot, pol, units)
    except BaseException:  # a refusal after attach detaches before it propagates: nothing stays open (K2)
        _detach(pol)
        raise


def _ready(robot: Robot, pol: Any, units: str | None) -> Bound:
    """What needs ``attach`` and can still refuse (rate, gains, units, requirements, reset); checks that need no
    session ran before it in ``bind``, so a refusal there opened nothing."""
    hz = _hook(pol, "hz", _rate, "a number > 0")  # after attach: a remote policy learns its rate from the server
    realtime = _hook(pol, "realtime", _flag, "True or False", False)
    gains = _hook(pol, "gains", _pair, "a (kp, kd) pair")
    if gains is not None and robot.sim is not None:  # the policy's PD is part of the policy, not of the world
        robot.set_gains(*gains)
    conv = resolve_units(pol, units, robot.description)
    cal, res = getattr(robot.body, "calibration", None), getattr(robot.body, "resolution", None)
    if conv is not None and cal is not None and res is not None:  # a calibrated bus: exact hardware degrees
        conv = calibrated(conv, cal, resolution=res)
    req = getattr(pol, "requirements", None)
    if req is not None:
        need = req()
        if not isinstance(need, Requirements):
            raise TypeError(f"{pol!r}.requirements() must return Requirements, got {need!r}")
        cameras = getattr(robot.body, "cameras", None)
        if cameras is not None and need.images:  # a hardware camera that declared no size takes the policy's
            cameras.fit(need.images)
        need.check(robot.observe(), who=repr(pol))
    pol.reset()
    return Bound(pol, conv, None if hz is None else float(hz), bool(realtime))


def _detach(pol: Any) -> None:
    detach = getattr(pol, "detach", None)
    if detach is not None:  # bind() refused a detach that is not a method
        detach()


def release(bound: Bound) -> None:
    """After the last step: let the policy close what ``attach`` opened (a session, a server)."""
    _detach(bound.policy)


def decide(
    robot: Robot, bound: Bound, obs: Observation, instruction: str | None, *, clip: bool = True
) -> np.ndarray | None:
    """One inference in the policy's units, as full targets in the robot's radians. A vector, or a ``{joint:
    value}`` mapping converted per key: joints it leaves out hold their position in ``obs`` (no second read).
    ``clip=False`` (an agent run): an out-of-range target is kept so the safety chain refuses it, instead of being
    silently moved to the limit (K6). ``None`` when the policy returned ``None``: it holds this tick (nothing to
    coerce, gate or write)."""
    pol, conv, q = bound.policy, bound.units, np.asarray(obs.joints.positions, dtype=float)
    if conv is None:
        out = pol.act(obs, instruction=instruction)
        return None if out is None else robot.coerce(out, clip=clip, measured=q)
    seen = obs.with_joints(conv.from_rad(obs.joints.positions))
    out = pol.act(seen, instruction=instruction)
    if out is None:
        return None
    if isinstance(out, JointState):
        out = out.as_dict()
    if isinstance(out, Mapping):  # per key, in policy units; an unnamed joint holds q, an unknown one coerce refuses
        names = list(obs.joints.names)
        u = np.array(seen.joints.positions, dtype=float)
        for k, v in out.items():
            if k in names:
                u[names.index(k)] = v
        rad = conv.to_rad(u)
        return robot.coerce(
            {k: rad[names.index(k)] if k in names else v for k, v in out.items()}, clip=clip, measured=q
        )
    return robot.coerce(conv.to_rad(np.asarray(out, dtype=float)), clip=clip, measured=q)


def looks(bound: Bound, instruction: str | None) -> bool:
    """Whether this step renders the cameras: only when the policy looks (a chunk replays without a frame)."""
    ask = getattr(bound.policy, "needs_observation", None)
    return ask is None or bool(ask(instruction=instruction))
