"""Shared motion envelope for the Reachy Mini hardware layer.

The Reachy Mini has no arms and no gait. What it does have is a 6-DOF Stewart
head on a rotating body, and the two are mechanically coupled: the head's
platform is carried by the body, so the head cannot be yawed arbitrarily far
away from whichever way the body is facing. The limits are therefore not a
matter of taste, and they are not all of one kind:

* Per-axis bounds are the platform's own travel. Pitch and roll are the
  Stewart legs' short axes; yaw is unbounded in the sense that the head can
  face any direction, so its bound is the full turn.
* The head-body yaw delta is a *coupling* limit between two values. Neither
  value alone is out of range when the pair is: a head at +60 and a body at
  -60 are each individually legal and together ask for a 120-degree twist the
  neck cannot make.

What the coupling limit refuses is a substitution, not an impossibility. The
daemon's default kinematics solves the head pose through
``inverse_kinematics_safe(pose, body_yaw, max_relative_yaw=65 deg,
max_body_yaw=160 deg)``, and those two figures are
:data:`HEAD_BODY_YAW_DELTA_LIMIT_DEG` and ``MOTION_ENVELOPE_DEG["body_yaw"]``.
So the twist can never be mechanically violated: the solver keeps it inside the
limit by *moving the body*, and the head pose is its primary task. A pair
outside the limit therefore does not fail - it succeeds with a body yaw the
caller did not ask for, which is the silent substitution this module exists to
refuse.

That makes the two directions of a lone yaw value different things, and the
difference is why one of them is checked and the other is not:

* A lone ``head_yaw`` is honored. Asking the head to face 180 degrees turns the
  body to 115 so the twist lands at exactly the limit. Nothing is substituted,
  because the caller named no body yaw - so this is legal motion and refusing it
  would refuse the verb its own point.
* A lone ``body_yaw`` is honored only within the limit *of the head yaw the
  daemon is already targeting*. With the head target at 0, a body yaw of 160
  reaches 65 and stops: the caller's own explicit value is substituted. That is
  the same failure as an out-of-limit pair, and :func:`envelope_error` refuses it
  when the caller can say what the head target is - see ``head_yaw_target``.

The head pose is absolute, which is what makes the difference a difference: the
daemon's IK sets the requested pose as the head's world frame and drives
``body_yaw`` as a separate joint, so ``head_yaw - body_yaw`` really is the twist
and a full-turn head bound is only coherent because the body carries it round.

Kept in one module because two consumers need the same answer:
:class:`~strands_robots.drivers.reachy.ReachyDriver` (the ``mode="real"``
seam driver) and the agent ``@tool``s that will sit on the same daemon. A
second copy would let the two disagree about the same robot, which is exactly
the failure the existing Device Connect driver's ``look`` RPC and this one must
not have between them.

Refusal, not clamping. A clamp silently substitutes a value the caller did not
ask for, and ``AGENTS.md`` forbids exactly that ("no silent defaults"); for a
robot it is worse than a refusal, because the call reports success while the
head goes somewhere else. :func:`envelope_error` therefore *names the limit it
refused against* so the caller learns the envelope from the refusal rather than
from documentation.

This module imports nothing from the driver and nothing from a transport, so it
is importable and testable on a machine with no Reachy and no daemon.
"""

from __future__ import annotations

from typing import Any

from strands_robots.utils import finite_number_error

#: Per-axis travel, in degrees, as a symmetric bound: a value ``v`` is in range
#: when ``abs(v) <= MOTION_ENVELOPE_DEG[axis]``. Keyed by the name the driver
#: uses in an action dict so a refusal can quote the caller's own spelling.
MOTION_ENVELOPE_DEG: dict[str, float] = {
    "head_pitch": 40.0,
    "head_roll": 40.0,
    "head_yaw": 180.0,
    "body_yaw": 160.0,
}

#: Largest angle, in degrees, the head may be yawed away from the body. This is
#: a bound on ``head_yaw - body_yaw``, not on either one.
HEAD_BODY_YAW_DELTA_LIMIT_DEG: float = 65.0

#: The two keys :data:`HEAD_BODY_YAW_DELTA_LIMIT_DEG` couples. Named so a reader
#: of the pairwise check does not have to infer which pair is meant.
_YAW_PAIR: tuple[str, str] = ("head_yaw", "body_yaw")


def envelope_error(values: dict[str, Any], context: str, head_yaw_target: float | None = None) -> str | None:
    """Report why ``values`` cannot be commanded, or ``None`` when they can.

    Four checks, in this order, because each depends on the previous one
    having passed:

    1. Each bounded axis that is present carries a finite number. A ``nan``
       cannot be compared against a bound at all - ``abs(nan) <= 40`` is
       ``False``, so an unordered value would be refused with a message about
       travel rather than about being unusable. Delegated to
       :func:`~strands_robots.utils.finite_number_error` so this envelope and
       the Device Connect driver's movement RPCs accept the same numbers, which
       also rules out ``bool`` (``True`` would otherwise read as 1 degree).
    2. That axis is within its own travel.
    3. If both members of :data:`_YAW_PAIR` are present, their difference is
       within :data:`HEAD_BODY_YAW_DELTA_LIMIT_DEG`. Checked last because it is
       only meaningful once both values are known to be finite and individually
       legal - otherwise a caller would be told about a coupling when the real
       problem is one value.
    4. If ``body_yaw`` is present *without* ``head_yaw`` and the caller supplied
       ``head_yaw_target``, the same limit applies against that target. Same
       check, one value of which the caller rather than the action supplies:
       the daemon holds the head pose and turns the body no further than the
       limit, so a lone body yaw outside it is substituted exactly as an
       out-of-limit pair is.

    Check (3) is a property of one *action*; check (4) is what makes the limit a
    property of the robot as well. This module has no robot to ask for a missing
    half, so the caller supplies it: a surface that knows where the head is
    pointing passes ``head_yaw_target`` and the coupling is enforced on a
    body-only action too, which is the shape ``reachy_body_turn`` sends. A
    surface that does not know - the Device Connect driver's ``body`` RPC, which
    keeps no record of the pose its ``look`` RPC commanded - passes nothing, and
    a lone ``body_yaw`` stays per-axis there rather than being judged against a
    guess.

    A key this envelope does not know is ignored entirely - not bounded and not
    even checked for finiteness. The driver's action dict also carries antenna
    positions and other pass-through values, and refusing a name this module has
    no bound for would make adding an actuator a change to a safety helper.
    Finiteness for those values belongs to whoever puts them on the wire; the
    driver runs the same shared domain over its whole action dict before calling
    this.

    Args:
        values: Axis name to caller-supplied value. Only the keys named in
            :data:`MOTION_ENVELOPE_DEG` are bounded; others are ignored.
        context: Calling surface to quote in the reason, so a caller can tell
            which of several verbs refused.
        head_yaw_target: The head yaw, in degrees, the robot will be holding
            while ``values`` is carried out, for the case where ``values``
            names no ``head_yaw`` of its own. ``None`` means the caller cannot
            know it - the coupling is then left unchecked rather than guessed
            against a default, because refusing a body turn the robot could
            actually have made is worse than the substitution it prevents.
            Ignored when ``values`` carries ``head_yaw``, which is the stronger
            answer to the same question.

    Returns:
        A reason naming the limit that refused, or ``None`` when every value in
        ``values`` can be honored.
    """
    for axis, limit in MOTION_ENVELOPE_DEG.items():
        if axis not in values:
            continue
        if (reason := finite_number_error(values[axis], axis, context)) is not None:
            return reason
        value = float(values[axis])
        if abs(value) > limit:
            return f"{context}: {axis} {value:g} deg is outside the envelope +/-{limit:g} deg"

    head_key, body_key = _YAW_PAIR
    if body_key not in values:
        return None
    body = float(values[body_key])
    if head_key in values:
        delta = float(values[head_key]) - body
        if abs(delta) > HEAD_BODY_YAW_DELTA_LIMIT_DEG:
            return (
                f"{context}: {head_key} {float(values[head_key]):g} deg and {body_key} "
                f"{body:g} deg differ by {delta:g} deg, which exceeds the "
                f"head-body coupling limit of {HEAD_BODY_YAW_DELTA_LIMIT_DEG:g} deg"
            )
        return None
    if head_yaw_target is None:
        return None
    if (reason := finite_number_error(head_yaw_target, "head_yaw_target", context)) is not None:
        return reason
    delta = abs(float(head_yaw_target) - body)
    if delta > HEAD_BODY_YAW_DELTA_LIMIT_DEG:
        return (
            f"{context}: {body_key} {body:g} deg is {delta:g} deg from the head yaw the daemon "
            f"is targeting ({float(head_yaw_target):g} deg), which exceeds the head-body coupling "
            f"limit of {HEAD_BODY_YAW_DELTA_LIMIT_DEG:g} deg - the head pose is the daemon's "
            f"primary task, so it turns the body no further than the limit; name {head_key} in "
            f"the same action to turn the head with the body"
        )
    return None


def _handle_refusal_envelope(text: str) -> dict[str, Any]:
    """Wrap a refusal sentence in the envelope every ``@tool`` owes a caller."""
    return {"status": "error", "content": [{"text": text}]}


def live_handle_refusal(
    verb: str,
    driver: Any,
    *,
    accessor: str,
    reads: str,
    expected: str,
) -> dict[str, Any] | None:
    """Return the refusal envelope for an unusable live-handle ``driver``, or ``None``.

    The Reachy sibling of :func:`strands_robots.tools.g1._g1_common.live_handle_refusal`,
    kept as a second copy because the two packages must not import each other
    (each stays out of the other's SDK-load path). Every ``reachy_*`` verb that
    takes a wired :class:`~strands_robots.drivers.reachy.ReachyDriver` reads it
    through one accessor; the handle is a live Python object an agent cannot
    synthesize, typed :class:`~typing.Any` so no type leaks into the tool schema.

    The judgement keeps four invariants for every verb: the answer is an error
    *envelope* and never an exception, it names the verb, it names ``driver``,
    and it names the type it received. ``accessor`` must be *callable* on the
    handle, not merely present - a namespace built from a cache dump carries the
    name as data and would fail on the call.

    Args:
        verb: The tool name, opening the message so a transcript reader can
            tell which verb refused.
        driver: The handle to judge; a wrong handle is refused, not coerced.
        accessor: The attribute the verb reads the handle through.
        reads: Why an agent cannot supply the handle, completing "an agent
            cannot synthesize it, because ...".
        expected: What the handle failed to expose and what to pass instead,
            completing "of type 'str' does not expose ...".

    Returns:
        ``None`` when ``driver`` exposes a callable ``accessor``, otherwise the
        ``{"status": "error", "content": [...]}`` envelope.
    """
    if driver is None:
        return _handle_refusal_envelope(
            f"{verb}: `driver` is required. Pass the live ReachyDriver "
            "handle the orchestrator constructed - an agent cannot "
            f"synthesize it, because {reads}."
        )
    if not callable(getattr(driver, accessor, None)):
        return _handle_refusal_envelope(
            f"{verb}: `driver` of type {type(driver).__name__!r} does not expose {expected}"
        )
    return None
