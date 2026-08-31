"""Agent-facing wrapper for ``G1Driver.send_action``.

``G1Driver.send_action`` publishes one :class:`LowCmd_` frame on
``rt/lowcmd`` for a joint-name-keyed action dict.  It is the driver's
*one-frame* write - a caller who wants a schedule (500 Hz, 200 Hz)
calls this verb on their own timer, or reaches
:meth:`~strands_robots.drivers.g1.G1Driver.run_policy` which owns the
control loop today.  The driver's method already re-gates through
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` with
scope ``"arm"`` and refuses with the FSM / battery message its own
docstring names verbatim; the verb here is the agent-facing side of
that single write, not a second gate.

This module is a thin duck-typed wrapper.  It reads the driver through
``driver.send_action`` (the one method the underlying driver exposes for
this write) and returns the envelope the driver produced verbatim.  A
future field the driver adds on its success path (say, a sequence
number on the ``rt/lowcmd`` write) reaches a caller the moment the
driver writes it, because this verb does not restate the shape.  What
it does add is the same three refusal envelopes every ``@tool`` handler
in this package owes its callers instead of an exception: a live-handle
refusal (``driver`` is ``None`` or a robot *name* or an object without
the accessor), an ``action`` refusal (missing, not a dict, empty), and
the driver's own refusal surfaced verbatim.

The driver's method gate is left alone.  ``_check_motion_gates`` reads
the driver's cached ``mode_machine`` and battery percentage on every
write - a caller running this verb from a supervisor cannot bypass the
gate by pre-flighting from an agent that read the cache a second
earlier - and the gate's refusal is the same three sentences the
driver writes into every wire method (``send_action``, ``run_policy``,
``start_task``).  Restating any of that on this side would be a second
source of truth for a rule the driver's own path already enforces
(refs strands-labs/robots#358, strands-labs/robots#2916).  ``import
strands_robots.tools.g1.g1_send_action`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene contract,
refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_stop_task`` gives: the
driver module imports :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
from this package at load, so a runtime import of ``G1Driver`` here
would close a cycle, and ``@tool`` calls
:func:`typing.get_type_hints` at decoration time so a string forward
reference cannot resolve without pulling the driver at import.  The
verb is duck-typed on ``send_action``; any object with a synchronous
``send_action(action, robot_name=...)`` returning the driver's write
envelope satisfies it, which is also how the tests hand it a
hand-rolled double.

What this module does not do.

* Build the ``LowCmd_`` payload.  The driver's
  :func:`~strands_robots.drivers.g1._build_lowcmd_from_action` builds
  every wire frame the SDK writes; a second builder path here would
  fork the mapping between joint-name keys and the SDK's index-keyed
  arrays that the driver's :data:`~strands_robots.drivers.g1._G1_JOINT_INDEX`
  already names as one source of truth.
* Re-run the FSM or battery gate.  The driver's
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` is the
  one gate; a second gate call here would double the read against a
  cache the driver's own FSM refresher fills, and a caller who saw the
  first gate answer ``None`` and the second refuse could not tell
  which read it should trust.
* Schedule frames.  ``send_action`` is one wire frame; a caller who
  wants a 500 Hz loop reaches :meth:`~strands_robots.drivers.g1.G1Driver.run_policy`
  which owns the loop.  This verb is the one-shot companion.
* Decode the driver's ``fsm_id`` or ``mode_machine`` fields into
  friendlier labels.  The driver's own
  :meth:`~strands_robots.drivers.g1.G1Driver.get_task_status` and the
  ``g1_get_state`` verb surface the same fields under the same names;
  restating them here would be a second source of truth for a domain
  the driver's own envelope already carries verbatim.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


def _refusal_envelope(text: str) -> dict[str, Any]:
    """Return the ``@tool`` error envelope with ``text`` inside.

    Kept as a small free helper so every ``action``-refusal path in this
    module renders the same shape a caller can grep for, matching the
    driver's own :func:`~strands_robots.drivers.g1._refuse` free
    function on the write side.
    """
    return {"status": "error", "content": [{"text": text}]}


@tool
def g1_send_action(driver: Any, action: dict[str, Any] | None = None) -> dict[str, Any]:
    """Publish one ``LowCmd_`` frame on ``rt/lowcmd`` for ``action``.

    Calls :meth:`~strands_robots.drivers.g1.G1Driver.send_action` once
    and returns the envelope the driver produced verbatim.  The
    driver's method re-gates through
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    with scope ``"arm"``, so a caller whose FSM is outside the arm-SDK
    admission set or whose battery is below the driver's floor gets
    the driver's own refusal string (refs strands-labs/robots#2916).
    A caller who wants a 500 Hz control loop reaches
    :meth:`~strands_robots.drivers.g1.G1Driver.run_policy`; this verb
    is the one-frame write.

    Args:
        driver: An object with a callable ``send_action(action,
            robot_name=...)`` returning the driver's write envelope
            (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep
            this module out of the import cycle the driver's own
            :func:`~strands_robots.tools.g1._g1_common.ensure_dds` reach
            into this package would close - see the module docstring's
            SDK-load-hygiene note.  The verb is duck-typed on
            ``send_action``; any object with that method returning the
            envelope shape the driver writes will satisfy it.
        action: A joint-name-keyed dict.  Values are either a target
            position in radians (the driver falls back to reference
            gains) or an inner dict carrying any subset of ``q`` /
            ``kp`` / ``kd`` / ``dq`` / ``tau``; the driver's own
            :func:`~strands_robots.drivers.g1._build_lowcmd_from_action`
            refuses a missing ``q`` verbatim so a silently-zeroed
            target cannot make it onto the wire.  A caller who passes
            an empty dict is refused here rather than on the driver:
            a wire frame that names no joint is a no-op that would
            still consume the arm-SDK gate's admission read, and the
            refusal string names ``action`` and the remedy so a
            caller reading the envelope can fix the call.

    Returns:
        The envelope :meth:`G1Driver.send_action` returned.  On the
        success path this is ``{"status": "success", "content":
        [{"json": {"topic": "rt/lowcmd", "joints": [...],
        "fsm_id": ..., "mode_machine": ...}}]}``; on the driver's
        refusal path (gate flip, publisher not initialised,
        action-dict validation, SDK missing on the write path,
        publish error) it is ``{"status": "error", "content":
        [{"text": "..."}]}`` with the driver's own reason inside.
        The verb does not reshape either shape - a future field the
        driver adds on the success path reaches a caller the moment
        the driver writes it, because this verb passes the envelope
        through.
    """
    # The handle is a live Python object typed :class:`~typing.Any` (see the
    # module docstring's import-cycle note), so the tool schema carries no
    # signal that ``None`` or a robot *name* is refused.  The shared
    # ``live_handle_refusal`` guard is the one implementation of that
    # judgement for this package; it is keyed on the accessor the verb
    # reads, which for this verb is ``send_action`` (a callable that writes
    # one ``LowCmd_`` frame and returns the driver's envelope) rather than
    # the sensor verbs' ``_snapshot``.  Returning its refusal envelope here
    # rather than raising keeps the four invariants every ``@tool`` handler
    # owes a caller (envelope not exception, names the verb, names
    # ``driver``, names the type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_send_action",
        driver,
        accessor="send_action",
        reads=(
            "the verb publishes one LowCmd_ frame on rt/lowcmd through the "
            "driver's own write path and reads back the wire-frame envelope "
            "the driver produced"
        ),
        expected=(
            "a callable ``send_action(action, robot_name=...)`` returning "
            "the driver's write envelope - pass the live G1Driver handle "
            "the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    # ``action`` is a data parameter the tool schema *does* describe (a
    # dict of joint-name to target), so a model can synthesize the wrong
    # shape here as easily as it can reach the verb with the right one.
    # These three refusals cover the shapes the driver's own
    # ``_build_lowcmd_from_action`` would surface as an inner refusal
    # naming a joint (a missing ``q``), which is a shape a caller reading
    # ``action`` alone cannot map back to the parameter.  Naming ``action``
    # here keeps the four invariants: envelope not exception, names the
    # verb, names the parameter, names the shape received.
    if action is None:
        return _refusal_envelope(
            "g1_send_action: `action` is required. Pass a joint-name-keyed "
            "dict of target positions (radians) or per-joint {q, kp, kd, dq, "
            "tau} dicts - see G1Driver.send_action for the shape "
            "(refs strands-labs/robots#361)."
        )
    if not isinstance(action, dict):
        return _refusal_envelope(
            f"g1_send_action: `action` of type {type(action).__name__!r} is "
            "not a dict. Pass a joint-name-keyed dict of target positions "
            "(radians) or per-joint {q, kp, kd, dq, tau} dicts - see "
            "G1Driver.send_action for the shape (refs strands-labs/robots#361)."
        )
    if not action:
        return _refusal_envelope(
            "g1_send_action: `action` is empty. A wire frame that names no "
            "joint is a no-op that would still consume the arm-SDK gate's "
            "admission read; pass at least one joint-name key mapping to a "
            "target position or per-joint dict (refs strands-labs/robots#2916)."
        )

    return driver.send_action(action)
