"""Agent-facing wrapper for ``G1Driver.safe_stand_to_squat``.

``G1Driver.safe_stand_to_squat`` is the driver-side compound-posture
entry point for the STAND->SQUAT transition: a caller passes a
Damp-preamble duration in seconds and the driver publishes
:meth:`~unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.Damp`,
sleeps for ``preamble_s``, then issues
:meth:`~unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.SetFsmId(2)`
over the same DDS singleton
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` opens.  The
Damp preamble is the SDK's controller-to-controller handoff smoother
- firing it against an unheld robot leaves it slumping toward the
floor, so the driver's own path is where the FSM-set precondition
gate (``{500, 501, 801}`` - the set the neon bundle's
field notes name, refs strands-labs/robots#358) is enforced.  The neon
bundle's ``g1_safe_stand_to_squat`` verb
(``cagataycali/neon-the-g1/tools/g1_safe_posture.py``) wrapped the
call with an ``_assert_safe_for_damp`` FSM+pose guard that refused
outside ``{500, 501, 801}`` ; that FSM set and the transition
preamble range are the driver's own path to enforce, and
this module is the write-side companion that hands the target to
the driver.

The driver's method itself is not yet plumbed on
:class:`~strands_robots.drivers.g1.G1Driver` today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on); the accessor grades this verb through
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`, so
a handle without a ``safe_stand_to_squat`` method refuses with a
message naming the verb, the ``driver`` parameter and the accessor
it read for, and once the driver lands the same call returns the
envelope the driver wrote verbatim.  This is the same shape
``g1_set_fsm`` (refs strands-labs/robots#3025),
``g1_set_stand_height`` (refs strands-labs/robots#3031),
``g1_set_swing_height`` (refs strands-labs/robots#3032) and
``g1_balance_stand`` (refs strands-labs/robots#3033) already ship.

This module is a thin duck-typed wrapper.  It reads the driver
through ``driver.safe_stand_to_squat`` and returns the envelope the
driver produced verbatim.  A future field the driver adds on the
success path (say, an ``rc`` from the SDK's ``SetFsmId(2)``
handler, the FSM the controller settled into, the observed
``avg_knee`` at Damp-fire time) reaches a caller the moment the
driver writes it, because this verb does not restate the shape.
What it does add is the two refusal envelopes every ``@tool``
handler in this package owes its callers instead of an exception:
a live-handle refusal (``driver`` is ``None`` or a robot *name* or
an object without ``safe_stand_to_squat``) and the driver's own
refusal surfaced verbatim.

The FSM and pose gates are not consulted here.  The neon bundle's
own ``_assert_safe_for_damp`` read the FSM via ``read_fsm_id`` and
the average knee angle via ``_read_lowstate`` and refused outside
``{500, 501, 801}`` or above the ``1.4`` rad pose threshold; the
driver's own path (once landed) is where those gates fire, because
the driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
already grades against the same FSM refresher and the driver has
first-class access to the LowState cache.  A second gate call
here would double the FSM read against the driver's cache, would
refuse a preamble the driver's own path admits, and would fork
the FSM-set precondition table into a second source of truth this
module would then have to keep in sync with the driver's own gate.
Restating any of that on this side would fork the rule
the driver's own path already enforces (refs
strands-labs/robots#358, strands-labs/robots#2916).  ``import
strands_robots.tools.g1.g1_safe_stand_to_squat`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_set_fsm``,
``g1_set_stand_height``, ``g1_set_swing_height``,
``g1_balance_stand``, ``g1_start_task``, ``g1_run_policy``,
``g1_send_action`` and ``g1_stop_task`` give: the driver module
imports :func:`~strands_robots.tools.g1._g1_common.ensure_dds` from
this package at load, so a runtime import of ``G1Driver`` here
would close a cycle, and ``@tool`` calls
:func:`typing.get_type_hints` at decoration time so a string
forward reference cannot resolve without pulling the driver at
import.  The verb is duck-typed on ``safe_stand_to_squat``; any
object with a synchronous ``safe_stand_to_squat(preamble_s)``
returning the driver's envelope satisfies it, which is also how
the tests hand it a hand-rolled double.

What this module does not do.

* Refuse a ``preamble_s`` outside the neon-bundle-observed usable
  range.  The neon bundle defaulted to ``0.5`` seconds and its
  field notes name the observed range; refusing an unlisted
  duration here
  would fork the neon bundle's admission set into a second source
  of truth this module would then have to keep in sync with the
  envelope lookup.  The driver's own path (once landed) is where a
  domain-refusal on ``preamble_s`` lives - a caller who passes an
  unusably long preamble reaches the driver's own refusal, which
  round-trips through this verb verbatim.  The shape validator
  below refuses the cross-type shapes ``bool`` / non-numeric / non-
  finite / non-positive the SDK's ``time.sleep`` cannot use.
* Fire the FSM or pose guard the neon bundle's own
  ``_assert_safe_for_damp`` implements.  The driver's
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  (refs strands-labs/robots#2916) is the one FSM gate for
  actuation, and the LowState pose-check belongs on the same side
  because the driver already caches ``LowState`` for its own
  subscribers.  A second FSM read here would race the driver's
  refresher.
* Fall back to a bare ``LocoClient.SetFsmId(2)`` when the Damp
  preamble refuses.  The neon bundle's own remediation names that
  fallback in its refusal message ("call the bare loco operation
  via ``use_unitree(service='loco', operation='SetFsmId(2)',
  parameters={})``") but does not fire it silently, and this verb
  does not either - a caller who wants the bare transition names
  the ``use_unitree`` verb explicitly.

The verb is a synchronous ``@tool`` handler.  ``@tool`` calls
:func:`typing.get_type_hints` at decoration time and inspects the
signature to build the JSON schema the model reaches; the
``preamble_s: float`` parameter surfaces as a numeric field with a
neon-bundle-observed default of ``0.5`` seconds, and the ``driver:
Any`` parameter surfaces as a live handle the orchestrator
constructs (not a robot *name* - the accessor refusal below names
that shape explicitly).  The default value on ``preamble_s`` is
the same one the neon bundle documented: a caller who omits the
knob reaches the driver with the safe default the field notes
against the real robot preferred.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal
from strands_robots.utils import positive_finite_number_error


def _refusal_envelope(text: str) -> dict[str, Any]:
    """Return the ``@tool`` error envelope with ``text`` inside.

    Kept as a small free helper so every ``preamble_s`` refusal
    path in this module renders the same shape a caller can grep
    for, matching the driver's own
    :func:`~strands_robots.drivers.g1._refuse` free function on
    the write side and the ``g1_set_stand_height`` /
    ``g1_set_swing_height`` / ``g1_balance_stand`` verbs already
    shipped.
    """
    return {"status": "error", "content": [{"text": text}]}


@tool
def g1_safe_stand_to_squat(
    driver: Any,
    preamble_s: float = 0.5,
) -> dict[str, Any]:
    """Fire the Damp -> SetFsmId(2) compound-posture transition.

    Calls ``G1Driver.safe_stand_to_squat`` once and returns the
    envelope the driver produced verbatim.  The driver's method
    fires a ``LocoClient.Damp`` preamble, sleeps for
    ``preamble_s`` seconds, then issues
    ``LocoClient.SetFsmId(2)`` - the SDK's canonical STAND-to-
    SQUAT transition (working around the SDK's LocoClient.StandUp2Squat mis-route to FSM 706, refs strands-labs/robots#358).  The Damp preamble is a controller-to-
    controller handoff smoother the neon bundle's field notes
    documented (refs the merged
    strands-labs/robots#2916), and the driver's own path (once
    landed) is where the FSM-set precondition gate ``{500, 501, 801}``
    (the set the neon bundle's field notes name) and the upright FSM precondition
    fire.  This verb's only job is the pass-through of one call
    and the envelope-shaped refusal on two shapes the driver's
    own path cannot format: a wrong-shape handle, and a
    non-numeric / non-finite / non-positive / boolean
    ``preamble_s``.

    Args:
        driver: The live :class:`~strands_robots.drivers.g1.G1Driver`
            handle the orchestrator constructed.  Typed
            :class:`~typing.Any` at runtime rather than as
            ``G1Driver`` because the driver module imports
            :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            from this package at load, so a runtime import of
            ``G1Driver`` here would close a cycle (see the module
            docstring's "import-cycle" note).  A caller passing
            ``None``, a robot *name* (string) or any object
            without a callable ``safe_stand_to_squat`` accessor
            is refused by
            :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
            with a message naming the verb, the parameter and the
            accessor.
        preamble_s: How long the driver's ``Damp`` preamble holds
            before ``SetFsmId(2)`` fires, in seconds.  The neon
            bundle defaulted to ``0.5`` seconds
            (``cagataycali/neon-the-g1/tools/g1_safe_posture.py``)
            and its field notes name the observed range against
            the real robot.
            Validated against
            :func:`~strands_robots.utils.positive_finite_number_error`
            because the caller-facing domain is a positive real
            span of time (the value is the argument to
            :func:`time.sleep` inside the driver's method, which
            refuses a negative value and silently no-ops on
            ``0`` - a preamble of zero collapses to a bare
            ``SetFsmId(2)`` write, which the neon bundle's own
            wrapper documented as a distinct
            ``use_unitree`` verb).  ``nan`` / ``inf`` / non-numeric
            shapes are refused here rather than reaching the
            driver's own ``float`` call or the SDK's
            :func:`time.sleep` (``nan`` in a duration silently
            no-ops and ``inf`` hangs the write path).  A ``bool``
            payload is refused for the same reason
            :func:`positive_finite_number_error` refuses it:
            ``True`` would act as a silent ``1.0`` (a
            preamble twice as long as the neon default), ``False``
            as a silent ``0.0`` (skips the Damp preamble and
            silently collapses to a bare ``SetFsmId(2)``) -
            neither is a value a caller writing ``True`` would
            have named on purpose.

    Returns:
        The envelope ``G1Driver.safe_stand_to_squat`` returned.
        On the success path the driver's method will surface the
        SDK's ``rc`` inside a ``{"status": "success", "content":
        [{"json": {"rc": 0, "message": "Damp -> sleep(0.5s) ->
        SetFsmId(2) dispatched"}}]}`` envelope (or the
        equivalent shape once the driver's method lands); on the
        driver's refusal path (an SDK-side raise, an FSM outside
        ``{500, 501, 801}``,
        or a caller running this verb before the driver's method
        lands) it is ``{"status": "error", "content": [{"text":
        "..."}]}`` with the driver's own reason inside.  The verb
        does not reshape either shape - a future field the driver
        adds on the success path reaches a caller the moment the
        driver writes it, because this verb passes the envelope
        through.
    """
    # The handle is a live Python object typed :class:`~typing.Any`
    # (see the module docstring's import-cycle note), so the tool
    # schema carries no signal that ``None`` or a robot *name* is
    # refused.  The shared ``live_handle_refusal`` guard is the one
    # implementation of that judgement for this package; it is
    # keyed on the accessor the verb reads, which for this verb is
    # ``safe_stand_to_squat`` (a callable that fires a
    # ``LocoClient.Damp`` preamble and a follow-up
    # ``LocoClient.SetFsmId(2)``, then returns the driver's
    # envelope) rather than the sensor verbs' ``_snapshot``.
    # Returning its refusal envelope here rather than raising keeps
    # the four invariants every ``@tool`` handler owes a caller
    # (envelope not exception, names the verb, names ``driver``,
    # names the type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_safe_stand_to_squat",
        driver,
        accessor="safe_stand_to_squat",
        reads=(
            "the verb requests a G1 Damp -> SetFsmId(2) compound "
            "transition through the driver's own SDK-facing write "
            "path and reads back the envelope the driver produced"
        ),
        expected=(
            "a callable ``safe_stand_to_squat(preamble_s)`` "
            "returning the driver's write envelope - pass the live "
            "G1Driver handle the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    # ``preamble_s`` is a data parameter the tool schema *does*
    # describe (a positive finite float duration in seconds, with a
    # neon-bundle-observed default of ``0.5`` seconds), so a model
    # can synthesize the wrong shape here as easily as it can
    # reach the verb with the right one.  The shared
    # ``positive_finite_number_error`` validator covers the shapes
    # the driver's own ``safe_stand_to_squat`` would surface as an
    # inner raise: a ``None`` payload, a non-numeric shape, and
    # the non-finite / bool subclass / non-positive shapes the
    # validator refuses.  Naming ``preamble_s`` here keeps the
    # four invariants: envelope not exception, names the verb,
    # names the parameter, names the shape received.  The
    # positive-only constraint matches the driver's own
    # :func:`time.sleep` call (which refuses a negative value) and
    # refuses the ``0`` boundary because a zero-length preamble
    # collapses to a bare ``SetFsmId(2)`` write into Squat (FSM 2), which the neon
    # bundle's own wrapper documented as a distinct
    # ``use_unitree`` verb.
    preamble_error = positive_finite_number_error(preamble_s, "preamble_s", "g1_safe_stand_to_squat")
    if preamble_error is not None:
        return _refusal_envelope(preamble_error)

    return driver.safe_stand_to_squat(preamble_s)
