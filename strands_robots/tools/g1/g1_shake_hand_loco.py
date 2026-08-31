"""Agent-facing wrapper for ``G1Driver.shake_hand_loco``.

``G1Driver.shake_hand_loco`` is the driver-side ``LocoClient.ShakeHand``
entry point: a caller passes an integer ``stage`` (``-1`` toggle, ``0``
reach, ``1`` shake) and the driver publishes
:meth:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.ShakeHand` over
the same DDS singleton
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` opens.  The SDK
exposes ``ShakeHand`` as a public ``LocoClient`` method that composes
its argument into a ``SetTaskId`` payload against a fixed internal
three-stage table: ``0`` reaches the arm out, ``1`` shakes the extended
hand, and ``-1`` asks the SDK to toggle the internal stage counter
(the SDK's own default reads through the sentinel).  The neon bundle's
``g1_shake_hand_loco`` verb
(``cagataycali/neon-the-g1/tools/g1_locomotion.py``) wrapped the call
under a single-writer lock and coerced the argument through
:class:`int` before dispatch (refs strands-labs/robots#358), and this
module is the write-side companion that hands the target to the
driver.

The driver's method itself is not yet plumbed on
:class:`~strands_robots.drivers.g1.G1Driver` today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on); the accessor grades this verb through
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`, so a
handle without a ``shake_hand_loco`` method refuses with a message
naming the verb, the ``driver`` parameter and the accessor it read
for, and once the driver lands the same call returns the envelope
the driver wrote verbatim.  This is the same shape ``g1_set_fsm``
(refs strands-labs/robots#3025), ``g1_set_stand_height`` (refs
strands-labs/robots#3031), ``g1_set_swing_height`` (refs
strands-labs/robots#3032), ``g1_balance_stand`` (refs
strands-labs/robots#3033) and ``g1_wave_hand_loco`` (refs
strands-labs/robots#3041) already ship.

This module is a thin duck-typed wrapper.  It reads the driver
through ``driver.shake_hand_loco`` and returns the envelope the
driver produced verbatim.  A future field the driver adds on the
success path (say, an ``rc`` from the SDK's ``SetTaskId`` handler,
the composed task id the controller dispatched, the stage the SDK's
internal counter settled into after a ``-1`` toggle) reaches a
caller the moment the driver writes it, because this verb does not
restate the shape.  What it does add is the two refusal envelopes
every ``@tool`` handler in this package owes its callers instead
of an exception: a live-handle refusal (``driver`` is ``None`` or a
robot *name* or an object without ``shake_hand_loco``) and the
driver's own refusal surfaced verbatim.

The FSM gate is not consulted here.  The driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` is
the one gate for arm-SDK writes (``send_action`` / ``run_policy`` /
``start_task``); ``ShakeHand`` sits on the loco side of the same DDS
singleton and dispatches through ``SetTaskId``, whose admission is
the driver's own concern (the neon bundle's docstring named the
verb as call-twice-with-3s-between-calls to complete the motion -
``SetTaskId`` accepts the call from every FSM, though the *observed*
gesture depends on the current FSM the way the sibling
``g1_wave_hand_loco`` and the built-in wave/turn primitives do).  A
second gate call here would double the read against a cache the
driver's FSM refresher fills.  Restating any of that on this side
would be a second source of truth for a rule the driver's own path
already enforces (refs strands-labs/robots#358,
strands-labs/robots#2916).  ``import
strands_robots.tools.g1.g1_shake_hand_loco`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_set_fsm``,
``g1_set_stand_height``, ``g1_set_swing_height``, ``g1_balance_stand``,
``g1_wave_hand_loco``, ``g1_start_task``, ``g1_run_policy``,
``g1_send_action`` and ``g1_stop_task`` give: the driver module
imports :func:`~strands_robots.tools.g1._g1_common.ensure_dds` from
this package at load, so a runtime import of ``G1Driver`` here would
close a cycle, and ``@tool`` calls :func:`typing.get_type_hints`
at decoration time so a string forward reference cannot resolve
without pulling the driver at import.  The verb is duck-typed on
``shake_hand_loco``; any object with a synchronous
``shake_hand_loco(stage)`` returning the driver's envelope
satisfies it, which is also how the tests hand it a hand-rolled
double.

What this module does not do.

* Refuse a ``stage`` outside the SDK-observed admitted set
  ``{-1, 0, 1}``. Those three are the stages the SDK's dispatcher
  decodes against its fixed internal table; refusing an unlisted
  stage here would fork that admission set into a second source
  of truth this module would then have to keep in sync with the
  envelope lookup.  The SDK's own handler returns ``rc=7303``
  ("Invalid task id (loco)") on a stage outside its programmed
  set, so the driver's method (once landed) is where an in-set
  refusal lives - a caller who passes ``7`` here reaches the
  driver's own refusal, which round-trips through this verb
  verbatim.  The bare-int shape checks below refuse the cross-type
  shapes ``bool``/``float``/``str`` the SDK's :class:`int`
  coercion in the neon wrapper silently transformed rather than
  declined; those are separate from the in-set admission the
  envelope module owns.
* Encode the ``LocoClient.ShakeHand`` dispatch.  The SDK's public
  method and the neon bundle's single-writer lock are driver-side
  wire concerns; the verb passes ``stage`` through unchanged and
  the driver's method decides how to reach the SDK's handler.  A
  caller reading this verb's docstring learns the ergonomic
  contract, not the wire format.
* Decode the SDK's ``rc`` return into a label.  The
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` table
  (surfaced by :mod:`~strands_robots.tools.g1.g1_error_codes`)
  names every SDK return code the driver's method may quote;
  restating the label here would fork the same table.
* Restate the driver's refusal wording.  Whatever text the
  driver's method writes (a rc-decoded sentence, a gate refusal,
  a no-shake-hand-yet message like ``g1_start_task``'s
  registry-not-wired one) passes through this verb; a verbatim
  quote here would trap the verb to one release's prose (refs
  strands-labs/robots#2874).
* Sequence the two-stage gesture.  The neon bundle's docstring
  named ``ShakeHand`` as call-twice-with-3s-between-calls to
  complete the motion (once to reach out, again to shake); this
  verb does not chain the two calls.  A caller who wants the
  full gesture issues the two calls in order and reads each
  envelope, or passes ``-1`` and lets the SDK's internal
  toggle-stage counter advance on each call.
* Track the SDK's internal stage counter.  The ``-1`` sentinel
  reads the SDK's own default and toggles between the two
  admitted stages on successive calls; that state lives inside
  the SDK's dispatcher and is not surfaced through this verb.
  A caller who wants to know which stage a ``-1`` call settled
  into reads the driver's success envelope (once the write path
  lands) - the driver's method may quote the resulting stage id,
  though today's driver does not.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


def _refusal_envelope(text: str) -> dict[str, Any]:
    """Return the ``@tool`` error envelope with ``text`` inside.

    Kept as a small free helper so every ``stage`` refusal path in
    this module renders the same shape a caller can grep for,
    matching the driver's own
    :func:`~strands_robots.drivers.g1._refuse` free function on the
    write side and the shape ``g1_balance_stand`` and
    ``g1_wave_hand_loco``'s :func:`_refusal_envelope` sibling
    helpers render.
    """
    return {"status": "error", "content": [{"text": text}]}


@tool
def g1_shake_hand_loco(
    driver: Any,
    stage: int | None = None,
) -> dict[str, Any]:
    """Dispatch the G1's built-in ``LocoClient.ShakeHand`` task.

    Calls ``G1Driver.shake_hand_loco`` once and returns the envelope
    the driver produced verbatim.  The driver's method routes to
    ``LocoClient.ShakeHand`` (which internally composes a
    ``SetTaskId`` payload); a caller who passes ``0`` reaches the
    reach-out stage, a caller who passes ``1`` reaches the shake
    stage, and a caller who passes ``-1`` asks the SDK to toggle
    its internal stage counter (the SDK's own default reads through
    the sentinel).  The ``stage`` entry below names all three
    variants and the SDK's ``rc=7303`` refusal on any integer
    outside the set; a caller planning the write reads it first and
    reaches this verb once the target ``stage`` is decided.

    ``ShakeHand`` dispatches through ``SetTaskId`` rather than the
    arm-SDK path ``send_action`` / ``run_policy`` walk, so it does
    not read the driver's
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    admission set (refs strands-labs/robots#2916): the neon bundle's
    docstring named the verb as call-twice-with-3s-between-calls
    to complete the motion, and ``SetTaskId`` accepts the call from
    every FSM.  The *observed* behaviour of the resulting gesture
    depends on the FSM the controller was already in.  A caller who
    wants a particular pose before the shake issues a companion
    ``g1_set_fsm`` call (refs strands-labs/robots#3025) in order
    and reads each envelope; this verb does not chain the FSM
    transition itself.

    The two-stage gesture is not chained here either.  A caller who
    wants the full reach-out-then-shake motion issues two calls in
    order and reads each envelope, either explicitly (``stage=0``
    then ``stage=1``) or by letting the SDK's internal counter
    advance (``stage=-1`` twice).  The neon bundle's docstring named
    the 3s pause between calls as the observed pacing the
    controller admits; this verb does not sleep between calls
    (that is orchestration the caller owns).

    Args:
        driver: An object with a callable
            ``shake_hand_loco(stage)`` returning the driver's
            write envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to
            keep this module out of the import cycle the driver's
            own :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note.  The verb is duck-
            typed on ``shake_hand_loco``; any object with that
            method returning the envelope shape the driver writes
            will satisfy it.  On today's driver ``shake_hand_loco``
            is not yet exposed, so the :func:`live_handle_refusal`
            grader refuses the call with a message naming the
            accessor; once the driver's method lands (the SDK-
            facing gate work in strands-labs/robots#358), the
            same call returns the driver's envelope verbatim.
        stage: The integer stage argument
            ``LocoClient.ShakeHand`` admits.  The SDK's internal
            table decodes three values: ``0`` (reach out - extend
            the arm forward),
            ``1`` (shake - move the extended hand), and ``-1``
            (toggle the SDK's internal stage counter, the sentinel
            the SDK's default reads through).  This verb does not
            refuse a stage outside that set - the driver's method
            (once landed) is where an in-set refusal lives, and a
            caller who passes ``7`` reaches either the driver's
            own refusal or the SDK's ``rc=7303``
            ("Invalid task id (loco)") handler through the verb's
            pass-through.  The shape checks below refuse a
            ``None`` payload, a ``bool`` payload (``bool`` is an
            ``int`` subclass, so ``True`` would act as a silent
            ``1`` (shake) and ``False`` as a silent ``0`` (reach
            out) - neither is a stage id a caller writing the
            boolean would have named on purpose), and a
            non-integer shape (``float``, ``str``) that the neon
            bundle's own ``int(...)`` coercion silently
            transformed rather than declined.  Those are shape
            refusals, not domain refusals; the in-set admission
            belongs on the driver's write path the same way
            ``g1_balance_stand``'s ``{0, 3}`` admission does.

    Returns:
        The envelope ``G1Driver.shake_hand_loco`` returned.  On the
        success path the driver's method will surface the SDK's
        outcome inside a ``{"status": "success", "content":
        [{"json": {"rc": 0, "message": "ShakeHand(stage=0)
        dispatched"}}]}`` envelope (or whatever richer shape the
        driver settles on, potentially quoting the stage the SDK's
        internal counter settled into after a ``-1`` toggle); on
        the driver's refusal path (an SDK-side raise, a caller
        running this verb before the driver's method lands, an
        SDK ``rc`` other than ``0`` like ``rc=7303`` on an
        out-of-set stage) it is ``{"status": "error", "content":
        [{"text": "..."}]}`` with the driver's own reason inside.
        The verb does not reshape either shape - a future field
        the driver adds on the success path reaches a caller the
        moment the driver writes it, because this verb passes the
        envelope through.
    """
    # The handle is a live Python object typed :class:`~typing.Any`
    # (see the module docstring's import-cycle note), so the tool
    # schema carries no signal that ``None`` or a robot *name* is
    # refused.  The shared ``live_handle_refusal`` guard is the
    # one implementation of that judgement for this package; it is
    # keyed on the accessor the verb reads, which for this verb is
    # ``shake_hand_loco`` (a callable that calls
    # ``LocoClient.ShakeHand`` and returns the driver's envelope)
    # rather than the sensor verbs' ``_snapshot``.  Returning its
    # refusal envelope here rather than raising keeps the four
    # invariants every ``@tool`` handler owes a caller (envelope
    # not exception, names the verb, names ``driver``, names the
    # type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_shake_hand_loco",
        driver,
        accessor="shake_hand_loco",
        reads=(
            "the verb dispatches the G1's built-in LocoClient.ShakeHand "
            "task through the driver's own SDK-facing write path and "
            "reads back the envelope the driver produced"
        ),
        expected=(
            "a callable ``shake_hand_loco(stage)`` returning the "
            "driver's write envelope - pass the live G1Driver handle "
            "the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    # ``stage`` is a data parameter the tool schema *does* describe
    # (an integer stage id, with the SDK-observed admitted set
    # ``{-1, 0, 1}`` documented above),
    # so a model can synthesize the wrong shape here as easily as
    # it can reach the verb with the right one.  The refusals
    # below cover the shapes the driver's own ``shake_hand_loco``
    # would surface as an inner raise or as a silent coercion: a
    # ``None`` payload (no defensible default - the neon bundle's
    # ``stage=-1`` default routed the SDK to toggle its internal
    # counter, a semantic that requires the caller to have named
    # the sentinel on purpose), a ``bool`` payload (``int``
    # subclass, would coerce to ``0`` (reach out) or ``1``
    # (shake)), and a non-integer shape (``float`` / ``str``) the
    # neon bundle's own ``int(...)`` silently transformed.
    # Naming ``stage`` here keeps the four invariants: envelope
    # not exception, names the verb, names the parameter, names
    # the shape received.  The in-set admission ``{-1, 0, 1}`` is
    # not enforced here (see the module docstring's "does not
    # refuse a stage outside the admitted set" note); that
    # belongs on the driver.
    if stage is None:
        return _refusal_envelope(
            "g1_shake_hand_loco: `stage` is required. Pass an "
            "integer stage id (SDK-observed admitted set "
            "{-1, 0, 1}; -1 = toggle SDK's internal counter, "
            "0 = reach out, 1 = shake) - see "
            "`g1_list_shake_hand_stages` for the read-only "
            "descriptors (refs strands-labs/robots#358)."
        )
    # ``bool`` is an ``int`` subclass, so a bare ``isinstance(..., int)``
    # test would let ``True`` through as a silent ``1`` (shake) and
    # ``False`` as a silent ``0`` (reach out) - both inside the
    # ``{-1, 0, 1}`` admitted set, so the SDK would dispatch a stage
    # the caller writing the boolean did not name on purpose.
    # Refusing ``bool`` explicitly matches the shape refusal
    # ``finite_number_error`` / ``positive_count_error`` render for
    # the same reason on the numeric verbs, and keeps the verb's
    # message naming the parameter and the shape received rather
    # than silently transitioning to a stage the caller writing
    # ``True`` did not name.
    if isinstance(stage, bool):
        return _refusal_envelope(
            f"g1_shake_hand_loco: `stage` must be an integer stage id "
            f"(int, not bool), got {type(stage).__name__!r}. See "
            f"`g1_list_shake_hand_stages` for the read-only "
            f"descriptors (refs strands-labs/robots#358)."
        )
    if not isinstance(stage, int):
        return _refusal_envelope(
            f"g1_shake_hand_loco: `stage` must be an integer stage id "
            f"(int), got {type(stage).__name__!r}. See "
            f"`g1_list_shake_hand_stages` for the read-only "
            f"descriptors (refs strands-labs/robots#358)."
        )

    return driver.shake_hand_loco(stage)
