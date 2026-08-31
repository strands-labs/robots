"""Agent-facing wrapper for ``G1Driver.wave_hand_loco``.

``G1Driver.wave_hand_loco`` is the driver-side ``LocoClient.WaveHand``
entry point: a caller passes a boolean ``turn_flag`` and the driver
publishes :meth:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.WaveHand`
over the same DDS singleton
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` opens.  The SDK
exposes ``WaveHand`` as a public ``LocoClient`` method that composes
its argument into one of two ``SetTaskId`` payloads: ``turn_flag=False``
reaches the wave-in-place task and ``turn_flag=True`` reaches the
wave-and-turn-around task (the two variants the neon bundle observed
in-hand).  The neon bundle's ``g1_wave_hand_loco`` verb
(``cagataycali/neon-the-g1/tools/g1_locomotion.py``) wrapped the call
under a single-writer lock and coerced the argument through
:class:`bool` before dispatch (refs strands-labs/robots#358), and
this module is the write-side companion that hands the target to the
driver.

The driver's method itself is not yet plumbed on
:class:`~strands_robots.drivers.g1.G1Driver` today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on); the accessor grades this verb through
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`, so a
handle without a ``wave_hand_loco`` method refuses with a message
naming the verb, the ``driver`` parameter and the accessor it read
for, and once the driver lands the same call returns the envelope
the driver wrote verbatim.  This is the same shape ``g1_set_fsm``
(refs strands-labs/robots#3025), ``g1_set_stand_height`` (refs
strands-labs/robots#3031), ``g1_set_swing_height`` (refs
strands-labs/robots#3032) and ``g1_balance_stand`` (refs
strands-labs/robots#3033) already ship.

This module is a thin duck-typed wrapper.  It reads the driver
through ``driver.wave_hand_loco`` and returns the envelope the
driver produced verbatim.  A future field the driver adds on the
success path (say, an ``rc`` from the SDK's ``SetTaskId`` handler,
the composed task id the controller dispatched) reaches a caller
the moment the driver writes it, because this verb does not
restate the shape.  What it does add is the two refusal envelopes
every ``@tool`` handler in this package owes its callers instead
of an exception: a live-handle refusal (``driver`` is ``None`` or a
robot *name* or an object without ``wave_hand_loco``) and the
driver's own refusal surfaced verbatim.

The FSM gate is not consulted here.  The driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` is
the one gate for arm-SDK writes (``send_action`` / ``run_policy`` /
``start_task``); ``WaveHand`` sits on the loco side of the same DDS
singleton and dispatches through ``SetTaskId``, whose admission is
the driver's own concern (the neon bundle's docstring named the
verb as "does not require FSM 500+" - ``SetTaskId`` accepts the
call from every FSM, though the *observed* behaviour depends on the
current FSM the way the sibling ``g1_shake_hand_loco`` and the
built-in wave/turn primitives do).  A second gate call here would
double the read against a cache the driver's FSM refresher fills.
Restating any of that on this side would be a second source of
truth for a rule the driver's own path already enforces (refs
strands-labs/robots#358, strands-labs/robots#2916).  ``import
strands_robots.tools.g1.g1_wave_hand_loco`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_set_fsm``,
``g1_set_stand_height``, ``g1_set_swing_height``, ``g1_balance_stand``,
``g1_start_task``, ``g1_run_policy``, ``g1_send_action`` and
``g1_stop_task`` give: the driver module imports
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` from this
package at load, so a runtime import of ``G1Driver`` here would
close a cycle, and ``@tool`` calls :func:`typing.get_type_hints`
at decoration time so a string forward reference cannot resolve
without pulling the driver at import.  The verb is duck-typed on
``wave_hand_loco``; any object with a synchronous
``wave_hand_loco(turn_flag)`` returning the driver's envelope
satisfies it, which is also how the tests hand it a hand-rolled
double.

What this module does not do.

* Refuse a non-``bool`` ``turn_flag`` through Python's ``bool()``
  coercion.  The neon wrapper called ``bool(turn)`` before the SDK
  saw the value, silently transforming ``1`` / ``"yes"`` / ``None``
  into an admitted task id; this verb refuses those shapes
  explicitly because the two admitted values (``False``,
  ``True``) are the only two variants the neon bundle observed,
  and a caller passing ``1`` here is not naming a
  turn-flag variant on purpose.  The driver's own write path owns
  the in-set admission; the shape
  refusal here mirrors the neon verb's own ``turn_flag must be
  bool`` refusal so the two paths render the same shape a caller
  can grep for.
* Encode the ``LocoClient.WaveHand`` dispatch.  The SDK's public
  method and the neon bundle's single-writer lock are driver-side
  wire concerns; the verb passes ``turn_flag`` through unchanged
  and the driver's method decides how to reach the SDK's handler
  and which of the two ``SetTaskId`` payloads to compose.
* Decode the SDK's ``rc`` return into a label.  The
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` table
  (surfaced by :mod:`~strands_robots.tools.g1.g1_error_codes`)
  names every SDK return code the driver's method may quote;
  restating the label here would fork the same table.
* Restate the driver's refusal wording.  Whatever text the
  driver's method writes (an rc-decoded sentence, a gate refusal,
  a no-wave-hand-loco-yet message like ``g1_start_task``'s
  registry-not-wired one) passes through this verb; a verbatim
  quote here would trap the verb to one release's prose (refs
  strands-labs/robots#2874).
* Compose the ``SetTaskId`` payload id.  The driver's own path is
  where the composition lives, so a caller reaching this verb
  passes only the boolean.
* Chain a companion FSM transition.  The neon bundle's docstring
  named ``WaveHand`` as an FSM-agnostic dispatch (``SetTaskId``
  admits from every FSM) but observed the behaviour varies with
  the current FSM; a caller who wants a particular pose before
  the wave issues the two calls in order and reads each envelope
  (``g1_set_fsm`` refs strands-labs/robots#3025).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


def _refusal_envelope(text: str) -> dict[str, Any]:
    """Return the ``@tool`` error envelope with ``text`` inside.

    Kept as a small free helper so every ``turn_flag`` refusal path
    in this module renders the same shape a caller can grep for,
    matching the driver's own
    :func:`~strands_robots.drivers.g1._refuse` free function on the
    write side and the shape ``g1_balance_stand``'s
    :func:`_refusal_envelope` sibling helper renders.
    """
    return {"status": "error", "content": [{"text": text}]}


@tool
def g1_wave_hand_loco(
    driver: Any,
    turn_flag: bool | None = None,
) -> dict[str, Any]:
    """Dispatch the G1's built-in ``LocoClient.WaveHand`` task.

    Calls ``G1Driver.wave_hand_loco`` once and returns the envelope
    the driver produced verbatim.  The driver's method routes to
    ``LocoClient.WaveHand`` (which internally composes a
    ``SetTaskId`` payload); a caller who passes ``False`` reaches
    the wave-in-place task the neon bundle observed and a caller
    who passes ``True`` reaches the wave-and-turn-around task.
    The ``turn_flag`` entry below names both variants; a caller
    planning the write reads it first and reaches this
    verb once the target ``turn_flag`` is decided.

    ``WaveHand`` dispatches through ``SetTaskId`` rather than the
    arm-SDK path ``send_action`` / ``run_policy`` walk, so it does
    not read the driver's
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    admission set (refs strands-labs/robots#2916): the neon
    bundle's docstring named the verb as "does not require FSM
    500+ because it uses SetTaskId", though the *observed*
    behaviour of the resulting wave depends on the FSM the
    controller was already in.  A caller who wants a particular
    pose before the wave issues a companion ``g1_set_fsm`` call
    (refs strands-labs/robots#3025) in order and reads each
    envelope; this verb does not chain the FSM transition itself.

    Args:
        driver: An object with a callable
            ``wave_hand_loco(turn_flag)`` returning the driver's
            write envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to
            keep this module out of the import cycle the driver's
            own :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note.  The verb is duck-
            typed on ``wave_hand_loco``; any object with that
            method returning the envelope shape the driver writes
            will satisfy it.  On today's driver ``wave_hand_loco``
            is not yet exposed, so the :func:`live_handle_refusal`
            grader refuses the call with a message naming the
            accessor; once the driver's method lands (the SDK-
            facing gate work in strands-labs/robots#358), the
            same call returns the driver's envelope verbatim.
        turn_flag: The boolean turn-flag argument
            ``LocoClient.WaveHand`` admits.  The neon bundle
            observed two variants: ``False`` (wave-in-place, the SDK's
            wave-only ``SetTaskId`` composition) and ``True``
            (wave-and-turn-around).  A non-``bool`` payload is
            refused as a shape error rather than resolved through
            Python's ``bool()`` coercion (the neon wrapper called
            ``bool(turn)`` before the SDK saw the value, silently
            transforming ``1`` / ``"yes"`` / ``None`` into an
            admitted task id; refusing the shape here makes the
            boolean requirement decidable at the tool surface
            rather than at wire time).  ``None`` is refused
            because there is no defensible default: the two
            admitted variants are the two data points the
            envelope module surfaces, and a caller who did not
            pass one has not decided the write.  The verb does
            not consult the envelope's ``admitted`` payload; both
            ``False`` and ``True`` are admitted today and a
            firmware release that narrowed the set would land on
            the envelope module and this verb would pass the
            narrower refusal through the driver.

    Returns:
        The envelope ``G1Driver.wave_hand_loco`` returned.  On
        the success path the driver's method will surface the
        SDK's outcome inside a ``{"status": "success", "content":
        [{"json": {"rc": 0, "message": "WaveHand(turn=False)
        dispatched", ...}}]}`` envelope (or whatever richer shape
        the driver settles on); on the driver's refusal path (an
        SDK-side raise, a caller running this verb before the
        driver's method lands, an SDK ``rc`` other than ``0``) it
        is ``{"status": "error", "content": [{"text": "..."}]}``
        with the driver's own reason inside.  The verb does not
        reshape either shape - a future field the driver adds on
        the success path reaches a caller the moment the driver
        writes it, because this verb passes the envelope through.
    """
    # The handle is a live Python object typed :class:`~typing.Any`
    # (see the module docstring's import-cycle note), so the tool
    # schema carries no signal that ``None`` or a robot *name* is
    # refused.  The shared ``live_handle_refusal`` guard is the
    # one implementation of that judgement for this package; it is
    # keyed on the accessor the verb reads, which for this verb is
    # ``wave_hand_loco`` (a callable that calls
    # ``LocoClient.WaveHand`` and returns the driver's envelope)
    # rather than the sensor verbs' ``_snapshot``.  Returning its
    # refusal envelope here rather than raising keeps the four
    # invariants every ``@tool`` handler owes a caller (envelope
    # not exception, names the verb, names ``driver``, names the
    # type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_wave_hand_loco",
        driver,
        accessor="wave_hand_loco",
        reads=(
            "the verb dispatches the G1's built-in LocoClient.WaveHand "
            "task through the driver's own SDK-facing write path and "
            "reads back the envelope the driver produced"
        ),
        expected=(
            "a callable ``wave_hand_loco(turn_flag)`` returning the "
            "driver's write envelope - pass the live G1Driver handle "
            "the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    # ``turn_flag`` is a data parameter the tool schema *does*
    # describe (a boolean naming which of the two ``WaveHand``
    # ``SetTaskId`` compositions to dispatch), so a model can
    # synthesize the wrong shape here as easily as it can reach
    # the verb with the right one.  The two refusals below cover
    # the shapes the neon wrapper silently coerced through
    # ``bool(turn)``: a ``None`` payload (no default is
    # defensible - the two admitted variants are the two data
    # points and a caller who did not pass one has not decided
    # the write) and a non-``bool`` shape (``int`` / ``float`` /
    # ``str`` / ``None``-comparable, which ``bool()`` transforms
    # into an admitted task id without the caller having named
    # the variant on purpose).  Naming ``turn_flag`` here keeps
    # the four invariants: envelope not exception, names the
    # verb, names the parameter, names the shape received.  The
    # in-set admission is not enforced here - both ``False`` and
    # ``True`` are admitted today and the driver's own write path
    # owns the admission set the same way ``g1_balance_stand``'s
    # ``{0, 3}`` admission does.
    if turn_flag is None:
        return _refusal_envelope(
            "g1_wave_hand_loco: `turn_flag` is required. Pass a "
            "bool (False = wave in place, True = wave and turn "
            "around) - see `g1_list_wave_hand_turn_flags` for the "
            "read-only descriptors (refs strands-labs/robots#358)."
        )
    # ``bool`` is refused before ``isinstance(..., bool)`` accepts
    # it, because a caller passing ``1`` (``int``) reaches the SDK's
    # dispatcher through the neon wrapper's silent ``bool()`` coercion
    # as an admitted task id.  Refusing the non-``bool`` shape
    # explicitly matches the neon verb's
    # own ``turn_flag must be bool`` refusal, so both paths render the
    # same shape a caller can grep for and the boolean requirement is
    # decidable at the tool surface rather than at wire time.
    if not isinstance(turn_flag, bool):
        return _refusal_envelope(
            f"g1_wave_hand_loco: `turn_flag` must be a bool (True = "
            f"wave and turn around, False = wave in place), got "
            f"{type(turn_flag).__name__!r}. See "
            f"`g1_list_wave_hand_turn_flags` for the read-only "
            f"descriptors (refs strands-labs/robots#358)."
        )

    return driver.wave_hand_loco(turn_flag)
