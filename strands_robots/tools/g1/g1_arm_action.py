"""Agent-facing wrapper for ``G1Driver.arm_action``.

``G1Driver.arm_action`` is the driver-side arm-action entry point:
a caller invokes the verb with an action name (``"clap"``, ``"heart"``,
``"two-hand kiss"``) or a numeric ``action_id`` and the driver publishes
the SDK's
:meth:`~unitree_sdk2py.g1.arm.g1_arm_action_client.G1ArmActionClient.ExecuteAction`
call over the same DDS singleton
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` opens, using
the id the SDK's ``action_map`` reserves for that gesture.  The
action-id lookup lives one file over in
:mod:`~strands_robots.tools.g1.g1_arm_actions` where the
:data:`~strands_robots.tools.g1.g1_arm_actions._ARM_ACTION_MAP`
name-to-id table is the one source of truth for the numbers; this
verb does not re-name them, and a caller that wants the list ahead
of a write reaches
:func:`~strands_robots.tools.g1.g1_arm_actions.g1_list_arm_actions`
where the same map is already surfaced (refs
strands-labs/robots#2959).

The driver's method itself is not yet plumbed on
:class:`~strands_robots.drivers.g1.G1Driver` today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on); the accessor grades this verb through
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`, so a
handle without an ``arm_action`` method refuses with a message
naming the verb, the ``driver`` parameter and the accessor it read
for, and once the driver lands the same call returns the envelope
the driver wrote verbatim.  This is the same shape ``g1_release_arm``
(refs strands-labs/robots#3034), ``g1_send_action`` (refs
strands-labs/robots#3004), ``g1_start_task`` (refs
strands-labs/robots#3016) and ``g1_set_stand_height`` (refs
strands-labs/robots#3031) already ship.

This module is a thin duck-typed wrapper.  It reads the driver
through ``driver.arm_action`` (the one method the underlying driver
exposes for this write) and returns the envelope the driver produced
verbatim.  A future field the driver adds on the success path (a
sequence number on the ``rt/armsdk`` write, the FSM the gesture
settled into, the ``rc`` from the SDK's ``ExecuteAction`` handler)
reaches a caller the moment the driver writes it, because this verb
does not restate the shape.  What it does add is the two refusal
envelopes every ``@tool`` handler in this package owes its callers
instead of an exception: a live-handle refusal (``driver`` is
``None`` or a robot *name* or an object without the accessor), and a
name/id refusal when neither ``action`` nor ``action_id`` is passed.
The driver's own refusals - the topic-busy code (``rc=7400``)
surfaced when two callers issued arm actions concurrently on the
single-writer ``rt/armsdk`` topic, the SDK-side raise, the FSM /
battery gate that ``_check_motion_gates`` writes with scope
``"arm"`` - round-trip through this verb verbatim.

The FSM / motion gate is not consulted here.  The driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` is
the arm-write gate; a caller running this verb from a supervisor
cannot bypass it by pre-flighting from an agent that read the cache
a second earlier, and the gate's refusal is the same three
sentences the driver writes into every wire method
(``send_action`` / ``run_policy`` / ``start_task``, refs
strands-labs/robots#2916).  Restating any of that on this side
would be a second source of truth for a rule the driver's own path
already enforces.  ``import strands_robots.tools.g1.g1_arm_action``
still pulls no ``unitree_sdk2py`` submodule (the package's
SDK-load-hygiene contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_release_arm``,
``g1_send_action``, ``g1_start_task`` and ``g1_run_policy`` give:
the driver module imports
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` from this
package at load, so a runtime import of ``G1Driver`` here would
close a cycle, and ``@tool`` calls :func:`typing.get_type_hints` at
decoration time so a string forward reference cannot resolve
without pulling the driver at import.  The verb is duck-typed on
``arm_action``; any object with a synchronous
``arm_action(action, action_id, ...)`` returning the driver's
envelope satisfies it, which is also how the tests hand it a
hand-rolled double.

What this module does not do.

* Reach ``G1ArmActionClient.ExecuteAction`` itself.  The neon
  bundle's ``g1_arm_action`` verb held its own single-writer
  ``threading.Lock`` and called ``arm.ExecuteAction(action_id)``
  inline; that write is the ``rt/armsdk`` topic, which the driver's
  own write path owns (refs strands-labs/robots#2916 for the FSM
  producer wired for the arm-SDK gate).  A second writer path here
  would let two callers serialise against a lock the driver's own
  path does not observe, and the SDK's topic-busy refusal
  (``rc=7400``) would fire on the driver's write when this verb's
  write was the first one on the topic.  The verb reaches the
  driver's method instead so the SDK's single-writer answer is one
  place, not two.
* Resolve the action name to an id.  The name-to-id map is a
  module-level constant in
  :mod:`~strands_robots.tools.g1.g1_arm_actions` and the driver's
  own ``G1Driver.arm_action`` will name the same table in its
  write.  Restating it here would fork the map between two files;
  a caller who wants to see the id for a name reaches
  :func:`~strands_robots.tools.g1.g1_arm_actions.g1_list_arm_actions`
  where the map already names every gesture (refs
  strands-labs/robots#2959).  This verb passes ``action`` and
  ``action_id`` through unchanged so the driver's own resolution -
  and its refusal for an unknown name - is the single answer.
* Sleep ``hold_seconds`` then send a release.  The neon bundle's
  ``g1_arm_action`` verb held its own single-writer lock through a
  ``time.sleep(hold_seconds)`` and then issued
  ``ExecuteAction(99)`` inline; on the driver side that hold is a
  concern of the write scheduler (the caller's own timer, or the
  driver's ``run_policy`` loop when the gesture is part of a
  policy), not of a one-frame write.  A caller who wants a hold-
  then-release from a single call reaches the driver's method
  directly, or issues ``g1_release_arm`` (refs
  strands-labs/robots#3034) after the hold from the same schedule
  that fired this verb.
* Quote the SDK's exact refusal wording.  The neon bundle's
  docstring named ``rc=`` codes and the ``decode_code`` helper
  translated them to English; the driver's own path already renders
  the same table through
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`, so a re-
  quote here would trap the verb to one gesture's prose (refs
  strands-labs/robots#2874).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


def _refusal_envelope(text: str) -> dict[str, Any]:
    """Return the ``@tool`` error envelope with ``text`` inside.

    Kept as a small free helper so every parameter-refusal path in
    this module renders the same shape a caller can grep for, matching
    the driver's own :func:`~strands_robots.drivers.g1._refuse` free
    helper.  A caller reading the envelope on the wire sees the same
    ``status`` / ``content[0]["text"]`` shape whether the refusal came
    from the verb's parameter guards or from the driver's own gate.
    """
    return {"status": "error", "content": [{"text": text}]}


@tool
def g1_arm_action(
    driver: Any,
    action: str = "",
    action_id: int | None = None,
) -> dict[str, Any]:
    """Execute one G1 arm gesture through the driver's write path.

    Calls ``G1Driver.arm_action(action, action_id)`` once and returns
    the envelope the driver produced verbatim.  The driver's method
    publishes the SDK's
    :meth:`~unitree_sdk2py.g1.arm.g1_arm_action_client.G1ArmActionClient.ExecuteAction`
    call with the id the SDK's ``action_map`` reserves for the
    gesture (``"clap"`` -> ``17``, ``"heart"`` -> ``20``,
    ``"two-hand kiss"`` -> ``11``, and so on - the full map is the
    :data:`~strands_robots.tools.g1.g1_arm_actions._ARM_ACTION_MAP`
    surface, refs strands-labs/robots#2959).  The neon bundle's
    ``g1_arm_action`` verb documented the same lookup ("action name
    OR numeric id"); this verb ports the entry point unchanged so a
    caller upgrading from the neon bundle reaches the same behaviour
    through the driver's write path instead of a second lock in the
    tool layer.

    Args:
        driver: An object with a callable
            ``arm_action(action, action_id)`` returning the driver's
            write envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep
            this module out of the import cycle the driver's own
            :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note.  The verb is duck-
            typed on ``arm_action``; any object with that method
            returning the envelope shape the driver writes will
            satisfy it.  On today's driver ``arm_action`` is not yet
            exposed, so the
            :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
            grader refuses the call with a message naming the
            accessor; once the driver's method lands (the SDK-facing
            gate work in strands-labs/robots#358), the same call
            returns the driver's envelope verbatim.
        action: The gesture name (``"clap"``, ``"heart"``,
            ``"two-hand kiss"``, ...).  The neon bundle's docstring
            named the map; the driver's own ``arm_action`` resolves
            the name to an id through the same table, so a caller
            who wants to see the full list reaches
            :func:`~strands_robots.tools.g1.g1_arm_actions.g1_list_arm_actions`
            (refs strands-labs/robots#2959).  Ignored when
            ``action_id`` is passed; either one is required.
        action_id: The numeric id (``11`` two-hand kiss, ``17``
            clap, ``20`` heart, ...).  Wins over ``action`` when
            both are passed - matches the neon bundle's precedence
            so a caller pinning by id ahead of a driver-side name
            rename is not caught by the string being wrong.  An
            unknown id round-trips through the driver's own refusal
            (the SDK returns an error code, ``rc != 0``, and the
            driver names it in its envelope); this verb does not
            pre-flight against the map.

    Returns:
        The envelope ``G1Driver.arm_action`` returned.  On the
        success path this is
        ``{"status": "success", "content": [{"json": {"action": ...,
        "action_id": ..., "rc": 0, "message": ...}}]}``; on the
        driver's refusal path (gate flip, publisher not initialised,
        unknown action name or id, SDK missing on the write path,
        publish error, ``rc=7400`` topic-busy from a concurrent
        writer) it is ``{"status": "error", "content": [{"text":
        "..."}]}`` with the driver's own reason inside.  The verb
        does not reshape either shape - a future field the driver
        adds on the success path reaches a caller the moment the
        driver writes it, because this verb passes the envelope
        through.
    """
    # The handle is a live Python object typed :class:`~typing.Any` (see
    # the module docstring's import-cycle note), so the tool schema
    # carries no signal that ``None`` or a robot *name* is refused.  The
    # shared ``live_handle_refusal`` guard is the one implementation of
    # that judgement for this package; it is keyed on the accessor the
    # verb reads, which for this verb is ``arm_action`` (a callable that
    # writes one ``ExecuteAction`` call on ``rt/armsdk`` and returns the
    # driver's envelope) rather than the sensor verbs' ``_snapshot``.
    # Returning its refusal envelope here rather than raising keeps the
    # four invariants every ``@tool`` handler owes a caller (envelope
    # not exception, names the verb, names ``driver``, names the type
    # on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_arm_action",
        driver,
        accessor="arm_action",
        reads=(
            "the verb publishes one arm-gesture ExecuteAction call on "
            "rt/armsdk through the driver's own SDK-facing write path "
            "and reads back the envelope the driver produced"
        ),
        expected=(
            "a callable ``arm_action(action, action_id)`` returning "
            "the driver's write envelope - pass the live G1Driver "
            "handle the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    # ``action`` and ``action_id`` are data parameters the tool schema
    # *does* describe (a string name and an integer id), so a model can
    # synthesize a call with neither as easily as it can reach the verb
    # with one.  This refusal covers the shape the driver's own path
    # would surface as an inner refusal naming an empty name and a
    # ``None`` id, which is a shape a caller reading either parameter
    # alone cannot map back to "the verb needs one or the other".
    # Naming both parameters here keeps the four invariants: envelope
    # not exception, names the verb, names the parameters, names the
    # shape received.
    if not action and action_id is None:
        return _refusal_envelope(
            "g1_arm_action: pass one of `action` (a gesture name like "
            "'clap' or 'heart') or `action_id` (the SDK's numeric id "
            "like 17 or 20). The full name-to-id map is on "
            "g1_list_arm_actions (refs strands-labs/robots#2959)."
        )

    return driver.arm_action(action, action_id)
