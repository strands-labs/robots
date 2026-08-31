"""Agent-facing wrapper for ``G1Driver.stop_move``.

``G1Driver.stop_move`` is the driver-side stop-move entry point: a
caller invokes the verb and the driver publishes the SDK's
:meth:`~unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.StopMove`
call over the same DDS singleton
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` opens. The
Python SDK exposes ``StopMove`` as a public ``LocoClient`` method
that zeroes the last-commanded ``(vx, vy, vyaw)`` velocity triple
without changing the FSM the robot is in; the neon bundle's
``g1_stop_move`` verb (``cagataycali/neon-the-g1/tools/g1_locomotion.py``)
fronted the call under a single-writer lock and returned an rc
envelope. This module is the write-side companion of the velocity
envelope the neon bundle observed on a gantry, which the driver's own
write path owns (refs strands-labs/robots#358, #2965); every
other locomotion verb the neon bundle exposes
(``g1_move_velocity``, ``g1_walk_forward``, ``g1_turn``) hands the
same ``LocoClient`` its argument triple, and this verb hands it the
zero triple.

The driver's method itself is not yet plumbed on
:class:`~strands_robots.drivers.g1.G1Driver` today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on); the accessor grades this verb through
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`, so a
handle without a ``stop_move`` method refuses with a message naming
the verb, the ``driver`` parameter and the accessor it read for, and
once the driver lands the same call returns the envelope the driver
wrote verbatim. This is the same shape ``g1_release_arm`` (refs
strands-labs/robots#3034), ``g1_balance_stand`` (refs
strands-labs/robots#3033), ``g1_set_stand_height`` (refs
strands-labs/robots#3031) and ``g1_set_swing_height`` (refs
strands-labs/robots#3032) already ship.

This module is a thin duck-typed wrapper. It reads the driver
through ``driver.stop_move`` and returns the envelope the driver
produced verbatim. A future field the driver adds on the success
path (say, an ``rc`` from the SDK's ``StopMove`` handler, the FSM
the robot settled in after the zero triple flushed) reaches a caller
the moment the driver writes it, because this verb does not restate
the shape. What it does add is the live-handle refusal every
``@tool`` handler in this package owes its callers instead of an
exception: ``driver`` is ``None`` or a robot *name* or an object
without ``stop_move``; the driver's own refusal round-trips through
this verb verbatim.

The FSM gate is not consulted here. The driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` is
the one gate for arm-SDK writes (``send_action`` / ``run_policy`` /
``start_task``); ``StopMove`` sits on the loco side of the same DDS
singleton and belongs on the locomotion admission set
:data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` the driver's
own path will check when the write lands. A second gate call here
would double the read against a cache the driver's FSM refresher
fills, and would refuse a stop the driver's own path admits. That
is particularly costly on this verb: ``StopMove`` is the emergency-
halt path the neon bundle documented as ``SAFE to call anytime -
doesn't change FSM, just kills movement``, so refusing it under a
narrower gate here would leave the robot walking when a caller
asked it to stop. Restating any of that on this side would be a
second source of truth for a rule the driver's own path already
enforces (refs strands-labs/robots#358, strands-labs/robots#2916).
``import strands_robots.tools.g1.g1_stop_move`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_release_arm``,
``g1_balance_stand``, ``g1_set_stand_height``, ``g1_set_swing_height``,
``g1_start_task``, ``g1_run_policy``, ``g1_send_action`` and
``g1_stop_task`` give: the driver module imports
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` from this
package at load, so a runtime import of ``G1Driver`` here would
close a cycle, and ``@tool`` calls :func:`typing.get_type_hints`
at decoration time so a string forward reference cannot resolve
without pulling the driver at import. The verb is duck-typed on
``stop_move``; any object with a synchronous ``stop_move()``
returning the driver's envelope satisfies it, which is also how the
tests hand it a hand-rolled double.

What this module does not do.

* Reach ``LocoClient.StopMove`` itself. The neon bundle's
  ``g1_stop_move`` verb held its own single-writer lock and called
  ``loco.StopMove()`` inline; that write is the ``rt/lococmd``
  channel, which the driver's own write path owns. A second writer
  path here would let two callers serialise against a lock the
  driver's own path does not observe, and a walking robot needs
  exactly one call to halt - the verb reaches the driver's method
  instead so the SDK's single-writer answer is one place, not two.
* Encode the zero velocity triple. The SDK's ``StopMove`` handler
  is the one that names ``(0.0, 0.0, 0.0)`` as its side-effect on
  the last-commanded velocity; a caller who wanted an explicit
  zero-triple write reaches ``g1_move_velocity`` (once ported) with
  those arguments. Restating the triple here would fork the SDK's
  own contract into a second source of truth.
* Decode the SDK's ``rc`` return into a label. The
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` table
  (surfaced by :mod:`~strands_robots.tools.g1.g1_error_codes`)
  names every SDK return code the driver's method may quote;
  restating the label here would fork the same table.
* Restate the driver's refusal wording. Whatever text the driver's
  method writes (a rc-decoded sentence, a no-stop-move-yet message
  like ``g1_start_task``'s registry-not-wired one) passes through
  this verb; a verbatim quote here would trap the verb to one
  release's prose (refs strands-labs/robots#2874).
* Check the FSM before the stop. The stop is the *end* of a
  walking window; refusing to halt when the driver's FSM cache
  briefly reads a non-walk state (say, during a transition) would
  leave the robot moving when the caller asked it to stop.
  The driver's own stop path is the one that decides admission,
  and it admits stops broader than it admits starts.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


@tool
def g1_stop_move(driver: Any) -> dict[str, Any]:
    """Stop all G1 locomotion (zero velocity triple, FSM unchanged).

    Calls ``G1Driver.stop_move`` once and returns the envelope the
    driver produced verbatim. The driver's method publishes the SDK's
    :meth:`~unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.StopMove`
    call, which the SDK's own handler documents as zeroing the
    last-commanded ``(vx, vy, vyaw)`` velocity triple without
    changing the FSM the robot is in. The neon bundle's
    ``g1_stop_move`` verb documented the same one-shot contract
    (``SAFE to call anytime - doesn't change FSM, just kills
    movement``); this verb ports it unchanged so a caller upgrading
    from the neon bundle reaches the same behaviour.

    The stop is the *end* of a walking window a prior
    ``g1_move_velocity`` / ``g1_walk_forward`` / ``g1_turn`` opened,
    not a new locomotion frame. A caller who commanded a walk with
    an explicit duration issues this verb to halt earlier than the
    duration would expire on its own; the SDK's ``StopMove``
    handler admits the stop on any FSM the robot is in, which is
    the reason the neon bundle called it an emergency-stop path.

    Args:
        driver: An object with a callable ``stop_move()`` returning
            the driver's write envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`). Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to
            keep this module out of the import cycle the driver's
            own :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note. The verb is duck-
            typed on ``stop_move``; any object with that method
            returning the envelope shape the driver writes will
            satisfy it. On today's driver ``stop_move`` is not yet
            exposed, so the
            :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
            grader refuses the call with a message naming the
            accessor; once the driver's method lands (the SDK-
            facing gate work in strands-labs/robots#358), the same
            call returns the driver's envelope verbatim.

    Returns:
        The envelope ``G1Driver.stop_move`` returned. On the
        success path the driver's method will surface the SDK's
        ``rc`` inside a ``{"status": "success", "content":
        [{"json": {"rc": 0, "message": "..."}}]}`` envelope; on
        the driver's refusal path (an SDK-side raise, a caller
        running this verb before the driver's method lands, an
        ``rc=3104`` RPC-timeout wedged handler) it is
        ``{"status": "error", "content": [{"text": "..."}]}`` with
        the driver's own reason inside. The verb does not reshape
        either shape - a future field the driver adds on the
        success path (a settled FSM id, a decoded rc label from
        :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`)
        reaches a caller the moment the driver writes it, because
        this verb passes the envelope through.
    """
    # The handle is a live Python object typed :class:`~typing.Any`
    # (see the module docstring's import-cycle note), so the tool
    # schema carries no signal that ``None`` or a robot *name* is
    # refused. The shared ``live_handle_refusal`` guard is the one
    # implementation of that judgement for this package; it is keyed
    # on the accessor the verb reads, which for this verb is
    # ``stop_move`` (a callable that calls the SDK's ``StopMove``
    # and returns the driver's envelope) rather than the sensor
    # verbs' ``_snapshot``. Returning its refusal envelope here
    # rather than raising keeps the four invariants every ``@tool``
    # handler owes a caller (envelope not exception, names the
    # verb, names ``driver``, names the type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_stop_move",
        driver,
        accessor="stop_move",
        reads=(
            "the verb requests a G1 locomotion halt through the "
            "driver's own SDK-facing write path and reads back the "
            "envelope the driver produced"
        ),
        expected=(
            "a callable ``stop_move()`` returning the driver's "
            "write envelope - pass the live G1Driver handle the "
            "orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    return driver.stop_move()
