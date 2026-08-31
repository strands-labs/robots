"""Agent-facing wrapper for ``G1Driver.release_arm``.

``G1Driver.release_arm`` is the driver-side release-arm entry point: a
caller invokes the verb and the driver publishes the SDK's
:meth:`~unitree_sdk2py.g1.arm.g1_arm_action_client.G1ArmActionClient.ExecuteAction`
call over the same DDS singleton
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` opens, using the
release-arm action id (``99``) the SDK's ``action_map`` reserves for
"drop the arm-action hold and let the driver's ``send_action`` path
resume". The action-id lookup lives one file over in
:mod:`~strands_robots.tools.g1.g1_arm_actions` where
:data:`~strands_robots.tools.g1.g1_arm_actions._ARM_RELEASE_ACTION_ID`
names the number and the neon bundle's ``g1_release_arm`` verb (a
single-purpose ``ExecuteAction(99)`` wrapper) is called out as the
verb the release-side driver method will front.  That lookup is the
one source of truth for the id; this verb does not re-name it.

The driver's method itself is not yet plumbed on
:class:`~strands_robots.drivers.g1.G1Driver` today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on); the accessor grades this verb through
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`, so a
handle without a ``release_arm`` method refuses with a message naming
the verb, the ``driver`` parameter and the accessor it read for, and
once the driver lands the same call returns the envelope the driver
wrote verbatim.  This is the same shape ``g1_set_stand_height`` (refs
strands-labs/robots#3031), ``g1_set_swing_height`` (refs
strands-labs/robots#3032), ``g1_start_task`` and ``g1_send_action``
already ship.

This module is a thin duck-typed wrapper.  It reads the driver
through ``driver.release_arm`` (the one method the underlying driver
exposes for this write) and returns the envelope the driver produced
verbatim.  A future field the driver adds on the success path (say,
an ``rc`` from the SDK's ``ExecuteAction`` handler, a sequence number
on the ``rt/armsdk`` write, or the FSM the release settled into)
reaches a caller the moment the driver writes it, because this verb
does not restate the shape.  What it does add is the same live-handle
refusal every ``@tool`` handler in this package owes its callers
instead of an exception (``driver`` is ``None`` or a robot *name* or
an object without the accessor); the driver's own refusal - the
holding-code (``rc=7401``) surfaced when a release was issued on an
arm that was never holding, the topic-busy code (``rc=7400``)
surfaced when two callers issued releases concurrently, an SDK-side
raise - round-trips through this verb verbatim.

The FSM gate is not consulted here.  The driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` is
the arm-write gate (``send_action`` / ``run_policy`` /
``start_task``); a release is the *end* of the arm-write window a
prior action opened, not a new arm-write frame, so admitting a
release under a widened gate (or refusing one under a narrower gate)
would leave the arm holding when the driver's own release path
already knows the exact frame the SDK accepts.  A second gate call
here would double the read against a cache the driver's FSM
refresher fills, and would refuse releases the driver's own path
admits.  Restating any of that on this side would be a second source
of truth for a rule the driver's own path already enforces (refs
strands-labs/robots#358, strands-labs/robots#2916).  ``import
strands_robots.tools.g1.g1_release_arm`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_set_stand_height``,
``g1_start_task``, ``g1_run_policy``, ``g1_send_action`` and
``g1_stop_task`` give: the driver module imports
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` from this
package at load, so a runtime import of ``G1Driver`` here would
close a cycle, and ``@tool`` calls :func:`typing.get_type_hints` at
decoration time so a string forward reference cannot resolve
without pulling the driver at import.  The verb is duck-typed on
``release_arm``; any object with a synchronous ``release_arm()``
returning the driver's envelope satisfies it, which is also how the
tests hand it a hand-rolled double.

What this module does not do.

* Reach ``G1ArmActionClient.ExecuteAction`` itself.  The neon
  bundle's ``g1_release_arm`` verb held its own single-writer lock
  and called ``arm.ExecuteAction(99)`` inline; that write is the
  ``rt/armsdk`` topic, which the driver's own write path owns (refs
  strands-labs/robots#2916 for the FSM producer wired for the
  arm-SDK gate).  A second writer path here would let two callers
  serialise against a lock the driver's own path does not observe,
  and the SDK's topic-busy refusal (``rc=7400``) would fire on the
  driver's write when this verb's write was the first one on the
  topic.  The verb reaches the driver's method instead so the SDK's
  single-writer answer is one place, not two.
* Encode the action id.  The release-arm action id (``99``) is a
  module-level constant in
  :mod:`~strands_robots.tools.g1.g1_arm_actions` and the driver's
  own ``G1Driver.release_arm`` will name the same number in its
  write.  Restating it here would fork
  the id between two files; a caller who wanted to see the id ahead
  of a release reaches
  :func:`~strands_robots.tools.g1.g1_arm_actions.g1_list_arm_actions`
  where the map already names ``release-arm`` -> ``99`` (refs
  strands-labs/robots#2959).
* Decide whether the arm is currently holding.  The SDK's
  ``ExecuteAction(99)`` handler admits a release on any arm state
  (a release on an arm that was never holding is a no-op that
  returns ``rc=0``; the neon bundle's docstring said "safe to call
  anytime arm is initialized"); the driver's own release path
  routes accordingly and the caller reads the driver's envelope for
  the outcome.  A caller who wants to know the holding state
  *before* a release reaches
  :func:`~strands_robots.tools.g1.g1_state.g1_get_state` where the
  driver's cached ``mode_machine`` and the arm-SDK gate's
  admission window are already surfaced (refs
  strands-labs/robots#2916).
* Quote the SDK's exact refusal wording.  The neon bundle's
  docstring named ``rc=`` codes and the ``decode_code`` helper
  translated them to English; the driver's own path already renders
  the same table through
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`, so a re-
  quote here would trap the verb to one release's prose (refs
  strands-labs/robots#2874).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


@tool
def g1_release_arm(driver: Any) -> dict[str, Any]:
    """Force-release the G1 arm's holding action.

    Calls ``G1Driver.release_arm`` once and returns the envelope
    the driver produced verbatim.  The
    driver's method publishes the SDK's
    :meth:`~unitree_sdk2py.g1.arm.g1_arm_action_client.G1ArmActionClient.ExecuteAction`
    call with action id ``99`` (the id
    :data:`~strands_robots.tools.g1.g1_arm_actions._ARM_RELEASE_ACTION_ID`
    names in the module the arm-action lookup lives in, refs
    strands-labs/robots#2959) which the SDK's ``action_map`` reserves
    for "drop the arm-action hold and let the driver's
    ``send_action`` path resume".  The neon bundle's
    ``g1_release_arm`` verb documented the same one-shot contract
    ("safe to call anytime arm is initialized"); this verb ports it
    unchanged so a caller upgrading from the neon bundle reaches the
    same behaviour.

    A release is the *end* of the arm-write window a prior
    ``g1_send_action`` / ``g1_arm_action`` opened, not a new arm-
    write frame.  A caller who ran an arm action that left the arm
    holding (the SDK's own behaviour on gesture actions like ``11``
    two-hand kiss or ``17`` clap) issues this verb to release the
    hold before the next ``send_action`` write; the SDK's holding-
    code refusal (``rc=7401``, "Arm is holding - release first
    (id=99)"; refs
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`) is
    exactly the refusal a caller sees when they skip the release
    between two arm writes.

    Args:
        driver: An object with a callable ``release_arm()``
            returning the driver's write envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep
            this module out of the import cycle the driver's own
            :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note.  The verb is duck-
            typed on ``release_arm``; any object with that method
            returning the envelope shape the driver writes will
            satisfy it.  On today's driver ``release_arm`` is not
            yet exposed, so the
            :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
            grader refuses the call with a message naming the
            accessor; once the driver's method lands (the SDK-
            facing gate work in strands-labs/robots#358), the same
            call returns the driver's envelope verbatim.

    Returns:
        The envelope ``G1Driver.release_arm`` returned.  On the
        success path the driver's method will surface the SDK's
        ``rc`` inside a ``{"status": "success", "content": [{"json":
        {"rc": 0, "message": "Release arm rc=0 (OK)"}}]}`` envelope;
        on the driver's refusal path (an SDK-side raise, a caller
        running this verb before the driver's method lands, the
        SDK's own ``rc=7400`` topic-busy refusal, an ``rc=7401``
        released-when-not-holding path the SDK admits as a no-op) it
        is ``{"status": "error", "content": [{"text": "..."}]}`` with
        the driver's own reason inside.  The verb does not reshape
        either shape - a future field the driver adds on the success
        path (a settled FSM id, the ``mode_machine`` value the
        release resolved to) reaches a caller the moment the driver
        writes it, because this verb passes the envelope through.
    """
    # The handle is a live Python object typed :class:`~typing.Any`
    # (see the module docstring's import-cycle note), so the tool
    # schema carries no signal that ``None`` or a robot *name* is
    # refused.  The shared ``live_handle_refusal`` guard is the one
    # implementation of that judgement for this package; it is keyed
    # on the accessor the verb reads, which for this verb is
    # ``release_arm`` (a callable that calls the SDK's
    # ``ExecuteAction(99)`` and returns the driver's envelope)
    # rather than the sensor verbs' ``_snapshot``.  Returning its
    # refusal envelope here rather than raising keeps the four
    # invariants every ``@tool`` handler owes a caller (envelope
    # not exception, names the verb, names ``driver``, names the
    # type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_release_arm",
        driver,
        accessor="release_arm",
        reads=(
            "the verb requests a G1 arm-release write through the "
            "driver's own SDK-facing write path and reads back the "
            "envelope the driver produced"
        ),
        expected=(
            "a callable ``release_arm()`` returning the driver's "
            "write envelope - pass the live G1Driver handle the "
            "orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    return driver.release_arm()
