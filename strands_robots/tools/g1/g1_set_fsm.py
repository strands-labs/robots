"""Agent-facing wrapper for ``G1Driver.set_fsm``.

``G1Driver.set_fsm`` is the driver-side ``SetFsmId`` entry point: a
caller passes a target FSM id (``1`` Damp, ``500`` Start, ``501`` Walk,
``801`` BalanceExpert, ``3`` Sit, ``0`` ZeroTorque, ``2`` Squat2Stand,
``4`` Locomotion, ``706`` BalanceLie, ``802`` DampToBalance) and the
driver publishes the SDK's :class:`~unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.SetFsmId`
call over the same DDS singleton :func:`ensure_dds` opens, waits for
the transition to settle, and reads the driver's own live ``fsm_id``
back to surface the fsm-before / fsm-after round-trip the neon bundle's
``g1_set_fsm`` verb documented (a transition the SDK refused silently
still shows the fsm-after equal to fsm-before). The id set the SDK
admits is named in this module's ``fsm_id`` docstring; this verb is the
write half of that conversation.

The driver's method itself is not yet plumbed on
:class:`~strands_robots.drivers.g1.G1Driver` today (refs
strands-labs/robots#358 for the SDK-facing gate work the write belongs
on); the accessor grades this verb through
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`, so a
handle without a ``set_fsm`` method refuses with a message naming the
verb, the ``driver`` parameter and the accessor it read for, and once
the driver lands the same call returns the envelope the driver wrote
verbatim. This is the same shape ``g1_start_task`` (whose driver
method refuses with a registry-not-wired string today) already ships.

This module is a thin duck-typed wrapper. It reads the driver through
``driver.set_fsm`` and returns the envelope the driver produced
verbatim. A future field the driver adds on the success path (say, a
``fsm_before`` / ``fsm_after`` pair, an ``rc`` from the SDK's
``SetFsmId`` handler, a decoded ``rc=7302`` "Invalid FSM id" the
:mod:`~strands_robots.tools.g1.g1_error_codes` lookup already names)
reaches a caller the moment the driver writes it, because this verb
does not restate the shape. What it does add is the two refusal
envelopes every ``@tool`` handler in this package owes its callers
instead of an exception: a live-handle refusal (``driver`` is
``None`` or a robot *name* or an object without ``set_fsm``) and the
driver's own refusal surfaced verbatim.

The FSM gate is not consulted here. The driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` is
the one gate for arm-SDK writes (``send_action`` / ``run_policy`` /
``start_task``); ``SetFsmId`` sits on the loco side of the same DDS
singleton and admits transitions the arm-write gate refuses (the arm
gate admits ``{500, 501, 801}``, the SetFsmId handler admits ``{0, 1,
2, 3, 4, 500, 501, 706, 801, 802}``). A second gate call here would
double the read against a cache the driver's FSM refresher fills, and
would refuse targets the driver's own path admits. Restating any of
that on this side would be a second source of truth for a rule the
driver's own path already enforces (refs strands-labs/robots#358,
strands-labs/robots#2916). ``import
strands_robots.tools.g1.g1_set_fsm`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_start_task``,
``g1_run_policy``, ``g1_send_action`` and ``g1_stop_task`` give: the
driver module imports :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
from this package at load, so a runtime import of ``G1Driver`` here
would close a cycle, and ``@tool`` calls :func:`typing.get_type_hints`
at decoration time so a string forward reference cannot resolve
without pulling the driver at import. The verb is duck-typed on
``set_fsm``; any object with a synchronous
``set_fsm(fsm_id, wait=...)`` returning the driver's envelope
satisfies it, which is also how the tests hand it a hand-rolled
double.

What this module does not do.

* Decide which FSM ids the SDK admits. The driver's own ``set_fsm``
  owns the admission set; a caller who wants to know whether the target
  is reachable reads the snapshot in the ``fsm_id`` docstring below, and
  this verb writes the transition once the target is known. A second
  admission table here would fork that source
  of truth and drift as the SDK's admissions
  change.
* Decode ``rc=7302`` "Invalid FSM id" into a label. The
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` table (surfaced
  by :mod:`~strands_robots.tools.g1.g1_error_codes`) names every SDK
  return code the driver's method may quote; restating the label here
  would fork the same table.
* Warn about safety. The neon bundle's ``g1_set_fsm`` verb emitted a
  free-form "ZeroTorque collapses off-gantry" sentence into the
  message; the driver's own gate / dangerous-target set is where a
  refusal for that shape belongs. A caller running this verb from an
  agent that already read the id snapshot has been warned; a caller who
  jumped past it gets the driver's own admission wording
  rather than a duplicate here.
* Restate the driver's refusal wording. Whatever text the driver's
  method writes (a rc-decoded sentence, a gate refusal, a
  no-set-fsm-yet message like ``g1_start_task``'s
  registry-not-wired one) passes through this verb; a verbatim quote
  here would trap the verb to one release's prose (refs
  strands-labs/robots#2874).
* Schedule transitions. ``set_fsm`` is one call; a caller who wants a
  sequence (Damp → Squat → Start) issues them one at a time and reads
  ``fsm_after`` after each. The SDK's ``SetFsmId`` handler is not
  re-entrant and neon's own bundle held a single-writer lock; when
  the driver lands its ``set_fsm`` method it owns that
  serialisation, not this verb.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal
from strands_robots.utils import positive_finite_number_error


def _refusal_envelope(text: str) -> dict[str, Any]:
    """Return the ``@tool`` error envelope with ``text`` inside.

    Kept as a small free helper so every ``fsm_id`` / ``wait`` refusal
    path in this module renders the same shape a caller can grep for,
    matching the driver's own :func:`~strands_robots.drivers.g1._refuse`
    free function on the write side.
    """
    return {"status": "error", "content": [{"text": text}]}


@tool
def g1_set_fsm(
    driver: Any,
    fsm_id: int | None = None,
    wait: float = 3.0,
) -> dict[str, Any]:
    """Transition the G1 to a target FSM id and read the transition back.

    Calls ``G1Driver.set_fsm`` once and
    returns the envelope the driver produced verbatim. The driver's
    method calls the SDK's ``LocoClient.SetFsmId(fsm_id)`` under its
    own single-writer serialisation, waits ``wait`` seconds for the
    transition to settle, and surfaces a ``fsm_before`` /
    ``fsm_after`` / ``rc`` round-trip so a caller reading the envelope
    can see whether the SDK admitted the write (the SDK's
    ``SetFsmId`` handler returns ``rc=7302`` for ids outside its
    admission set and can also silently discard an id it admits but
    the physical state does not; the round-trip surfaces both, and
    :mod:`~strands_robots.tools.g1.g1_error_codes` is the rc
    catalogue).

    The transition is a loco-side write, not an arm-SDK write; the
    driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    gate is the arm-write gate (``send_action`` / ``run_policy`` /
    ``start_task``) and does not admit or refuse this verb (refs
    strands-labs/robots#2916). A caller who wants to know whether the
    target is reachable at all reads the id snapshot in the ``fsm_id``
    entry below; this verb is the write once the target is known.

    Args:
        driver: An object with a callable ``set_fsm(fsm_id, wait=...)``
            returning the driver's write envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`). Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep
            this module out of the import cycle the driver's own
            :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note. The verb is duck-typed
            on ``set_fsm``; any object with that method returning the
            envelope shape the driver writes will satisfy it. On
            today's driver ``set_fsm`` is not yet exposed, so the
            :func:`live_handle_refusal` grader refuses the call with
            a message naming the accessor; once the driver's method
            lands (the SDK-facing gate work in
            strands-labs/robots#358), the same call returns the
            driver's envelope verbatim.
        fsm_id: An FSM id the SDK's ``SetFsmId`` handler admits (the
            snapshot the neon bundle recorded: ``1`` Damp, ``500``
            Start, ``501`` Walk, ``801``
            BalanceExpert, ``3`` Sit, ``0`` ZeroTorque, ``2``
            Squat2Stand, ``4`` Locomotion, ``706`` BalanceLie, ``802``
            DampToBalance). The driver's own ``set_fsm`` refuses ids
            outside that set at wire time with the SDK's own ``rc=7302``
            envelope; this verb refuses a ``None`` or non-int shape
            here rather than on the driver so the refusal names
            ``fsm_id`` and the remedy. The id set a caller wants
            before issuing the write is the snapshot above.
        wait: Seconds to wait after ``SetFsmId`` returns before reading
            ``fsm_after``. Defaults to ``3.0`` seconds - the neon
            bundle's own default, chosen to let the slowest documented
            transition (Damp -> Start requires a leg extension the
            robot walks through mechanically) settle before the
            second FSM read. Validated against
            :func:`~strands_robots.utils.positive_finite_number_error`
            because a fractional value is usable here (a caller who
            already saw the target FSM live in ``fsm_before`` may
            want a shorter read-back) and ``nan`` / ``inf`` /
            negative / non-numeric shapes poison the driver's own
            :func:`time.sleep` call.

    Returns:
        The envelope ``G1Driver.set_fsm`` returned. On the success
        path the driver's method will surface the fsm-before /
        fsm-after / rc / message round-trip inside a ``{"status":
        "success", "content": [{"json": {"fsm_before": ...,
        "fsm_after": ..., "rc": ..., "message": ...}}]}`` envelope;
        on the driver's refusal path (an id outside the SDK's admission
        set, an SDK-side raise, a caller running this verb before the
        driver's method lands) it is ``{"status": "error", "content":
        [{"text": "..."}]}`` with the driver's own reason inside. The
        verb does not reshape either shape - a future field the driver
        adds on the success path reaches a caller the moment the
        driver writes it, because this verb passes the envelope
        through.
    """
    # The handle is a live Python object typed :class:`~typing.Any`
    # (see the module docstring's import-cycle note), so the tool
    # schema carries no signal that ``None`` or a robot *name* is
    # refused. The shared ``live_handle_refusal`` guard is the one
    # implementation of that judgement for this package; it is keyed
    # on the accessor the verb reads, which for this verb is
    # ``set_fsm`` (a callable that calls the SDK's ``SetFsmId`` and
    # returns the driver's envelope) rather than the sensor verbs'
    # ``_snapshot``. Returning its refusal envelope here rather than
    # raising keeps the four invariants every ``@tool`` handler owes a
    # caller (envelope not exception, names the verb, names ``driver``,
    # names the type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_set_fsm",
        driver,
        accessor="set_fsm",
        reads=(
            "the verb requests a G1 FSM transition through the driver's "
            "own SDK-facing write path and reads back the fsm-before / "
            "fsm-after round-trip the driver produced"
        ),
        expected=(
            "a callable ``set_fsm(fsm_id, wait=...)`` returning the "
            "driver's write envelope - pass the live G1Driver handle "
            "the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    # ``fsm_id`` is a data parameter the tool schema *does* describe
    # (an integer id from the SDK's ``SetFsmId`` admission set), so a
    # model can synthesize the wrong shape here as easily as it can
    # reach the verb with the right one. These three refusals cover
    # the shapes the driver's own ``set_fsm`` would surface as an
    # inner refusal naming an id (a ``None`` payload, a non-int shape,
    # a boolean subclass whose ``True`` would act as a silent ``1``).
    # Naming ``fsm_id`` here keeps the four invariants: envelope not
    # exception, names the verb, names the parameter, names the shape
    # received.
    if fsm_id is None:
        return _refusal_envelope(
            "g1_set_fsm: `fsm_id` is required. Pass an integer id from "
            "the SDK's SetFsmId admission set - see "
            "strands_robots.tools.g1.g1_fsm_targets for the snapshot "
            "(refs strands-labs/robots#358)."
        )
    if isinstance(fsm_id, bool) or not isinstance(fsm_id, int):
        return _refusal_envelope(
            f"g1_set_fsm: `fsm_id` of type {type(fsm_id).__name__!r} is "
            "not an int. Pass an integer id from the SDK's SetFsmId "
            "admission set - see strands_robots.tools.g1.g1_fsm_targets "
            "for the snapshot (refs strands-labs/robots#358)."
        )

    # ``wait`` is the seconds-to-sleep between the ``SetFsmId`` write
    # and the ``fsm_after`` read. The shared validator refuses the
    # same domain :meth:`G1Driver.run_policy`'s ``duration`` refuses
    # (``nan`` poisons every comparison it reaches, ``inf`` collapses
    # the sleep to unbounded, negative reverses the sleep's contract,
    # a non-numeric shape raises out of a call that must return an
    # envelope). Refusing here surfaces the parameter name and the
    # remedy in one envelope; refusing on the driver would name the
    # driver's own call site.
    wait_error = positive_finite_number_error(
        wait, "wait", "the seconds to wait between SetFsmId and the fsm-after read"
    )
    if wait_error is not None:
        return _refusal_envelope(f"g1_set_fsm: {wait_error}")

    return driver.set_fsm(fsm_id, wait=wait)
