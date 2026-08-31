"""Agent-facing wrapper for ``G1Driver.balance_stand``.

``G1Driver.balance_stand`` is the driver-side BalanceStand entry
point: a caller passes a balance-mode id and the driver publishes
:meth:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.BalanceStand`
(which internally reaches the SDK's ``SetBalanceMode`` handler) over
the same DDS singleton :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
opens. The Python SDK exposes ``BalanceStand`` as a public
``LocoClient`` method that admits a small set of pre-programmed
modes: ``0`` (static balance, the default) and ``3`` (dynamic
balance, from the neon bundle's field notes against the real
robot). The neon bundle's ``g1_balance_stand`` verb
(``cagataycali/neon-the-g1/tools/g1_posture.py``) wrapped the call
under a single-writer lock and coerced the argument through
:class:`int` before dispatch; the admitted set that read-only half
described is documented inline below (refs strands-labs/robots#358),
and this module is the write side that hands the target to the
driver.

The driver's method itself is not yet plumbed on
:class:`~strands_robots.drivers.g1.G1Driver` today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on); the accessor grades this verb through
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`, so a
handle without a ``balance_stand`` method refuses with a message
naming the verb, the ``driver`` parameter and the accessor it read
for, and once the driver lands the same call returns the envelope
the driver wrote verbatim. This is the same shape ``g1_set_fsm``
(refs strands-labs/robots#3025), ``g1_set_stand_height`` (refs
strands-labs/robots#3031) and ``g1_set_swing_height`` (refs
strands-labs/robots#3032) already ship.

This module is a thin duck-typed wrapper. It reads the driver
through ``driver.balance_stand`` and returns the envelope the
driver produced verbatim. A future field the driver adds on the
success path (say, an ``rc`` from the SDK's ``SetBalanceMode``
handler, the FSM the controller settled into) reaches a caller the
moment the driver writes it, because this verb does not restate
the shape. What it does add is the two refusal envelopes every
``@tool`` handler in this package owes its callers instead of an
exception: a live-handle refusal (``driver`` is ``None`` or a robot
*name* or an object without ``balance_stand``) and the driver's
own refusal surfaced verbatim.

The FSM gate is not consulted here. The driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` is
the one gate for arm-SDK writes (``send_action`` / ``run_policy`` /
``start_task``); ``BalanceStand`` sits on the loco side of the
same DDS singleton and belongs on the locomotion admission set
:data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` the driver's
own path will check when the write lands. A second gate call here
would double the read against a cache the driver's FSM refresher
fills, and would refuse targets the driver's own path admits.
Restating any of that on this side would be a second source of
truth for a rule the driver's own path already enforces (refs
strands-labs/robots#358, strands-labs/robots#2916). ``import
strands_robots.tools.g1.g1_balance_stand`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_set_fsm``,
``g1_set_stand_height``, ``g1_set_swing_height``, ``g1_start_task``,
``g1_run_policy``, ``g1_send_action`` and ``g1_stop_task`` give:
the driver module imports
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` from this
package at load, so a runtime import of ``G1Driver`` here would
close a cycle, and ``@tool`` calls :func:`typing.get_type_hints`
at decoration time so a string forward reference cannot resolve
without pulling the driver at import. The verb is duck-typed on
``balance_stand``; any object with a synchronous
``balance_stand(balance_mode)`` returning the driver's envelope
satisfies it, which is also how the tests hand it a hand-rolled
double.

What this module does not do.

* Refuse a ``balance_mode`` outside the neon-bundle-observed
  admitted set ``{0, 3}`` - the two modes the neon bundle
  observed as walkable. Refusing an
  unlisted mode here would fork that admission set
  into a second source of truth this module would then have to
  keep in sync with the driver's own gate. The SDK's own handler
  silently accepts an unknown mode and ignores it when outside
  its programmed set, so the driver's method (once landed) is
  where an in-set refusal lives - a caller who passes ``7`` here
  reaches the driver's own refusal, which round-trips through
  this verb verbatim. The bare-int shape checks below refuse the
  cross-type shapes ``bool``/``float``/``str`` the SDK's
  :class:`int` coercion in the neon wrapper silently transformed
  rather than declined; those are separate from the in-set
  admission the envelope module owns.
* Encode the ``LocoClient.BalanceStand`` dispatch. The SDK's
  public method and the neon bundle's single-writer lock are
  driver-side wire concerns; the verb passes ``balance_mode``
  through unchanged and the driver's method decides how to reach
  the SDK's handler. A caller reading this verb's docstring
  learns the ergonomic contract, not the wire format.
* Decode the SDK's ``rc`` return into a label. The
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` table
  (surfaced by :mod:`~strands_robots.tools.g1.g1_error_codes`)
  names every SDK return code the driver's method may quote;
  restating the label here would fork the same table.
* Restate the driver's refusal wording. Whatever text the
  driver's method writes (a rc-decoded sentence, a gate refusal,
  a no-balance-stand-yet message like ``g1_start_task``'s
  registry-not-wired one) passes through this verb; a verbatim
  quote here would trap the verb to one release's prose (refs
  strands-labs/robots#2874).
* Schedule the FSM prerequisite. The neon bundle's docstring
  named ``g1_set_fsm(801)`` as a companion call to reach FSM 801
  (``BalanceExpert``) from other FSMs before ``BalanceStand``
  itself is admitted by the controller; ``g1_set_fsm`` already
  ships as a sibling verb (refs strands-labs/robots#3025). A
  caller who wants that sequence issues the two calls in order
  and reads each envelope; this verb does not chain them.
* Check whether the driver's live ``_fsm_id`` is inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That
  membership test is the driver's own gate concern. The set is
  surfaced to a caller planning the write by ``WALK_FSMS``
  itself, but this verb does not enforce it (the driver's
  method will).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


def _refusal_envelope(text: str) -> dict[str, Any]:
    """Return the ``@tool`` error envelope with ``text`` inside.

    Kept as a small free helper so every ``balance_mode`` refusal
    path in this module renders the same shape a caller can grep
    for, matching the driver's own
    :func:`~strands_robots.drivers.g1._refuse` free function on the
    write side.
    """
    return {"status": "error", "content": [{"text": text}]}


@tool
def g1_balance_stand(
    driver: Any,
    balance_mode: int | None = None,
) -> dict[str, Any]:
    """Enter the G1's BalanceStand state with the given balance mode.

    Calls ``G1Driver.balance_stand`` once and returns the envelope
    the driver produced verbatim. The driver's method routes to
    ``LocoClient.BalanceStand`` (which internally reaches the SDK's
    ``SetBalanceMode`` handler); a caller who passes ``0`` reaches
    the static balance mode (the default the neon bundle
    documented), a caller who passes ``3`` reaches the dynamic
    balance mode the neon bundle observed as walkable, and a value
    outside the neon-bundle-observed ``{0, 3}`` admitted set
    reaches either the driver's own refusal or the SDK's silent-
    accept-and-ignore handler through the verb's pass-through (the
    SDK does not ship a distinct ``rc`` for an unknown mode). The
    neon bundle's own ``g1_balance_stand`` verb documented ``0``
    and ``3`` as the two modes it observed, and this verb ports
    the write side of that contract unchanged so a caller
    upgrading from the neon bundle reaches the same
    behaviour.

    Reaching FSM 801 (``BalanceExpert``) may require a companion
    ``g1_set_fsm(801)`` call before this verb (refs
    strands-labs/robots#3025); the neon bundle documented that
    prerequisite and this verb does not chain the FSM transition
    itself - a caller who wants the sequence issues the two calls
    in order and reads each envelope.

    ``BalanceStand`` is a locomotion-shaped write; the driver's
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    gate will (once the driver's method lands) refuse the write
    while ``_fsm_id`` is outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` (refs
    strands-labs/robots#2916). A caller planning the write compares
    the driver's live ``fsm_id`` (from ``get_status``) against the
    walk-ready ids before reaching this verb; this verb does not
    re-run that check itself.

    Args:
        driver: An object with a callable
            ``balance_stand(balance_mode)`` returning the driver's
            write envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`). Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to
            keep this module out of the import cycle the driver's
            own :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note. The verb is duck-
            typed on ``balance_stand``; any object with that
            method returning the envelope shape the driver writes
            will satisfy it. On today's driver ``balance_stand``
            is not yet exposed, so the :func:`live_handle_refusal`
            grader refuses the call with a message naming the
            accessor; once the driver's method lands (the SDK-
            facing gate work in strands-labs/robots#358), the
            same call returns the driver's envelope verbatim.
        balance_mode: The balance-mode id
            ``LocoClient.BalanceStand`` admits. The neon bundle
            observed two walkable modes: ``0`` (static balance,
            the default the neon bundle documented) and ``3``
            (dynamic balance).
            This verb does not refuse a mode outside that set -
            the driver's method (once landed) is where an in-set
            refusal lives, and a caller who passes ``7`` reaches
            either the driver's own refusal or the SDK's silent-
            accept-and-ignore handler through the verb's pass-
            through. The shape checks below refuse a ``None``
            payload, a ``bool`` payload (``bool`` is an ``int``
            subclass, so ``True`` would act as a silent ``1``
            and ``False`` as a silent ``0`` - neither is a mode
            id a caller writing the boolean would have named on
            purpose), and a non-integer shape (``float``,
            ``str``) that the neon bundle's own ``int(...)``
            coercion silently transformed rather than declined.
            Those are shape refusals, not domain refusals; the
            in-set admission belongs on the driver's write path
            and the read-only envelope module the same way
            ``g1_set_swing_height``'s ``0.05..0.15`` typical
            range belongs on the driver rather than this verb.

    Returns:
        The envelope ``G1Driver.balance_stand`` returned. On the
        success path the driver's method will surface the SDK's
        outcome inside a ``{"status": "success", "content":
        [{"json": {"rc": 0, "message": "BalanceStand(mode=0)
        dispatched"}}]}`` envelope (or whatever richer shape the
        driver settles on); on the driver's refusal path (an
        SDK-side raise, a caller running this verb before the
        driver's method lands, a gate refusal) it is
        ``{"status": "error", "content": [{"text": "..."}]}`` with
        the driver's own reason inside. The verb does not reshape
        either shape - a future field the driver adds on the
        success path reaches a caller the moment the driver writes
        it, because this verb passes the envelope through.
    """
    # The handle is a live Python object typed :class:`~typing.Any`
    # (see the module docstring's import-cycle note), so the tool
    # schema carries no signal that ``None`` or a robot *name* is
    # refused.  The shared ``live_handle_refusal`` guard is the
    # one implementation of that judgement for this package; it is
    # keyed on the accessor the verb reads, which for this verb is
    # ``balance_stand`` (a callable that calls
    # ``LocoClient.BalanceStand`` and returns the driver's
    # envelope) rather than the sensor verbs' ``_snapshot``.
    # Returning its refusal envelope here rather than raising
    # keeps the four invariants every ``@tool`` handler owes a
    # caller (envelope not exception, names the verb, names
    # ``driver``, names the type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_balance_stand",
        driver,
        accessor="balance_stand",
        reads=(
            "the verb requests a G1 BalanceStand transition through "
            "the driver's own SDK-facing write path and reads back "
            "the envelope the driver produced"
        ),
        expected=(
            "a callable ``balance_stand(balance_mode)`` returning "
            "the driver's write envelope - pass the live G1Driver "
            "handle the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    # ``balance_mode`` is a data parameter the tool schema *does*
    # describe (an integer mode id, with the neon-bundle-observed
    # admitted set ``{0, 3}`` documented above), so a
    # model can synthesize the wrong shape here as easily as it
    # can reach the verb with the right one.  The refusals below
    # cover the shapes the driver's own ``balance_stand`` would
    # surface as an inner raise or as a silent coercion: a
    # ``None`` payload, a ``bool`` payload (``int`` subclass,
    # would coerce to ``0`` or ``1``), and a non-integer shape
    # (``float`` / ``str``) the neon bundle's own ``int(...)``
    # silently transformed.  Naming ``balance_mode`` here keeps
    # the four invariants: envelope not exception, names the
    # verb, names the parameter, names the shape received.  The
    # in-set admission ``{0, 3}`` is not enforced here (see the
    # module docstring's "does not refuse a balance_mode outside
    # the admitted set" note); that belongs on the driver.
    if balance_mode is None:
        return _refusal_envelope(
            "g1_balance_stand: `balance_mode` is required. Pass an "
            "integer mode id (neon-bundle-observed admitted set "
            "{0, 3}; 0 = static, 3 = dynamic) - see "
            "`g1_list_balance_modes` for the read-only descriptors "
            "(refs strands-labs/robots#358)."
        )
    # ``bool`` is an ``int`` subclass, so a bare ``isinstance(..., int)``
    # test would let ``True`` through as a silent ``1`` and ``False`` as
    # a silent ``0`` (both are inside the ``{0, 3}`` admitted set - a
    # ``True`` payload would even reach the SDK's static-balance handler
    # unnoticed).  Refusing ``bool`` explicitly matches the shape refusal
    # ``finite_number_error`` / ``positive_count_error`` render for the
    # same reason on the numeric verbs, and keeps the verb's message
    # naming the parameter and the shape received rather than silently
    # transitioning to a mode the caller writing ``True`` did not name.
    if isinstance(balance_mode, bool):
        return _refusal_envelope(
            f"g1_balance_stand: `balance_mode` must be an integer mode id "
            f"(int, not bool), got {type(balance_mode).__name__!r}. See "
            f"`g1_list_balance_modes` for the read-only descriptors "
            f"(refs strands-labs/robots#358)."
        )
    if not isinstance(balance_mode, int):
        return _refusal_envelope(
            f"g1_balance_stand: `balance_mode` must be an integer mode id "
            f"(int), got {type(balance_mode).__name__!r}. See "
            f"`g1_list_balance_modes` for the read-only descriptors "
            f"(refs strands-labs/robots#358)."
        )

    return driver.balance_stand(balance_mode)
