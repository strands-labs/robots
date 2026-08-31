"""Agent-facing wrapper for ``G1Driver.set_swing_height``.

``G1Driver.set_swing_height`` is the driver-side swing-height entry
point: a caller passes a target leg-lift clearance in meters and the
driver publishes the SDK's raw ``_Call`` on API id ``7103`` over the
same DDS singleton :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
opens. The Python SDK
(:class:`~unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) does not
expose a public ``SetSwingHeight`` method - the setter is reachable
only through the raw ``_Call`` on API ``7103``, which the neon
bundle's ``g1_set_swing_height`` verb
(``cagataycali/neon-the-g1/tools/g1_posture.py`` and the shared
``_g1_common.set_swing_height`` helper) fronted under a single-writer
lock. The neon bundle's own wrapper narrowed the argument to
``max(0.0, min(0.2, float(height)))`` before dispatch; the read-only
half of that envelope already landed as
``g1_swing_height_envelope`` (removed; envelope constants live inline here) (refs
strands-labs/robots#358), and this module is the write-side companion
that hands the target to the driver.

The driver's method itself is not yet plumbed on
:class:`~strands_robots.drivers.g1.G1Driver` today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on); the accessor grades this verb through
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`, so a
handle without a ``set_swing_height`` method refuses with a message
naming the verb, the ``driver`` parameter and the accessor it read
for, and once the driver lands the same call returns the envelope
the driver wrote verbatim. This is the same shape ``g1_set_fsm``
(refs strands-labs/robots#3025) and ``g1_set_stand_height`` (refs
strands-labs/robots#3031) already ship.

This module is a thin duck-typed wrapper. It reads the driver
through ``driver.set_swing_height`` and returns the envelope the
driver produced verbatim. A future field the driver adds on the
success path (say, an ``rc`` from the raw ``_Call`` handler, the
measured swing height after the gait settled) reaches a caller the
moment the driver writes it, because this verb does not restate the
shape. What it does add is the two refusal envelopes every ``@tool``
handler in this package owes its callers instead of an exception: a
live-handle refusal (``driver`` is ``None`` or a robot *name* or an
object without ``set_swing_height``) and the driver's own refusal
surfaced verbatim.

The FSM gate is not consulted here. The driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` is
the one gate for arm-SDK writes (``send_action`` / ``run_policy`` /
``start_task``); the API-``7103`` setter sits on the loco side of
the same DDS singleton and belongs on the locomotion admission set
:data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` the driver's
own path will check when the write lands. A second gate call here
would double the read against a cache the driver's FSM refresher
fills, and would refuse targets the driver's own path admits.
Restating any of that on this side would be a second source of
truth for a rule the driver's own path already enforces (refs
strands-labs/robots#358, strands-labs/robots#2916). ``import
strands_robots.tools.g1.g1_set_swing_height`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_set_fsm``,
``g1_set_stand_height``, ``g1_start_task``, ``g1_run_policy``,
``g1_send_action`` and ``g1_stop_task`` give: the driver module
imports :func:`~strands_robots.tools.g1._g1_common.ensure_dds` from
this package at load, so a runtime import of ``G1Driver`` here
would close a cycle, and ``@tool`` calls
:func:`typing.get_type_hints` at decoration time so a string
forward reference cannot resolve without pulling the driver at
import. The verb is duck-typed on ``set_swing_height``; any object
with a synchronous ``set_swing_height(height)`` returning the
driver's envelope satisfies it, which is also how the tests hand it
a hand-rolled double.

What this module does not do.

* Clamp ``height`` into the neon-bundle-observed range. The
  read-only envelope ``g1_swing_height_envelope`` (removed; envelope constants live inline here)
  names ``0.0`` and ``0.2`` as the bounds the neon bundle's own
  wrapper enforced; the driver's method is where a wire-side clamp
  lives, not this verb. A caller who passes ``0.5`` reaches the
  driver's own refusal (or the firmware's clamp-and-warn), which
  round-trips through this verb verbatim. Refusing above the
  envelope here would fork the neon bundle's admission set into
  a second source of truth this module would then have to keep
  in sync with the envelope lookup - and the SDK itself places no
  clamp on the ``7103`` argument (refs the
  ``g1_swing_height_envelope`` module docstring).
* Encode the API-``7103`` dispatch. The raw ``_Call`` and the
  neon bundle's :func:`_g1_common.set_swing_height` helper are
  driver-side wire concerns; the verb passes ``height`` through
  unchanged and the driver's method decides how to reach the
  SDK's private handler. A caller reading this verb's docstring
  learns the ergonomic contract, not the wire format.
* Decode the SDK's ``rc`` return into a label. The
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` table
  (surfaced by :mod:`~strands_robots.tools.g1.g1_error_codes`)
  names every SDK return code the driver's method may quote;
  restating the label here would fork the same table.
* Restate the driver's refusal wording. Whatever text the
  driver's method writes (a rc-decoded sentence, a firmware
  clamp warning, a no-set-swing-height-yet message like
  ``g1_start_task``'s registry-not-wired one) passes through
  this verb; a verbatim quote here would trap the verb to one
  release's prose (refs strands-labs/robots#2874).
* Schedule sequences. ``set_swing_height`` is one call; a
  caller who wants a gait-clearance ramp (shuffle -> normal ->
  high-lift) issues them one at a time and reads the driver's
  envelope after each. The SDK's handler is not re-entrant and
  neon's own bundle held a single-writer lock; when the driver
  lands its ``set_swing_height`` method it owns that
  serialisation, not this verb.
* Check whether the driver's live ``_fsm_id`` is inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That
  membership test is the driver's own gate concern; the read-
  only lookup ``g1_swing_height_envelope`` (removed; envelope constants live inline here)
  surfaces the set to a caller planning the write, but this
  verb does not enforce it (the driver's method will).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal
from strands_robots.utils import finite_number_error


def _refusal_envelope(text: str) -> dict[str, Any]:
    """Return the ``@tool`` error envelope with ``text`` inside.

    Kept as a small free helper so every ``height`` refusal path in
    this module renders the same shape a caller can grep for,
    matching the driver's own :func:`~strands_robots.drivers.g1._refuse`
    free function on the write side.
    """
    return {"status": "error", "content": [{"text": text}]}


@tool
def g1_set_swing_height(
    driver: Any,
    height: float | None = None,
) -> dict[str, Any]:
    """Set the G1's swing height (leg-lift clearance while walking).

    Calls ``G1Driver.set_swing_height``
    once and returns the envelope the driver produced verbatim. The
    driver's method routes to the SDK's raw ``_Call`` on API id
    ``7103`` (the setter the Python SDK's ``LocoClient`` does not
    expose as a public method); a caller who passes ``0.05``
    reaches the SDK's minimum-clearance shuffle gait, a caller who
    passes ``0.15`` reaches the neon-bundle-observed upper end of
    the typical safe range, and a value outside the neon bundle's
    ``[0.0, 0.2]`` window reaches the driver's own refusal or the
    firmware's clamp-and-warn (the SDK itself places no clamp on
    the ``7103`` argument). The neon bundle's own
    ``g1_set_swing_height`` verb documented ``0.05..0.15 m`` as
    the typical safe range; this verb ports the write-side of that
    contract unchanged so a caller upgrading from the neon bundle
    reaches the same behaviour.

    The swing-height transition is a locomotion-shaped write; the
    driver's
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    gate will (once the driver's method lands) refuse the write
    while ``_fsm_id`` is outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` (refs
    strands-labs/robots#2916). A caller planning the write compares
    the driver's live ``fsm_id`` (from ``get_status``) against the
    ``walk_ready_fsm_ids`` ``g1_swing_height_envelope`` (removed; envelope constants live inline here)
    surfaces before reaching this verb; this verb does not re-run
    that check itself.

    Args:
        driver: An object with a callable
            ``set_swing_height(height)`` returning the driver's
            write envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`). Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to
            keep this module out of the import cycle the driver's
            own :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note. The verb is duck-
            typed on ``set_swing_height``; any object with that
            method returning the envelope shape the driver writes
            will satisfy it. On today's driver ``set_swing_height``
            is not yet exposed, so the :func:`live_handle_refusal`
            grader refuses the call with a message naming the
            accessor; once the driver's method lands (the SDK-
            facing gate work in strands-labs/robots#358), the
            same call returns the driver's envelope verbatim.
        height: The target swing height in meters. The neon bundle
            documented the typical safe range as ``0.05..0.15 m``
            and clamped its own wrapper's argument to
            ``max(0.0, min(0.2, float(height)))`` before dispatch;
            the read-only envelope
            ``g1_swing_height_envelope`` (removed; envelope constants live inline here)
            names those bounds as
            ``_SWING_HEIGHT_MIN``
            and ``_SWING_HEIGHT_MAX``.
            This verb does not clamp - the driver's method (once
            landed) is where a wire-side clamp lives, and a caller
            who passes an out-of-range value reaches either the
            driver's refusal or the firmware's own clamp-and-warn
            through the verb's pass-through. Validated against
            :func:`~strands_robots.utils.finite_number_error`
            (NOT :func:`~strands_robots.utils.positive_finite_number_error`)
            because ``0.0`` is a legitimate command the neon
            bundle's own wrapper did not reject: it is the
            minimum-clearance / shuffle gait, and refusing it
            here would drop a caller's most conservative
            locomotion command. The verb does not refuse
            strictly-negative values either - the neon bundle's
            own wrapper rounded any negative input up to ``0.0``
            before dispatch, so a caller-facing sentinel does not
            live on that side of the domain (unlike
            ``g1_set_stand_height``'s HighStand fallback), and
            the driver's method (once landed) is where any wire-
            side clamp lives. A ``bool`` payload is refused for
            the same reason :func:`finite_number_error` refuses
            it: ``True`` would act as a silent ``1.0`` (5x the
            neon bundle's upper bound), ``False`` as a silent
            ``0.0`` (LOW / shuffle gait) - neither is a value a
            caller writing ``True`` would have named on purpose.
            ``nan`` and ``inf`` are refused because the
            firmware's own clamp path assumes finite arithmetic
            and the SDK's raw ``_Call`` handler would pass either
            through unchanged.

    Returns:
        The envelope ``G1Driver.set_swing_height`` returned. On
        the success path the driver's method will surface the SDK's
        ``rc`` inside a ``{"status": "success", "content":
        [{"json": {"rc": 0, "message": "SetSwingHeight(...) rc=0
        (OK)"}}]}`` envelope; on the driver's refusal path (an
        SDK-side raise, a caller running this verb before the
        driver's method lands, a firmware clamp warning) it is
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
    # ``set_swing_height`` (a callable that calls the SDK's raw
    # ``_Call`` on API 7103 and returns the driver's envelope)
    # rather than the sensor verbs' ``_snapshot``.  Returning its
    # refusal envelope here rather than raising keeps the four
    # invariants every ``@tool`` handler owes a caller (envelope
    # not exception, names the verb, names ``driver``, names the
    # type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_set_swing_height",
        driver,
        accessor="set_swing_height",
        reads=(
            "the verb requests a G1 swing-height (walking leg-lift "
            "clearance) transition through the driver's own SDK-facing "
            "write path and reads back the envelope the driver produced"
        ),
        expected=(
            "a callable ``set_swing_height(height)`` returning the "
            "driver's write envelope - pass the live G1Driver handle "
            "the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    # ``height`` is a data parameter the tool schema *does*
    # describe (a finite float target in meters, with the
    # neon-bundle-observed range ``0.0..0.2`` surfaced by
    # ``g1_swing_height_envelope`` (removed; envelope constants live inline here)), so
    # a model can synthesize the wrong shape here as easily as it
    # can reach the verb with the right one.  The refusals below
    # cover the shapes the driver's own ``set_swing_height`` would
    # surface as an inner raise: a ``None`` payload, a non-numeric
    # shape, and the non-finite / bool subclass shapes
    # ``finite_number_error`` refuses.  Naming ``height``
    # here keeps the four invariants: envelope not exception,
    # names the verb, names the parameter, names the shape
    # received.
    if height is None:
        return _refusal_envelope(
            "g1_set_swing_height: `height` is required. Pass a finite "
            "float in meters (neon-bundle-observed range "
            "0.0..0.2; typical safe range 0.05..0.15) - see "
            "``g1_list_swing_height_envelope`` for the read-only bounds "
            "(refs strands-labs/robots#358)."
        )
    # The shared validator refuses the same domain
    # ``g1_set_stand_height`` refuses (``nan`` poisons every
    # comparison it reaches, ``inf`` collapses arithmetic, a
    # ``bool`` acts as a silent ``0.0`` / ``1.0``, a non-numeric
    # shape raises out of a call that must return an envelope).
    # This verb uses ``finite_number_error`` rather than
    # ``positive_finite_number_error`` because the neon bundle's
    # own wrapper admits ``0.0`` (the minimum-clearance shuffle
    # gait) and clamps rather than refuses a negative value; a
    # ``> 0`` refusal here would drop a caller's most conservative
    # locomotion command. The driver's method is where any wire-
    # side clamp lives.
    height_error = finite_number_error(height, "height", "g1_set_swing_height")
    if height_error is not None:
        return _refusal_envelope(height_error)

    return driver.set_swing_height(height)
