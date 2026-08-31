"""Agent-facing wrapper for ``G1Driver.set_stand_height``.

``G1Driver.set_stand_height`` is the driver-side stand-height entry
point: a caller passes a target height in meters and the driver
publishes the SDK's :class:`~unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.SetStandHeight`
call over the same DDS singleton :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
opens, or - when the caller passes a negative sentinel - falls back
to :meth:`~unitree_sdk2py.g1.loco.g1_loco_client.LocoClient.HighStand`
which uses a ``UINT32_MAX`` height sentinel the raw SDK exposes only
as a bare method call.  The negative-value fallback is the neon
bundle's ``g1_set_stand_height`` verb's one addition over
``use_unitree(service='loco', operation='SetStandHeight', ...)``: it
lets a caller name "the tallest stance the robot admits" without
knowing the SDK's own sentinel encoding.

The driver's method itself is not yet plumbed on
:class:`~strands_robots.drivers.g1.G1Driver` today (refs
strands-labs/robots#358 for the SDK-facing gate work the write
belongs on); the accessor grades this verb through
:func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`, so a
handle without a ``set_stand_height`` method refuses with a message
naming the verb, the ``driver`` parameter and the accessor it read
for, and once the driver lands the same call returns the envelope
the driver wrote verbatim.  This is the same shape ``g1_set_fsm``
(refs strands-labs/robots#3025) already ships.

This module is a thin duck-typed wrapper.  It reads the driver
through ``driver.set_stand_height`` and returns the envelope the
driver produced verbatim.  A future field the driver adds on the
success path (say, an ``rc`` from the SDK's ``SetStandHeight``
handler, the measured stand height after the transition settled)
reaches a caller the moment the driver writes it, because this verb
does not restate the shape.  What it does add is the two refusal
envelopes every ``@tool`` handler in this package owes its callers
instead of an exception: a live-handle refusal (``driver`` is
``None`` or a robot *name* or an object without
``set_stand_height``) and the driver's own refusal surfaced
verbatim.

The FSM gate is not consulted here.  The driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` is
the one gate for arm-SDK writes (``send_action`` / ``run_policy`` /
``start_task``); ``SetStandHeight`` sits on the loco side of the
same DDS singleton and admits stance transitions the arm-write gate
refuses (a caller who lowered the stand height while sitting is
outside the arm-SDK admission set anyway).  A second gate call here
would double the read against a cache the driver's FSM refresher
fills, and would refuse targets the driver's own path admits.
Restating any of that on this side would be a second source of
truth for a rule the driver's own path already enforces (refs
strands-labs/robots#358, strands-labs/robots#2916).  ``import
strands_robots.tools.g1.g1_set_stand_height`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_set_fsm``,
``g1_start_task``, ``g1_run_policy``, ``g1_send_action`` and
``g1_stop_task`` give: the driver module imports
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` from this
package at load, so a runtime import of ``G1Driver`` here would
close a cycle, and ``@tool`` calls :func:`typing.get_type_hints` at
decoration time so a string forward reference cannot resolve
without pulling the driver at import.  The verb is duck-typed on
``set_stand_height``; any object with a synchronous
``set_stand_height(height)`` returning the driver's envelope
satisfies it, which is also how the tests hand it a hand-rolled
double.

What this module does not do.

* Decide which heights the SDK admits.  The SDK's ``SetStandHeight``
  handler admits any finite float and clamps at wire time against a
  firmware-owned range (the neon bundle's docstring names the
  typical usable range as ``0.0..~0.8`` meters); the driver's method
  is where a wire-side clamp lives, not this verb.  A caller who
  passes ``5.0`` reaches the driver's own refusal (or the firmware's
  clamp-and-warn), which round-trips through this verb verbatim.
* Encode the ``UINT32_MAX`` sentinel the SDK's ``HighStand`` uses.
  The negative-value fallback is a public agent-facing shape (the
  neon bundle's own default), but the sentinel encoding is a
  driver-side wire concern.  The verb passes ``height`` through
  unchanged and the driver's method decides whether to route to
  ``SetStandHeight`` or ``HighStand``; a caller reading this verb's
  docstring learns the ergonomic contract, not the wire format.
* Decode the SDK's ``rc`` return into a label.  The
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` table
  (surfaced by :mod:`~strands_robots.tools.g1.g1_error_codes`) names
  every SDK return code the driver's method may quote; restating
  the label here would fork the same table.
* Restate the driver's refusal wording.  Whatever text the driver's
  method writes (a rc-decoded sentence, a firmware clamp warning, a
  no-set-stand-height-yet message like ``g1_start_task``'s
  registry-not-wired one) passes through this verb; a verbatim
  quote here would trap the verb to one release's prose (refs
  strands-labs/robots#2874).
* Schedule sequences.  ``set_stand_height`` is one call; a caller
  who wants a stance ramp (crouch -> upright -> HighStand) issues
  them one at a time and reads the driver's envelope after each.
  The SDK's handler is not re-entrant and neon's own bundle held a
  single-writer lock; when the driver lands its
  ``set_stand_height`` method it owns that serialisation, not this
  verb.
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
def g1_set_stand_height(
    driver: Any,
    height: float | None = None,
) -> dict[str, Any]:
    """Set the G1's stand height, with a negative-value HighStand fallback.

    Calls ``G1Driver.set_stand_height`` once and returns the
    envelope the driver produced verbatim.  The
    driver's method routes on the sign of ``height``: a non-negative
    value publishes ``LocoClient.SetStandHeight(height)`` with the
    caller's target in meters, and a negative value publishes
    ``LocoClient.HighStand()`` which uses the SDK's ``UINT32_MAX``
    height sentinel to select the tallest stance the firmware
    admits.  The neon bundle's ``g1_set_stand_height`` verb
    documented the same ergonomic contract; this verb ports it
    unchanged so a caller upgrading from the neon bundle reaches the
    same behaviour.

    The stand-height transition is a loco-side write, not an arm-SDK
    write; the driver's
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    gate is the arm-write gate (``send_action`` / ``run_policy`` /
    ``start_task``) and does not admit or refuse this verb (refs
    strands-labs/robots#2916).  A caller who wants the robot in a
    posture that admits arm writes issues ``g1_set_fsm`` first (refs
    strands-labs/robots#3025); this verb adjusts the height *within*
    the current stance.

    Args:
        driver: An object with a callable
            ``set_stand_height(height)`` returning the driver's
            write envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep
            this module out of the import cycle the driver's own
            :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note.  The verb is duck-typed
            on ``set_stand_height``; any object with that method
            returning the envelope shape the driver writes will
            satisfy it.  On today's driver ``set_stand_height`` is
            not yet exposed, so the :func:`live_handle_refusal`
            grader refuses the call with a message naming the
            accessor; once the driver's method lands (the SDK-facing
            gate work in strands-labs/robots#358), the same call
            returns the driver's envelope verbatim.
        height: The target stand height in meters, or a negative
            value to select the SDK's ``HighStand`` sentinel.  The
            neon bundle documented the typical usable range as
            ``0.0..~0.8`` meters (the exact upper bound is firmware-
            owned and the SDK clamps at wire time); ``0.0`` selects
            the LOW / crouched stance and any negative value routes
            to the ``HighStand`` fallback.  Validated against
            :func:`~strands_robots.utils.finite_number_error` because
            the caller-facing domain is any finite real (positive
            for direct SetStandHeight, negative for HighStand
            fallback), so ``nan`` / ``inf`` / non-numeric shapes are
            refused here rather than reaching the driver's own
            :func:`float` call or the SDK's ``SetStandHeight``
            handler (``nan`` and ``inf`` in a stand-height target
            are shapes the firmware would not clamp meaningfully and
            the driver's own arithmetic would poison).  A ``bool``
            payload is refused for the same reason
            :func:`finite_number_error` refuses it: ``True`` would
            act as a silent ``1.0`` (nearly-max stance), ``False``
            as a silent ``0.0`` (LOW stance) - neither is a value a
            caller writing ``True`` would have named on purpose.

    Returns:
        The envelope ``G1Driver.set_stand_height`` returned.  On
        the success path the driver's method will surface the SDK's
        ``rc`` inside a ``{"status": "success", "content": [{"json":
        {"rc": 0, "message": "SetStandHeight(...) rc=0 (OK)"}}]}``
        envelope (or the equivalent shape for the ``HighStand``
        branch); on the driver's refusal path (an SDK-side raise, a
        caller running this verb before the driver's method lands,
        a firmware clamp warning) it is ``{"status": "error",
        "content": [{"text": "..."}]}`` with the driver's own reason
        inside.  The verb does not reshape either shape - a future
        field the driver adds on the success path reaches a caller
        the moment the driver writes it, because this verb passes
        the envelope through.
    """
    # The handle is a live Python object typed :class:`~typing.Any`
    # (see the module docstring's import-cycle note), so the tool
    # schema carries no signal that ``None`` or a robot *name* is
    # refused.  The shared ``live_handle_refusal`` guard is the one
    # implementation of that judgement for this package; it is keyed
    # on the accessor the verb reads, which for this verb is
    # ``set_stand_height`` (a callable that calls the SDK's
    # ``SetStandHeight`` or ``HighStand`` and returns the driver's
    # envelope) rather than the sensor verbs' ``_snapshot``.
    # Returning its refusal envelope here rather than raising keeps
    # the four invariants every ``@tool`` handler owes a caller
    # (envelope not exception, names the verb, names ``driver``,
    # names the type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_set_stand_height",
        driver,
        accessor="set_stand_height",
        reads=(
            "the verb requests a G1 stand-height transition through the "
            "driver's own SDK-facing write path and reads back the "
            "envelope the driver produced"
        ),
        expected=(
            "a callable ``set_stand_height(height)`` returning the "
            "driver's write envelope - pass the live G1Driver handle "
            "the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    # ``height`` is a data parameter the tool schema *does* describe
    # (a finite float target in meters, with a negative value routing
    # to the ``HighStand`` fallback), so a model can synthesize the
    # wrong shape here as easily as it can reach the verb with the
    # right one.  The three refusals below cover the shapes the
    # driver's own ``set_stand_height`` would surface as an inner
    # raise: a ``None`` payload, a non-numeric shape, and the
    # non-finite / bool subclass shapes ``finite_number_error``
    # refuses.  Naming ``height`` here keeps the four invariants:
    # envelope not exception, names the verb, names the parameter,
    # names the shape received.
    if height is None:
        return _refusal_envelope(
            "g1_set_stand_height: `height` is required. Pass a finite "
            "float in meters (typical range 0.0..~0.8; 0.0 = LOW / "
            "crouched, negative = HighStand fallback) - see "
            "G1Driver.set_stand_height for the shape (refs "
            "strands-labs/robots#358)."
        )
    # The shared validator refuses the domain
    # :meth:`G1Driver.__init__`'s ``battery_floor_pct`` refuses (``nan``
    # poisons every comparison it reaches, ``inf`` collapses arithmetic,
    # a ``bool`` acts as a silent ``0.0`` / ``1.0``, a non-numeric shape
    # raises out of a call that must return an envelope).  Unlike
    # :func:`~strands_robots.utils.positive_finite_number_error` the sign
    # is NOT constrained here, because a negative value is the caller-facing
    # signal that selects the SDK's ``HighStand`` sentinel.
    height_error = finite_number_error(height, "height", "g1_set_stand_height")
    if height_error is not None:
        return _refusal_envelope(height_error)

    return driver.set_stand_height(height)
