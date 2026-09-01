"""Agent-facing wrapper for the driver's cached ``rt/lf/bmsstate`` snapshot.

``G1Driver`` subscribes ``rt/lf/bmsstate`` at connect time and decodes each
message into a small ``_battery`` dict carrying the SOC percentage, the
pack current, the pack cycle count and the wall time the last message
decoded at. ``G1Driver.get_status`` publishes ``battery_pct``
from that same dict on the mesh's status wire, which the ``g1_get_state``
verb already surfaces to an agent; this verb is the *cached-snapshot*
companion the ``g1_state`` docstring names, returning every field the
driver's decoder actually wrote rather than the pack percentage alone.

This verb does not subscribe DDS. The driver's own subscriber already
delivers ``rt/lf/bmsstate`` under the singleton ``_DDS_INIT_LOCK`` from
:mod:`~strands_robots.tools.g1._g1_common`, and a second subscriber path
on the same topic would compete for the wire and duplicate the bus load
that lock is meant to prevent (this is the same rule the ``g1_state``
module names, refs strands-labs/robots#358).  The verb is duck-typed on
``driver._snapshot("_battery")`` - the same accessor the driver's own
``stream(action="sensors")`` path reads through - which returns a copy of
the cache under the driver's ``_cache_lock`` so a caller mutating the
result does not race the DDS thread that writes into it.

The driver argument is typed :class:`~typing.Any` at runtime rather than
as ``G1Driver`` for the same reason the ``g1_state`` module gives: the
driver module imports
``ensure_dds`` from this package at load, so a runtime import of
``G1Driver`` here would close a cycle, and ``@tool`` calls
:func:`typing.get_type_hints` at decoration time so a string forward
reference cannot resolve without pulling the driver at import. The verb
is duck-typed on ``_snapshot`` (any object with a ``_snapshot(attr)``
returning the cache dict answers), which is also how the tests hand it a
hand-rolled double. ``import strands_robots.tools.g1.g1_battery`` still
pulls no ``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

What this module does not do.

* Subscribe DDS. Every field it returns is written by the driver's own
  ``_on_bms`` handler; adding a second subscriber path would compete for
  ``rt/lf/bmsstate`` and double the bus touch ``_DDS_INIT_LOCK`` is meant
  to prevent.
* Rewrite the driver's decode. ``_on_bms`` names ``pct`` / ``current`` /
  ``cycle`` / ``t``; those are what this verb returns verbatim.  It names
  no charge flag because ``BmsState_`` declares none - see that decoder's
  docstring.  A neon-side ``g1_battery`` port additionally read ``soh``,
  cell-level voltages and a per-cell temperature vector off ``BmsState_``
  directly - fields the driver's decoder does not carry today. Adding
  them here would be a second decoder for the same message, so those
  fields land (if at all) on the driver's ``_on_bms``, not this verb.
* Decide a battery-floor refusal. ``G1Driver._check_motion_gates`` reads
  ``self._battery["pct"]`` against ``self._battery_floor_pct`` on every
  write; a caller planning that write reads this verb's ``pct`` and can
  compare it themselves.  Restating the driver's floor rule on this
  side would be a second source of truth for a domain the driver's own
  refusal string already names verbatim.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import snapshot_handle_refusal


@tool
def g1_battery(driver: Any) -> dict[str, Any]:
    """Return the driver's cached ``rt/lf/bmsstate`` snapshot.

    Read-only. Calls ``driver._snapshot("_battery")`` (a copy under
    the driver's ``_cache_lock``, so a caller mutating the result does not
    race the DDS thread) and reshapes the dict into an agent-facing
    envelope. A driver whose subscriber has not received a BMS message
    yet - just-connected, or wire dropped - reports ``present=False`` and
    every field ``None``; the verb does not fabricate a reading the
    driver does not have.

    Args:
        driver: An object with a ``_snapshot(attr: str)`` method returning
            the cached sensor dict (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep this
            module out of the import cycle the driver's own ``ensure_dds``
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note.  The verb is duck-typed on
            ``_snapshot``; any object with that method answering
            ``"_battery"`` and returning the cache dict shape the
            driver's ``_on_bms`` writes will satisfy it.

    Returns:
        A dict with ``status``, a ``present`` flag naming whether the
        driver has a cached reading yet, and the four fields ``_on_bms``
        writes: ``pct`` (SOC percentage, ``float`` or ``None``),
        ``current`` (pack current as ``BmsState_.current`` reports it,
        ``float`` or ``None``), ``cycle`` (integer cycle count or ``None``)
        and ``t`` (the wall time the reading was decoded at, seconds since
        epoch, ``float`` or ``None``).  There is no charging flag:
        ``BmsState_`` declares no charge field, so reporting one would be a
        guess with the shape of a reading.  On a driver whose subscriber has
        not received a BMS message yet the returned dict carries
        ``present=False`` and every field ``None`` - the verb does not
        fabricate a reading the driver does not have.
    """
    refusal = snapshot_handle_refusal("g1_battery", driver)
    if refusal is not None:
        return refusal

    snapshot = driver._snapshot("_battery")
    if snapshot is None:
        return {
            "status": "success",
            "present": False,
            "pct": None,
            "current": None,
            "cycle": None,
            "t": None,
        }
    return {
        "status": "success",
        "present": True,
        "pct": snapshot.get("pct"),
        "current": snapshot.get("current"),
        "cycle": snapshot.get("cycle"),
        "t": snapshot.get("t"),
    }
