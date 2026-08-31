"""Agent-facing wrapper for the driver's cached ``rt/pressuresensorstate`` snapshot.

``G1Driver`` subscribes ``rt/pressuresensorstate`` at connect time and
``_on_pressure`` decodes each ``PressSensorState_`` message into a small
``_pressure`` dict carrying the pressure vector (one reading per foot
pressure sensor), the temperature vector (one per pressure sensor), the
``lost`` packet counter, the ``reserve`` scalar, and the wall time the last
message decoded at.  This verb is the cached-snapshot companion to
``g1_battery`` (``strands-labs/robots#2938``), ``g1_imu``
(``strands-labs/robots#2939``), ``g1_lidar_state``
(``strands-labs/robots#2941``) and ``g1_mainboard``
(``strands-labs/robots#2947``), returning every field the driver's
pressure decoder actually wrote rather than computing anything new.

This verb does not subscribe DDS.  The driver's own subscriber already
delivers ``rt/pressuresensorstate`` under the singleton
``_DDS_INIT_LOCK`` from
:mod:`~strands_robots.tools.g1._g1_common`, and a second subscriber path
on the same topic would compete for the wire and duplicate the bus load
that lock is meant to prevent (this is the same rule the sibling readers
``g1_battery``, ``g1_imu``, ``g1_lidar_state`` and ``g1_mainboard`` state
in their own module docstrings, refs ``strands-labs/robots#358``).  The
verb is duck-typed on ``driver._snapshot("_pressure")`` -- the same
accessor the driver's own ``stream(action="sensors")`` path reads
through -- which returns a copy of the cache under the driver's
``_cache_lock`` so a caller mutating the result does not race the DDS
thread that writes into it.

The driver argument is typed :class:`~typing.Any` at runtime rather than
as ``G1Driver`` for the same reason ``g1_battery`` gives: the driver
module imports ``ensure_dds`` from this package at load, so a runtime
import of ``G1Driver`` here would close a cycle, and ``@tool`` calls
:func:`typing.get_type_hints` at decoration time so a string forward
reference cannot resolve without pulling the driver at import.  The verb
is duck-typed on ``_snapshot`` (any object with a ``_snapshot(attr)``
returning the cache dict answers), which is also how the tests hand it a
hand-rolled double.  ``import strands_robots.tools.g1.g1_pressure``
still pulls no ``unitree_sdk2py`` submodule (the package's SDK-load-
hygiene contract, refs ``strands-labs/robots#358``).

What this module does not do.

* Subscribe DDS.  Every field it returns is written by the driver's own
  ``_on_pressure`` handler; adding a second subscriber path would
  compete for ``rt/pressuresensorstate`` and double the bus touch
  ``_DDS_INIT_LOCK`` is meant to prevent.
* Rewrite the driver's decode.  ``_on_pressure`` names ``pressure`` /
  ``temperature`` / ``lost`` / ``reserve`` / ``t``; those are what this
  verb returns verbatim.  The neon-side ``g1_pressure`` port
  additionally read ``voltage`` / ``timestamp`` / ``tick`` off
  ``PressSensorState_`` directly -- fields the current
  ``PressSensorState_`` IDL layout does not declare (the IDL declares
  ``pressure`` and ``temperature`` as ``float32[12]`` vectors and
  ``lost`` / ``reserve`` as ``uint32`` scalars, and nothing else).  A
  decoder reading those absent fields would land the ``getattr``
  default in the record forever -- the same failure mode
  ``strands-labs/robots#2941``'s ``_reads_the_declared_fields`` cell was
  landed to catch on ``LidarState_``.  If a future firmware adds them
  back, ``_DECLARED_PRESSURE_FIELDS`` in the driver-decode tests fires
  so the frozen declaration and the decoder can be updated together.
* Interpret positions.  ``pressure`` and ``temperature`` are 12-element
  vectors on the IDL; the mapping from index to foot sensor location is
  a firmware concern the driver does not restate, so a caller that
  needs it reads the Unitree SDK docs (which the neon-side note refers
  to) rather than a translation table this verb would drift from.
* Convert units.  ``pressure`` is in the units ``PressSensorState_``
  declares (raw sensor reading on the current firmware), ``temperature``
  is in the units the IDL declares (Celsius per the SDK docs); a caller
  who wants a different unit converts them themselves so the numbers
  this verb returns are bit-identical to what a re-published log would
  carry.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import snapshot_handle_refusal


@tool
def g1_pressure(driver: Any) -> dict[str, Any]:
    """Return the driver's cached ``rt/pressuresensorstate`` snapshot.

    Read-only.  Calls ``driver._snapshot("_pressure")`` (a copy under
    the driver's ``_cache_lock``, so a caller mutating the result does
    not race the DDS thread) and reshapes the dict into an agent-facing
    envelope.  A driver whose subscriber has not received a
    ``PressSensorState_`` message yet -- just-connected, or wire
    dropped -- reports ``present=False`` and every field ``None``; the
    verb does not fabricate a reading the driver does not have.

    Args:
        driver: An object with a ``_snapshot(attr: str)`` method
            returning the cached sensor dict (in practice a
            ``G1Driver``).  Typed :class:`~typing.Any` rather than as
            ``G1Driver`` to keep this module out of the import cycle
            the driver's own ``ensure_dds`` reach into this package
            would close -- see the module docstring's SDK-load-hygiene
            note.  The verb is duck-typed on ``_snapshot``; any object
            with that method answering ``"_pressure"`` and returning
            the cache dict shape the driver's ``_on_pressure`` writes
            will satisfy it.

    Returns:
        A dict with ``status``, a ``present`` flag naming whether the
        driver has a cached reading yet, and the five fields
        ``_on_pressure`` writes: ``pressure`` (a per-sensor vector of
        raw pressure readings as ``list[float]`` or ``None``),
        ``temperature`` (a per-sensor vector of Celsius readings as
        ``list[float]`` or ``None``), ``lost`` (the packet-loss counter
        as ``int`` or ``None``), ``reserve`` (the reserve scalar the
        IDL declares next to it as ``int`` or ``None``), and ``t`` (the
        wall time the reading was decoded at, seconds since epoch,
        ``float`` or ``None``).  On a driver whose subscriber has not
        received a ``PressSensorState_`` message yet the returned dict
        carries ``present=False`` and every field ``None`` -- the verb
        does not fabricate a reading the driver does not have.  A
        field the current firmware does not declare surfaces as
        ``None`` for that key alone (the driver's ``_on_pressure``
        reads through ``getattr`` with a default), so a partial
        reading is decidable rather than surfaced as an empty dict.
    """
    refusal = snapshot_handle_refusal("g1_pressure", driver)
    if refusal is not None:
        return refusal

    snapshot = driver._snapshot("_pressure")
    if snapshot is None:
        return {
            "status": "success",
            "present": False,
            "pressure": None,
            "temperature": None,
            "lost": None,
            "reserve": None,
            "t": None,
        }
    return {
        "status": "success",
        "present": True,
        "pressure": snapshot.get("pressure"),
        "temperature": snapshot.get("temperature"),
        "lost": snapshot.get("lost"),
        "reserve": snapshot.get("reserve"),
        "t": snapshot.get("t"),
    }
