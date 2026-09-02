"""Agent-facing wrapper for the driver's cached ``rt/mainboardstate`` snapshot.

``G1Driver`` subscribes ``rt/mainboardstate`` at connect time and
``_on_mainboard`` decodes each ``MainBoardState_`` message into a small
``_mainboard`` dict carrying the fan state vector, the temperature vector
across the board thermistors, the system-state code, the firmware tick and
the wall time the last message decoded at.  This verb is the cached-snapshot
companion to ``g1_battery`` (``strands-labs/robots#2938``), ``g1_imu``
(``strands-labs/robots#2939``) and ``g1_lidar_state``
(``strands-labs/robots#2941``), returning every field the driver's
mainboard decoder actually wrote rather than computing anything new.

This verb does not subscribe DDS.  The driver's own subscriber already
delivers ``rt/mainboardstate`` under the singleton ``_DDS_INIT_LOCK`` from
:mod:`~strands_robots.tools.g1._g1_common`, and a second subscriber path
on the same topic would compete for the wire and duplicate the bus load
that lock is meant to prevent (this is the same rule the sibling readers
``g1_battery``, ``g1_imu`` and ``g1_lidar_state`` state in their own
module docstrings, refs ``strands-labs/robots#358``).  The verb is
duck-typed on ``driver._snapshot("_mainboard")`` -- the same accessor the
driver's own ``stream(action="sensors")`` path reads through -- which
returns a copy of the cache under the driver's ``_cache_lock`` so a
caller mutating the result does not race the DDS thread that writes into
it.

The driver argument is typed :class:`~typing.Any` at runtime rather than
as ``G1Driver`` for the same reason ``g1_battery`` gives: the driver
module imports ``ensure_dds`` from this package at load, so a runtime
import of ``G1Driver`` here would close a cycle, and ``@tool`` calls
:func:`typing.get_type_hints` at decoration time so a string forward
reference cannot resolve without pulling the driver at import.  The verb
is duck-typed on ``_snapshot`` (any object with a ``_snapshot(attr)``
returning the cache dict answers), which is also how the tests hand it a
hand-rolled double.  ``import strands_robots.tools.g1.g1_mainboard``
still pulls no ``unitree_sdk2py`` submodule (the package's SDK-load-
hygiene contract, refs ``strands-labs/robots#358``).

What this module does not do.

* Subscribe DDS.  Every field it returns is written by the driver's own
  ``_on_mainboard`` handler; adding a second subscriber path would
  compete for ``rt/mainboardstate`` and double the bus touch
  ``_DDS_INIT_LOCK`` is meant to prevent.
* Rewrite the driver's decode.  ``_on_mainboard`` names ``fan_state`` /
  ``temperature`` / ``value`` / ``state`` / ``t``; those are what this
  verb returns verbatim.  The neon-side ``g1_mainboard`` port additionally
  read ``fan_speed`` / ``cpu_temperature`` / ``sys_state`` /
  ``sys_bat_state`` / ``bms_state`` / ``tick`` off ``MainBoardState_``
  directly, each behind a ``hasattr`` gate because the layout declares
  none of them; adding them on the verb would be a second decoder for the
  same message that agrees with the driver's writer only when both name
  the same fields.  Those fields land (if at all) on the driver's
  ``_on_mainboard``, not this verb.
* Rewrite pressure-sensor readings.  ``rt/pressuresensorstate`` is a
  separate DDS topic with its own IDL (``PressSensorState_``); a
  ``g1_pressure`` port lands as its own verb once the driver caches
  that topic, not folded into this one.
* Convert units.  ``temperature`` is in the units ``MainBoardState_``
  declares (Celsius on the current firmware), ``fan_state`` is a vector
  of integer fan flags as ``_on_mainboard`` wrote it; a caller who wants
  Fahrenheit converts them themselves so the numbers this verb returns
  are bit-identical to what a re-published log would carry.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import snapshot_handle_refusal


@tool
def g1_mainboard(driver: Any) -> dict[str, Any]:
    """Return the driver's cached ``rt/mainboardstate`` snapshot.

    Read-only.  Calls ``driver._snapshot("_mainboard")`` (a copy
    under the driver's ``_cache_lock``, so a caller mutating the result
    does not race the DDS thread) and reshapes the dict into an
    agent-facing envelope.  A driver whose subscriber has not received a
    ``MainBoardState_`` message yet -- just-connected, or wire dropped --
    reports ``present=False`` and every field ``None``; the verb does
    not fabricate a reading the driver does not have.

    Args:
        driver: An object with a ``_snapshot(attr: str)`` method
            returning the cached sensor dict (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep
            this module out of the import cycle the driver's own
            ``ensure_dds`` reach into this package would close -- see
            the module docstring's SDK-load-hygiene note.  The verb is
            duck-typed on ``_snapshot``; any object with that method
            answering ``"_mainboard"`` and returning the cache dict
            shape the driver's ``_on_mainboard`` writes will satisfy
            it.

    Returns:
        A dict with ``status``, a ``present`` flag naming whether the
        driver has a cached reading yet, and the five fields
        ``_on_mainboard`` writes: ``fan_state`` (a vector of integer
        fan flags as ``list[int]`` or ``None``), ``temperature`` (a
        vector of board-thermistor readings as ``list[float]`` or
        ``None``), ``value`` and ``state`` (the two remaining vectors
        ``MainBoardState_`` declares, as ``list[float]`` / ``list[int]``
        or ``None``, under the vendor's own names because the IDL
        documents no semantics for them) and ``t`` (the wall time the
        reading was decoded at, seconds since epoch, ``float`` or
        ``None``).  On a driver
        whose subscriber has not received a ``MainBoardState_``
        message yet the returned dict carries ``present=False`` and
        every field ``None`` -- the verb does not fabricate a reading
        the driver does not have.  A field the current firmware does
        not declare surfaces as ``None`` for that key alone (the
        driver's ``_on_mainboard`` reads through ``getattr`` with a
        default), so a partial reading is decidable rather than
        surfaced as an empty dict.
    """
    refusal = snapshot_handle_refusal("g1_mainboard", driver)
    if refusal is not None:
        return refusal

    snapshot = driver._snapshot("_mainboard")
    if snapshot is None:
        return {
            "status": "success",
            "present": False,
            "fan_state": None,
            "temperature": None,
            "value": None,
            "state": None,
            "t": None,
        }
    return {
        "status": "success",
        "present": True,
        "fan_state": snapshot.get("fan_state"),
        "temperature": snapshot.get("temperature"),
        "value": snapshot.get("value"),
        "state": snapshot.get("state"),
        "t": snapshot.get("t"),
    }
