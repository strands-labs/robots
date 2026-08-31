"""Agent-facing wrapper for the driver's cached ``rt/utlidar/lidar_state`` snapshot.

``G1Driver`` subscribes ``rt/utlidar/lidar_state`` at connect time and
``_on_lidar_state`` decodes each ``LidarState_`` message into a small
``_lidar_state`` dict carrying the MID-360's fault code, its rendered text,
the cloud frequency, the system rotation speed and the wall time the last
message decoded at. ``G1Driver.get_status`` already surfaces those same fields
under ``lidar_state`` on the mesh's status wire, which ``g1_get_state``
(``strands-labs/robots#2934``) hands to an agent as one dict among many; this
verb is the cached-snapshot companion that returns the lidar-state fields
alone, at the same shape ``_on_lidar_state`` writes them.

This verb does not subscribe DDS. The driver's own subscriber already delivers
``rt/utlidar/lidar_state`` under the singleton ``_DDS_INIT_LOCK`` from
:mod:`~strands_robots.tools.g1._g1_common`, and a second subscriber path on
the same topic would compete for the wire and duplicate the bus load that
lock is meant to prevent (this is the same rule the ``g1_battery`` module
names, refs ``strands-labs/robots#358``). The verb is duck-typed on
``driver._snapshot("_lidar_state")`` -- the same accessor the driver's own
``stream(action="sensors")`` path reads through -- which returns a copy of
the cache under the driver's ``_cache_lock`` so a caller mutating the result
does not race the DDS thread that writes into it.

The driver argument is typed :class:`~typing.Any` at runtime rather than as
``G1Driver`` for the same reason the ``g1_battery`` module gives: the driver
module imports ``ensure_dds`` from this package at load, so a runtime import
of ``G1Driver`` here would close a cycle, and ``@tool`` calls
:func:`typing.get_type_hints` at decoration time so a string forward
reference cannot resolve without pulling the driver at import. The verb is
duck-typed on ``_snapshot`` (any object with a ``_snapshot(attr)`` returning
the cache dict answers), which is also how the tests hand it a hand-rolled
double. ``import strands_robots.tools.g1.g1_lidar_state`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene contract,
refs ``strands-labs/robots#358``).

What this module does not do.

* Subscribe DDS. Every field it returns is written by the driver's own
  ``_on_lidar_state`` handler; adding a second subscriber path would compete
  for ``rt/utlidar/lidar_state`` and double the bus touch
  ``_DDS_INIT_LOCK`` is meant to prevent.
* Rewrite the driver's decode. ``_on_lidar_state`` names ``code`` /
  ``code_text`` / ``freq`` / ``sys_rotation_speed`` / ``t``; those are what
  this verb returns verbatim. A neon-side ``g1_lidar`` port additionally
  read the firmware, software and SDK versions directly off ``LidarState_``,
  fields the driver's decoder does not carry today. Adding them here would
  be a second decoder for the same message, so those fields land (if at
  all) on the driver's ``_on_lidar_state``, not this verb.
* Turn the LiDAR on or off. The neon-side port opened a
  ``rt/utlidar/switch`` publisher next to the state subscriber, which would
  be a second publisher path on top of the driver's DDS engine and is
  outside the read-only contract this verb keeps.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import snapshot_handle_refusal


@tool
def g1_lidar_state(driver: Any) -> dict[str, Any]:
    """Return the driver's cached ``rt/utlidar/lidar_state`` snapshot.

    Read-only. Calls ``driver._snapshot("_lidar_state")`` (a copy
    under the driver's ``_cache_lock``, so a caller mutating the result
    does not race the DDS thread) and reshapes the dict into an
    agent-facing envelope. A driver whose subscriber has not received a
    ``LidarState_`` message yet -- just-connected, or wire dropped --
    reports ``present=False`` and every field ``None``; the verb does not
    fabricate a reading the driver does not have.

    Args:
        driver: An object with a ``_snapshot(attr: str)`` method returning
            the cached sensor dict (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep this
            module out of the import cycle the driver's own ``ensure_dds``
            reach into this package would close -- see the module
            docstring's SDK-load-hygiene note. The verb is duck-typed on
            ``_snapshot``; any object with that method answering
            ``"_lidar_state"`` and returning the cache dict shape the
            driver's ``_on_lidar_state`` writes will satisfy it.

    Returns:
        A dict with ``status``, a ``present`` flag naming whether the
        driver has a cached reading yet, and the five fields
        ``_on_lidar_state`` writes: ``code`` (the MID-360's fault code as
        an integer, or ``None``), ``code_text`` (the same code rendered
        through :func:`~strands_robots.tools.g1._g1_common.decode_code`,
        or ``None``), ``freq`` (the cloud frequency in Hz reported off
        the message's ``cloud_frequency`` field, ``float`` or ``None``),
        ``sys_rotation_speed`` (the system rotation speed off the
        message's ``sys_rotation_speed`` field, ``float`` or ``None``)
        and ``t`` (the wall time the reading was decoded at, seconds
        since epoch, ``float`` or ``None``). On a driver whose
        subscriber has not received a state message yet the returned
        dict carries ``present=False`` and every field ``None`` -- the
        verb does not fabricate a reading the driver does not have.
    """
    refusal = snapshot_handle_refusal("g1_lidar_state", driver)
    if refusal is not None:
        return refusal

    snapshot = driver._snapshot("_lidar_state")
    if snapshot is None:
        return {
            "status": "success",
            "present": False,
            "code": None,
            "code_text": None,
            "freq": None,
            "sys_rotation_speed": None,
            "t": None,
        }
    return {
        "status": "success",
        "present": True,
        "code": snapshot.get("code"),
        "code_text": snapshot.get("code_text"),
        "freq": snapshot.get("freq"),
        "sys_rotation_speed": snapshot.get("sys_rotation_speed"),
        "t": snapshot.get("t"),
    }
