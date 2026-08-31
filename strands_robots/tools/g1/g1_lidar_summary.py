"""Agent-facing wrapper for the driver's cached lidar-cloud summary.

``G1Driver`` subscribes ``rt/utlidar/cloud_livox_mid360`` at connect time
and ``_on_lidar_cloud`` builds a small ``_lidar_summary`` dict from the
message header alone - ``count`` (``width * height``), ``width``,
``height``, ``point_step``, ``row_step`` and ``t`` - so no point is ever
enumerated on the DDS thread and the record's size is the same for a
sparse cloud and a full one. This verb is the cached-snapshot companion
to a lidar-state reader (``strands_robots.tools.g1.g1_lidar_state``, a
sibling port whose PR is open at time of writing), returning every field
``_on_lidar_cloud`` actually writes rather than computing anything new.

This verb does not subscribe DDS. The driver's own subscriber already
delivers ``rt/utlidar/cloud_livox_mid360`` under the singleton
``_DDS_INIT_LOCK`` from :mod:`~strands_robots.tools.g1._g1_common`, and
a second subscriber path on the same topic would compete for the wire
and duplicate the bus load that lock is meant to prevent (this is the
same rule the sibling readers ``g1_battery``, ``g1_imu`` and
``g1_lidar_state`` state in their own module docstrings, refs
strands-labs/robots#358). The verb is duck-typed on
``driver._snapshot("_lidar_summary")`` - the same accessor the driver's
own ``stream(action="sensors")`` path reads through - which returns a
copy of the cache under the driver's ``_cache_lock`` so a caller
mutating the result does not race the DDS thread that writes into it.

The driver argument is typed :class:`~typing.Any` at runtime rather than
as ``G1Driver`` for the same reason g1_battery gives: the driver module
imports ``ensure_dds`` from this package at load, so a runtime import of
``G1Driver`` here would close a cycle, and ``@tool`` calls
:func:`typing.get_type_hints` at decoration time so a string forward
reference cannot resolve without pulling the driver at import. The verb
is duck-typed on ``_snapshot`` (any object with a ``_snapshot(attr)``
returning the cache dict answers), which is also how the tests hand it a
hand-rolled double. ``import strands_robots.tools.g1.g1_lidar_summary``
still pulls no ``unitree_sdk2py`` submodule (the package's SDK-load-
hygiene contract, refs strands-labs/robots#358).

What this module does not do.

* Subscribe DDS. Every field it returns is written by the driver's own
  ``_on_lidar_cloud`` handler; adding a second subscriber path would
  compete for ``rt/utlidar/cloud_livox_mid360`` and double the bus
  touch ``_DDS_INIT_LOCK`` is meant to prevent.
* Enumerate points, downsample, or cap. ``_on_lidar_cloud`` reads only
  the ``PointCloud2_`` *header*, so ``count`` is the cloud's true size
  (``width * height``). Clamping it here would hide exactly what the
  strands-labs/robots#2752 change was landed to stop hiding: a MID-360
  that drops from 24000 points to 3000 is reporting a fault, and a
  ``capped_at`` value alongside would tell a consumer the number had
  been clipped when it had not.
* Rebuild the cloud. This verb returns the *summary* record the mesh's
  health chip already reads. The 3D tile (issue strands-labs/robots#356)
  subscribes the raw cloud itself through a paced publisher and does
  its own downsampling; a caller who needs points reaches for that
  path, not this one.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import snapshot_handle_refusal


@tool
def g1_lidar_summary(driver: Any) -> dict[str, Any]:
    """Return the driver's cached lidar-cloud header summary.

    Read-only. Calls ``driver._snapshot("_lidar_summary")`` (a
    copy under the driver's ``_cache_lock``, so a caller mutating the
    result does not race the DDS thread) and reshapes the dict into an
    agent-facing envelope. A driver whose subscriber has not received a
    ``PointCloud2`` message yet - just-connected, or wire dropped -
    reports ``present=False`` and every field ``None``; the verb does
    not fabricate a reading the driver does not have.

    Args:
        driver: An object with a ``_snapshot(attr: str)`` method
            returning the cached sensor dict (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`). Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep
            this module out of the import cycle the driver's own
            ``ensure_dds`` reach into this package would close - see the
            module docstring's SDK-load-hygiene note. The verb is
            duck-typed on ``_snapshot``; any object with that method
            answering ``"_lidar_summary"`` and returning the cache dict
            shape the driver's ``_on_lidar_cloud`` writes will satisfy
            it.

    Returns:
        A dict with ``status``, a ``present`` flag naming whether the
        driver has a cached reading yet, and the six fields
        ``_on_lidar_cloud`` writes: ``count`` (the cloud's true point
        count as ``width * height``, integer, or ``None``), ``width``
        (integer or ``None``), ``height`` (integer or ``None``),
        ``point_step`` (bytes per point, integer or ``None``),
        ``row_step`` (bytes per row, integer or ``None``) and ``t`` (the
        wall time the cloud was summarised at, seconds since epoch,
        ``float`` or ``None``). On a driver whose subscriber has not
        received a ``PointCloud2`` message yet the returned dict carries
        ``present=False`` and every field ``None`` - the verb does not
        fabricate a reading the driver does not have. ``count`` is the
        cloud's uncapped size on purpose (refs
        strands-labs/robots#2752): a MID-360 that drops from 24000
        points to 3000 is reporting a fault, and clamping the number
        would hide it.
    """
    refusal = snapshot_handle_refusal("g1_lidar_summary", driver)
    if refusal is not None:
        return refusal

    snapshot = driver._snapshot("_lidar_summary")
    if snapshot is None:
        return {
            "status": "success",
            "present": False,
            "count": None,
            "width": None,
            "height": None,
            "point_step": None,
            "row_step": None,
            "t": None,
        }
    return {
        "status": "success",
        "present": True,
        "count": snapshot.get("count"),
        "width": snapshot.get("width"),
        "height": snapshot.get("height"),
        "point_step": snapshot.get("point_step"),
        "row_step": snapshot.get("row_step"),
        "t": snapshot.get("t"),
    }
