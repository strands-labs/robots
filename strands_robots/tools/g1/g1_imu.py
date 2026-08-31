"""Agent-facing wrapper for the driver's cached ``rt/lowstate`` IMU snapshot.

``G1Driver`` subscribes ``rt/lowstate`` at connect time and ``_on_lowstate``
decodes each message into a small ``_imu`` dict carrying the base orientation
(``rpy``), the gyroscope reading, the accelerometer reading, the quaternion,
and the wall time the last message decoded at.  This verb is the
cached-snapshot companion to ``g1_battery`` (``strands-labs/robots#2938``),
returning every field the driver's IMU decoder actually wrote rather than
computing anything new.

This verb does not subscribe DDS.  The driver's own subscriber already
delivers ``rt/lowstate`` under the singleton ``_DDS_INIT_LOCK`` from
:mod:`~strands_robots.tools.g1._g1_common`, and a second subscriber path
on the same topic would compete for the wire and duplicate the bus load
that lock is meant to prevent (this is the same rule ``g1_state``
(``strands-labs/robots#2934``) and ``g1_battery`` name, refs
strands-labs/robots#358).  The verb is duck-typed on
``driver._snapshot("_imu")`` - the same accessor the driver's own
``stream(action="sensors")`` path reads through - which returns a copy of
the cache under the driver's ``_cache_lock`` so a caller mutating the
result does not race the DDS thread that writes into it.

The driver argument is typed :class:`~typing.Any` at runtime rather than
as ``G1Driver`` for the same reason ``g1_battery`` gives: the driver
module imports ``ensure_dds`` from this package at load, so a runtime
import of ``G1Driver`` here would close a cycle, and ``@tool`` calls
:func:`typing.get_type_hints` at decoration time so a string forward
reference cannot resolve without pulling the driver at import.  The verb
is duck-typed on ``_snapshot`` (any object with a ``_snapshot(attr)``
returning the cache dict answers), which is also how the tests hand it a
hand-rolled double.  ``import strands_robots.tools.g1.g1_imu`` still
pulls no ``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

What this module does not do.

* Subscribe DDS.  Every field it returns is written by the driver's own
  ``_on_lowstate`` handler; adding a second subscriber path would compete
  for ``rt/lowstate`` and double the bus touch ``_DDS_INIT_LOCK`` is
  meant to prevent.
* Rewrite the driver's decode.  ``_on_lowstate`` names ``rpy`` /
  ``gyroscope`` / ``accelerometer`` / ``quaternion`` / ``t``; those are
  what this verb returns verbatim.  The neon-side ``g1_read_lowstate``
  additionally decoded joint angles, torques, ``mode_machine`` /
  ``mode_pr`` / ``tick`` and computed a posture heuristic off the knees.
  ``mode_machine`` is surfaced by ``g1_state`` already; joint reads are a
  separate verb the driver's ``_on_lowstate`` does not yet cache (it
  caches the IMU sub-record only), and a posture label would be a second
  source of truth for a domain the driver's own gate decides at wire time.
  Those fields land (if at all) on ``_on_lowstate``, not this verb.
* Convert units.  ``rpy`` is in radians as ``_on_lowstate`` wrote it,
  ``gyroscope`` is rad/s, ``accelerometer`` is m/s²; a caller who wants
  degrees converts them themselves so the number this verb returns is
  bit-identical to what a re-published log would carry.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import snapshot_handle_refusal


@tool
def g1_imu(driver: Any) -> dict[str, Any]:
    """Return the driver's cached ``rt/lowstate`` IMU snapshot.

    Read-only.  Calls ``driver._snapshot("_imu")`` (a copy under
    the driver's ``_cache_lock``, so a caller mutating the result does
    not race the DDS thread) and reshapes the dict into an agent-facing
    envelope.  A driver whose subscriber has not received a ``LowState``
    message yet - just-connected, or wire dropped - reports
    ``present=False`` and every field ``None``; the verb does not
    fabricate a reading the driver does not have.

    Args:
        driver: An object with a ``_snapshot(attr: str)`` method
            returning the cached sensor dict (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep this
            module out of the import cycle the driver's own
            ``ensure_dds`` reach into this package would close - see the
            module docstring's SDK-load-hygiene note.  The verb is
            duck-typed on ``_snapshot``; any object with that method
            answering ``"_imu"`` and returning the cache dict shape the
            driver's ``_on_lowstate`` writes will satisfy it.

    Returns:
        A dict with ``status``, a ``present`` flag naming whether the
        driver has a cached reading yet, and the five fields
        ``_on_lowstate`` writes: ``rpy`` (roll/pitch/yaw as a
        three-element ``list[float]`` in radians, or ``None``),
        ``gyroscope`` (three-element ``list[float]`` in rad/s, or
        ``None``), ``accelerometer`` (three-element ``list[float]`` in
        m/s², or ``None``), ``quaternion`` (four-element ``list[float]``
        as [w, x, y, z], or ``None``) and ``t`` (the wall time the
        reading was decoded at, seconds since epoch, ``float`` or
        ``None``).  On a driver whose subscriber has not received a
        ``LowState`` message yet the returned dict carries
        ``present=False`` and every field ``None`` - the verb does not
        fabricate a reading the driver does not have.
    """
    refusal = snapshot_handle_refusal("g1_imu", driver)
    if refusal is not None:
        return refusal

    snapshot = driver._snapshot("_imu")
    if snapshot is None:
        return {
            "status": "success",
            "present": False,
            "rpy": None,
            "gyroscope": None,
            "accelerometer": None,
            "quaternion": None,
            "t": None,
        }
    return {
        "status": "success",
        "present": True,
        "rpy": snapshot.get("rpy"),
        "gyroscope": snapshot.get("gyroscope"),
        "accelerometer": snapshot.get("accelerometer"),
        "quaternion": snapshot.get("quaternion"),
        "t": snapshot.get("t"),
    }
