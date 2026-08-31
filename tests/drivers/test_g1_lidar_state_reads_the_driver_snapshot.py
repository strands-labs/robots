"""``g1_lidar_state`` returns exactly what ``G1Driver._snapshot("_lidar_state")`` gives it.

``g1_lidar_state`` is the fourth driver-instance-taking verb in
:mod:`strands_robots.tools.g1`, after ``g1_get_state`` (``strands-labs/robots#2934``),
``g1_battery`` (``strands-labs/robots#2938``) and ``g1_imu``
(``strands-labs/robots#2939``). Every earlier read-only verb in the package
(``g1_joint_reference``, ``g1_list_motion_gates``, ``g1_fsm_admits``) is a
pure reader over module-level constants; this one and its three siblings all
take a driver instance and read its cache through a named accessor --
:meth:`G1Driver._snapshot` for the ``rt/utlidar/lidar_state`` snapshot. The
tests here fix that contract by handing a hand-rolled driver double to the
verb and asserting the returned dict names each field ``_on_lidar_state``
writes into ``_lidar_state``.

The cache field names are read here off the driver's own writer
(:meth:`G1Driver._on_lidar_state` names ``code`` / ``code_text`` / ``freq`` /
``sys_rotation_speed`` / ``t``) rather than being restated in the tests, so
a widen or rename on the driver side moves both the write path and this verb
together. What the tests do restate is the SDK-load-hygiene contract every
file under :mod:`strands_robots.tools.g1` carries: importing the module must
not pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_lidar_state import g1_lidar_state


class _StubG1Driver:
    """A driver double whose ``_snapshot`` returns a fixed cache dict.

    ``g1_lidar_state`` calls ``driver._snapshot("_lidar_state")`` and reads
    the five fields ``_on_lidar_state`` writes into that dict. This double
    sits under the same interface without pulling the real driver's imports
    (the real class reaches CycloneDDS at construction time in some paths),
    so a test can hand a wired-shape cache to the verb without a bus. A
    ``None`` cache models the just-connected state where ``_on_lidar_state``
    has not fired yet.
    """

    def __init__(self, cache: dict[str, Any] | None) -> None:
        self._cache = cache
        self.calls: list[str] = []

    def _snapshot(self, attr: str) -> dict[str, Any] | None:
        self.calls.append(attr)
        if self._cache is None:
            return None
        return dict(self._cache)


def _call(driver: _StubG1Driver) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on that: the
    wrapper's contract is that it returns the wrapped function's return
    value verbatim. This helper is where a shape drift would surface once,
    rather than at every call site.
    """
    return g1_lidar_state(driver=driver)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable with
    the SDK absent; a module that pulled a submodule at import time would
    break every headless CI runner and Thor before an office bring-up. The
    driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only path
    that loads the SDK); this cell holds the lidar-state verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_lidar_state")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_lidar_state imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_with_no_state_message_yet_reports_absent() -> None:
    """A ``_snapshot`` returning ``None`` becomes ``present=False``.

    ``_on_lidar_state`` writes ``_lidar_state`` on every
    ``rt/utlidar/lidar_state`` decode, and ``_lidar_state`` starts as
    ``None`` on driver init. A just-connected driver whose lidar-state
    message has not arrived yet gives back ``None`` from the snapshot
    accessor, and the verb must report that decidably rather than
    fabricating a fault code of ``0`` (which the MID-360 uses for its
    healthy state) or a frequency of ``0.0`` (which would look like a
    stopped scan).
    """
    driver = _StubG1Driver(cache=None)
    result = _call(driver)

    assert result["status"] == "success"
    assert result["present"] is False
    assert result["code"] is None
    assert result["code_text"] is None
    assert result["freq"] is None
    assert result["sys_rotation_speed"] is None
    assert result["t"] is None
    assert driver.calls == ["_lidar_state"], (
        f"g1_lidar_state must read exactly _lidar_state from the cache; got {driver.calls}"
    )


def test_a_healthy_lidar_reports_every_field_the_decoder_wrote() -> None:
    """Each of the five ``_on_lidar_state`` fields rides through unchanged.

    ``_on_lidar_state`` writes ``code`` (the integer fault code off
    ``error_state``), ``code_text`` (the same code rendered through
    :func:`~strands_robots.tools.g1._g1_common.decode_code`), ``freq`` (the
    cloud frequency in Hz off ``cloud_frequency``, float),
    ``sys_rotation_speed`` (float) and ``t`` (wall time of decode). The
    verb is not the place to reword or convert those -- a caller reading
    this reading against a health chip on the mesh reads the same units
    the driver wrote.
    """
    cache: dict[str, Any] = {
        "code": 0,
        "code_text": "ok",
        "freq": 10.0,
        "sys_rotation_speed": 600.0,
        "t": 1_700_000_000.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["code"] == 0
    assert result["code_text"] == "ok"
    assert result["freq"] == 10.0
    assert result["sys_rotation_speed"] == 600.0
    assert result["t"] == 1_700_000_000.0


def test_a_faulted_lidar_reports_the_code_the_decoder_wrote() -> None:
    """A non-zero fault code round-trips from the driver's decode.

    ``_on_lidar_state`` reads ``error_state`` off the message and writes it
    into ``code`` as an integer along with its rendered text in
    ``code_text``. A MID-360 reporting a fault -- say code ``2`` -- must
    surface through this verb decidably, not be masked to zero. Reading a
    non-``ok`` code against ``freq`` and ``sys_rotation_speed`` at the same
    call is how a caller distinguishes a stopped scan from a healthy one.
    """
    cache: dict[str, Any] = {
        "code": 2,
        "code_text": "lidar fault (code=2)",
        "freq": 0.0,
        "sys_rotation_speed": 0.0,
        "t": 1_700_000_001.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["code"] == 2
    assert result["code_text"] == "lidar fault (code=2)"
    assert result["freq"] == 0.0
    assert result["sys_rotation_speed"] == 0.0


def test_a_partial_cache_reports_none_for_missing_fields() -> None:
    """A cache missing a field surfaces ``None`` for it, not a made-up value.

    ``_on_lidar_state`` writes every field on every decode, but the verb
    reads through ``dict.get`` rather than direct indexing so a cache that
    a future driver refactor writes as a subset does not raise here. What
    ``None`` names in that case is exactly that the driver did not write
    the field; a caller who needs the missing field cannot get it from
    this side and has to reach for the driver's own writer, which is
    where the omission lives.
    """
    cache: dict[str, Any] = {
        "code": 0,
        "code_text": "ok",
        "freq": 10.0,
        # sys_rotation_speed and t deliberately absent
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["code"] == 0
    assert result["code_text"] == "ok"
    assert result["freq"] == 10.0
    assert result["sys_rotation_speed"] is None
    assert result["t"] is None


def test_mutating_the_result_does_not_alter_the_cache() -> None:
    """The verb's dict is safe to mutate without racing the DDS thread.

    ``G1Driver._snapshot`` returns a copy of the cache under the
    ``_cache_lock``, so a caller that appends to or overwrites a field on
    the returned dict does not race the DDS thread's next
    ``_on_lidar_state`` write. This cell holds the verb to the same
    contract on the double: mutating the result must leave the source
    cache unchanged (the double copies via ``dict(self._cache)`` for the
    same reason).
    """
    cache: dict[str, Any] = {
        "code": 0,
        "code_text": "ok",
        "freq": 10.0,
        "sys_rotation_speed": 600.0,
        "t": 1_700_000_000.0,
    }
    driver = _StubG1Driver(cache=cache)
    result = _call(driver)

    result["code"] = 999
    result["injected"] = "should not appear in source"

    assert cache["code"] == 0, "mutating the verb's dict must not alter the driver's cache"
    assert "injected" not in cache
