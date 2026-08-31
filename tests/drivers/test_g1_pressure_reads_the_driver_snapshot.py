"""``g1_pressure`` returns exactly what ``G1Driver._snapshot("_pressure")`` gives it.

``g1_pressure`` is the sixth driver-instance-taking verb in
:mod:`strands_robots.tools.g1`, after ``g1_get_state``
(``strands-labs/robots#2934``), ``g1_battery``
(``strands-labs/robots#2938``), ``g1_imu``
(``strands-labs/robots#2939``), ``g1_lidar_state``
(``strands-labs/robots#2941``) and ``g1_mainboard``
(``strands-labs/robots#2947``).  Every earlier read-only verb in the
package that takes a driver instance reads its cache through a named
accessor -- ``G1Driver._snapshot`` -- and this one does the same for
the ``rt/pressuresensorstate`` cache the driver's own ``_on_pressure``
writes.  The tests here fix that contract by handing a hand-rolled
driver double to the verb and asserting the returned dict names each
field ``_on_pressure`` writes into ``_pressure``.

The cache field names are read here off the driver's own writer
(``G1Driver._on_pressure`` names ``pressure`` / ``temperature`` /
``lost`` / ``reserve`` / ``t``) rather than being restated in the
tests, so a widen or rename on the driver side moves both the write
path and this verb together.  What the tests do restate is the
SDK-load-hygiene contract every file under
:mod:`strands_robots.tools.g1` carries: importing the module must not
pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_pressure import g1_pressure


class _StubG1Driver:
    """A driver double whose ``_snapshot`` returns a fixed cache dict.

    ``g1_pressure`` calls ``driver._snapshot("_pressure")`` and reads
    the five fields ``_on_pressure`` writes into that dict.  This
    double sits under the same interface without pulling the real
    driver's imports (the real class reaches CycloneDDS at construction
    time in some paths), so a test can hand a wired-shape cache to the
    verb without a bus.  A ``None`` cache models the just-connected
    state where ``_on_pressure`` has not fired yet.
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
    directly when called in-process, but a caller cannot rely on that:
    the wrapper's contract is that it returns the wrapped function's
    return value verbatim.  This helper is where a shape drift would
    surface once, rather than at every call site.
    """
    return g1_pressure(driver=driver)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an office
    bring-up.  The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the pressure verb to it
    too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_pressure")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_pressure imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_with_no_pressure_message_yet_reports_absent() -> None:
    """A ``_snapshot`` returning ``None`` becomes ``present=False``.

    ``_on_pressure`` writes ``_pressure`` on every
    ``rt/pressuresensorstate`` decode, and ``_pressure`` starts as
    ``None`` on driver init.  A just-connected driver whose first
    ``PressSensorState_`` has not arrived yet gives back ``None`` from
    the snapshot accessor, and the verb must report that decidably
    rather than fabricating a zero-pressure reading (which would look
    like the robot's feet were off the ground when in fact no message
    has arrived at all).
    """
    driver = _StubG1Driver(cache=None)
    result = _call(driver)
    assert result == {
        "status": "success",
        "present": False,
        "pressure": None,
        "temperature": None,
        "lost": None,
        "reserve": None,
        "t": None,
    }
    assert driver.calls == ["_pressure"], (
        "the verb must read the driver's cache through the named ``_snapshot`` accessor exactly once per call"
    )


def test_a_healthy_reading_rides_through_unchanged() -> None:
    """Every field the decoder wrote reaches the envelope verbatim.

    ``_on_pressure`` writes exactly the five keys the verb reshapes,
    and the reshape is a per-key read through ``dict.get`` -- so a
    reading that names all five keys with plausible values (positive
    pressures, mid-range temperatures, low ``lost`` and ``reserve``
    counts, a recent wall time) must land in the envelope bit-identical
    to what the cache held.
    """
    pressure_reading = [1.5, 2.1, 0.8, 0.9, 1.2, 1.4, 1.6, 2.3, 0.7, 1.0, 1.1, 1.3]
    temperature_reading = [
        25.5,
        26.0,
        25.8,
        25.9,
        26.1,
        25.7,
        25.6,
        26.2,
        25.5,
        25.9,
        26.0,
        25.8,
    ]
    driver = _StubG1Driver(
        cache={
            "pressure": pressure_reading,
            "temperature": temperature_reading,
            "lost": 0,
            "reserve": 0,
            "t": 1_700_000_000.5,
        },
    )
    result = _call(driver)
    assert result == {
        "status": "success",
        "present": True,
        "pressure": pressure_reading,
        "temperature": temperature_reading,
        "lost": 0,
        "reserve": 0,
        "t": 1_700_000_000.5,
    }


def test_a_reading_with_packet_loss_reports_lost_verbatim() -> None:
    """The ``lost`` counter is reported at the driver's units, not clipped.

    ``lost`` is a ``uint32`` on the IDL, so any value in ``[0, 2**32-1]``
    is a legal counter that the DDS thread might see under a wire
    disruption.  The verb passes it through as ``int`` -- a caller who
    wants to raise on a high count reads the number itself rather than
    the verb making that decision for it.
    """
    driver = _StubG1Driver(
        cache={
            "pressure": [0.5] * 12,
            "temperature": [25.0] * 12,
            "lost": 42,
            "reserve": 7,
            "t": 1_700_000_000.5,
        },
    )
    result = _call(driver)
    assert result["lost"] == 42
    assert result["reserve"] == 7


def test_a_missing_field_in_the_cache_reports_none() -> None:
    """A ``dict.get`` returns ``None`` for a key the cache omits.

    ``_on_pressure`` writes every declared field on every message, but
    a firmware rename would leave one out (the decoder would land
    ``None`` for that key -- see the driver-decode suite).  The verb
    passes the ``None`` through decidably rather than fabricating a
    placeholder, and the ``present`` flag stays ``True`` because the
    cache is not itself absent.
    """
    driver = _StubG1Driver(
        cache={
            "pressure": [0.5] * 12,
            # temperature deliberately absent
            "lost": 0,
            "reserve": 0,
            "t": 1_700_000_000.5,
        },
    )
    result = _call(driver)
    assert result["present"] is True
    assert result["pressure"] == [0.5] * 12
    assert result["temperature"] is None
    assert result["lost"] == 0
    assert result["reserve"] == 0
    assert result["t"] == 1_700_000_000.5


def test_the_verb_reads_the_snapshot_exactly_once() -> None:
    """Two reads would race the DDS thread's next write.

    ``_snapshot`` copies under ``_cache_lock``; reading it twice in one
    call would open a window where the two reads see different vectors
    (the DDS thread wrote in between).  The verb must build its
    envelope from a single ``_snapshot`` call so the ``pressure`` and
    ``temperature`` vectors in the returned dict come from the same
    message.
    """
    driver = _StubG1Driver(
        cache={
            "pressure": [0.5] * 12,
            "temperature": [25.0] * 12,
            "lost": 0,
            "reserve": 0,
            "t": 1_700_000_000.5,
        },
    )
    _call(driver)
    assert driver.calls == ["_pressure"], (
        f"the verb must call _snapshot exactly once per invocation; observed {driver.calls}"
    )


def test_the_returned_dict_does_not_alias_the_cache() -> None:
    """Mutating the verb's result must not corrupt the driver's cache.

    ``_snapshot`` returns a shallow ``dict(self._pressure)`` copy under
    ``_cache_lock``, but the vectors inside are references to the same
    lists the DDS thread will re-write.  This test asserts the shallow
    copy protects the top-level keys -- popping ``pressure`` from the
    result must not remove it from what a second call would read.
    """
    original_cache = {
        "pressure": [0.5] * 12,
        "temperature": [25.0] * 12,
        "lost": 0,
        "reserve": 0,
        "t": 1_700_000_000.5,
    }
    driver = _StubG1Driver(cache=original_cache)
    result_first = _call(driver)
    result_first.pop("pressure")
    result_first["status"] = "mutated"
    result_second = _call(driver)
    assert result_second["pressure"] == [0.5] * 12
    assert result_second["status"] == "success"


def test_a_bare_zero_reading_is_still_present() -> None:
    """A robot with its feet off the ground still has a fresh reading.

    ``present`` names whether the DDS subscriber has received a
    message, not whether the message reported non-zero pressures.  A
    robot in the air (or on a soft surface) reads bare zeros on every
    pressure sensor, which is a valid reading and must not collapse
    the ``present`` flag to ``False`` -- otherwise a caller could not
    tell "the wire dropped" from "the robot is airborne".
    """
    driver = _StubG1Driver(
        cache={
            "pressure": [0.0] * 12,
            "temperature": [0.0] * 12,
            "lost": 0,
            "reserve": 0,
            "t": 1_700_000_000.5,
        },
    )
    result = _call(driver)
    assert result["present"] is True
    assert result["pressure"] == [0.0] * 12
    assert result["temperature"] == [0.0] * 12
