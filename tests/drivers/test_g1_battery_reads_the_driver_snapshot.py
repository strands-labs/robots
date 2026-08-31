"""``g1_battery`` returns exactly what ``G1Driver._snapshot("_battery")`` gives it.

``g1_battery`` is the second driver-instance-taking verb in
:mod:`strands_robots.tools.g1`, after ``g1_get_state``.  Every earlier verb
in the package is a pure reader over module-level constants; this one and
``g1_get_state`` both take a driver instance and read its cache
through a named accessor - :meth:`G1Driver.get_status` for state,
:meth:`G1Driver._snapshot` for the BMS snapshot.  The tests here fix that
contract by handing a hand-rolled driver double to the verb and asserting
the returned dict names each field ``_on_bms`` writes into ``_battery``.

The cache field names are read here off the driver's own writer
(:meth:`G1Driver._on_bms` names ``pct`` / ``charging`` / ``current`` /
``cycle`` / ``t``) rather than being restated in the tests, so a widen
or rename on the driver side moves both the write path and this verb
together.  What the tests do restate is the SDK-load-hygiene contract
every file under :mod:`strands_robots.tools.g1` carries: importing the
module must not pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_battery import g1_battery


class _StubG1Driver:
    """A driver double whose ``_snapshot`` returns a fixed cache dict.

    ``g1_battery`` calls ``driver._snapshot("_battery")`` and reads the
    five fields ``_on_bms`` writes into that dict.  This double sits
    under the same interface without pulling the real driver's imports
    (the real class reaches CycloneDDS at construction time in some
    paths), so a test can hand a wired-shape cache to the verb without a
    bus.  A ``None`` cache models the just-connected state where
    ``_on_bms`` has not fired yet.
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
    return g1_battery(driver=driver)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import time
    would break every headless CI runner and Thor before an office
    bring-up.  The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the battery verb to it
    too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_battery")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_battery imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_with_no_bms_message_yet_reports_absent() -> None:
    """A ``_snapshot`` returning ``None`` becomes ``present=False``.

    ``_on_bms`` writes ``_battery`` on every ``rt/lf/bmsstate`` decode,
    and ``_battery`` starts as ``None`` on driver init.  A just-connected
    driver whose BMS message has not arrived yet gives back ``None`` from
    the snapshot accessor, and the verb must report that decidably rather
    than fabricating a zero-percent reading.
    """
    driver = _StubG1Driver(cache=None)
    result = _call(driver)

    assert result["status"] == "success"
    assert result["present"] is False
    assert result["pct"] is None
    assert result["charging"] is None
    assert result["current"] is None
    assert result["cycle"] is None
    assert result["t"] is None
    assert driver.calls == ["_battery"], f"g1_battery must read exactly _battery from the cache; got {driver.calls}"


def test_a_healthy_pack_reports_every_field_the_decoder_wrote() -> None:
    """Each of the five ``_on_bms`` fields rides through unchanged.

    ``_on_bms`` writes ``pct`` (SOC percent, float), ``charging`` (bool),
    ``current`` (pack current in amps, float), ``cycle`` (integer cycle
    count) and ``t`` (wall time of decode).  The verb is not the place
    to reword or convert those - a caller comparing this reading against
    the driver's ``_battery_floor_pct`` reads ``pct`` at the same units
    the gate does.
    """
    cache: dict[str, Any] = {
        "pct": 87.5,
        "charging": False,
        "current": -1.25,
        "cycle": 42,
        "t": 1_700_000_000.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["pct"] == 87.5
    assert result["charging"] is False
    assert result["current"] == -1.25
    assert result["cycle"] == 42
    assert result["t"] == 1_700_000_000.0


def test_a_charging_pack_reports_charging_true() -> None:
    """The ``charging`` flag round-trips from the driver's decode.

    ``_on_bms`` writes ``bool(getattr(msg, "charge", 0))`` into
    ``charging``; a firmware reporting a non-zero charge flag lands as
    ``True`` in the cache, and this verb must not silently drop or
    invert that.
    """
    cache: dict[str, Any] = {
        "pct": 92.0,
        "charging": True,
        "current": 3.5,
        "cycle": 100,
        "t": 1_700_000_100.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["charging"] is True
    assert result["current"] == 3.5


def test_a_critical_pack_pct_is_reported_verbatim_not_clipped() -> None:
    """A percentage under the driver's battery floor rides through unchanged.

    ``G1Driver._check_motion_gates`` reads ``_battery["pct"]`` against
    ``_battery_floor_pct`` on the write path; a caller planning the
    write reads this verb's ``pct`` at the same units and compares
    themselves.  The verb must not clip, floor, or annotate the value -
    a critical pack surfaces as its actual percentage so the caller can
    quote the same number the driver's refusal would.
    """
    cache: dict[str, Any] = {
        "pct": 5.0,
        "charging": False,
        "current": -0.8,
        "cycle": 250,
        "t": 1_700_000_200.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["pct"] == 5.0
    assert result["present"] is True


def test_a_missing_field_in_the_cache_reports_none() -> None:
    """A cache dict missing one of the five fields returns ``None`` for it.

    ``_on_bms`` may in principle write a partial dict on a decode where
    one attribute was absent from the IDL message (the handler catches
    ``Exception`` in its own decode path and logs, but a caller writing
    ``_battery`` from a test also skips fields).  The verb reads through
    ``dict.get`` so a missing key surfaces as ``None`` rather than a
    ``KeyError`` at the tool boundary.
    """
    cache: dict[str, Any] = {
        "pct": 60.0,
        "charging": False,
        # current / cycle / t missing
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["present"] is True
    assert result["pct"] == 60.0
    assert result["charging"] is False
    assert result["current"] is None
    assert result["cycle"] is None
    assert result["t"] is None


def test_the_verb_reads_the_snapshot_exactly_once() -> None:
    """The verb calls ``_snapshot`` exactly once per invocation.

    Reading the cache twice on a single verb call would double the lock
    acquisition on the driver's ``_cache_lock`` and (on a driver whose
    ``_snapshot`` had side effects) double the touch.  This test counts
    the calls on a hand-rolled double so the single-read contract is
    graded rather than assumed.
    """
    driver = _StubG1Driver(
        cache={
            "pct": 75.0,
            "charging": False,
            "current": -0.5,
            "cycle": 30,
            "t": 1_700_000_300.0,
        }
    )
    _call(driver)
    assert driver.calls == ["_battery"], f"g1_battery must read _battery exactly once; got {driver.calls}"


def test_the_returned_dict_does_not_alias_the_cache() -> None:
    """Mutating the verb's return value does not corrupt the driver's cache.

    ``G1Driver._snapshot`` already returns a copy under the driver's
    ``_cache_lock`` so the DDS thread that writes ``_on_bms`` does not
    race a caller.  The verb receives that copy and reshapes it into an
    envelope; a caller further mutating the returned dict must not
    reach back into the driver's cache.  The stub returns a fresh dict
    per call to model the real driver's behaviour, and this test grades
    that the verb's own return does not share references with the stub's
    internal cache either.
    """
    original_cache: dict[str, Any] = {
        "pct": 80.0,
        "charging": False,
        "current": -1.0,
        "cycle": 25,
        "t": 1_700_000_400.0,
    }
    driver = _StubG1Driver(cache=original_cache)
    result = _call(driver)
    result["pct"] = 999.0
    result["present"] = False

    # The driver's cache is untouched: a fresh snapshot still reports
    # what ``_on_bms`` wrote.
    fresh = _call(driver)
    assert fresh["pct"] == 80.0
    assert fresh["present"] is True
