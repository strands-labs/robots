"""``g1_mainboard`` returns exactly what ``G1Driver._snapshot("_mainboard")`` gives it.

``g1_mainboard`` is the fifth driver-instance-taking verb in
:mod:`strands_robots.tools.g1`, after ``g1_get_state``
(``strands-labs/robots#2934``), ``g1_battery``
(``strands-labs/robots#2938``), ``g1_imu``
(``strands-labs/robots#2939``) and ``g1_lidar_state``
(``strands-labs/robots#2941``).  Every earlier read-only verb in the
package that takes a driver instance reads its cache through a named
accessor -- :meth:`G1Driver._snapshot` -- and this one does the same for
the ``rt/mainboardstate`` cache the driver's own ``_on_mainboard``
writes.  The tests here fix that contract by handing a hand-rolled
driver double to the verb and asserting the returned dict names each
field ``_on_mainboard`` writes into ``_mainboard``.

The cache field names are read here off the driver's own writer
(:meth:`G1Driver._on_mainboard` names ``fan_state`` / ``temperature`` /
``value`` / ``state`` / ``t``) rather than being restated in the
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

from strands_robots.tools.g1.g1_mainboard import g1_mainboard


class _StubG1Driver:
    """A driver double whose ``_snapshot`` returns a fixed cache dict.

    ``g1_mainboard`` calls ``driver._snapshot("_mainboard")`` and reads
    the five fields ``_on_mainboard`` writes into that dict.  This
    double sits under the same interface without pulling the real
    driver's imports (the real class reaches CycloneDDS at construction
    time in some paths), so a test can hand a wired-shape cache to the
    verb without a bus.  A ``None`` cache models the just-connected
    state where ``_on_mainboard`` has not fired yet.
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
    return g1_mainboard(driver=driver)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an office
    bring-up.  The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the mainboard verb to it
    too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_mainboard")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_mainboard imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_with_no_mainboard_message_yet_reports_absent() -> None:
    """A ``_snapshot`` returning ``None`` becomes ``present=False``.

    ``_on_mainboard`` writes ``_mainboard`` on every ``rt/mainboardstate``
    decode, and ``_mainboard`` starts as ``None`` on driver init.  A
    just-connected driver whose first ``MainBoardState_`` has not
    arrived yet gives back ``None`` from the snapshot accessor, and the
    verb must report that decidably rather than fabricating a
    temperature of ``0.0`` (which would look like a bench-cold board) or
    a ``fan_state`` of ``[]`` (which would look like a unit with no
    fans reported at all).
    """
    driver = _StubG1Driver(cache=None)
    result = _call(driver)

    assert result["status"] == "success"
    assert result["present"] is False
    assert result["fan_state"] is None
    assert result["temperature"] is None
    assert result["value"] is None
    assert result["state"] is None
    assert result["t"] is None
    assert driver.calls == ["_mainboard"], (
        f"g1_mainboard must read exactly _mainboard from the cache; got {driver.calls}"
    )


def test_a_healthy_reading_reports_every_field_the_decoder_wrote() -> None:
    """Each of the five ``_on_mainboard`` fields rides through unchanged.

    ``_on_mainboard`` writes ``fan_state`` (``list[int]``, one entry per
    fan flag), ``temperature`` (``list[float]``, one entry per board
    thermistor in the units the firmware declares), ``value``
    (``list[float]``), ``state`` (``list[int]``) and ``t`` (wall time of
    decode).  The verb is not the place to reword, convert or truncate
    those -- a caller reading the values against a health chip on the
    mesh reads the same units the driver wrote.
    """
    cache: dict[str, Any] = {
        "fan_state": [1, 1, 1, 0],
        "temperature": [42.5, 44.0, 41.75],
        "value": [1.5, 2.5, 3.5],
        "state": [0, 1, 2],
        "t": 1_700_000_000.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["fan_state"] == [1, 1, 1, 0]
    assert result["temperature"] == [42.5, 44.0, 41.75]
    assert result["value"] == [1.5, 2.5, 3.5]
    assert result["state"] == [0, 1, 2]
    assert result["t"] == 1_700_000_000.0


def test_a_hot_board_reading_is_returned_verbatim_not_clipped() -> None:
    """An extreme temperature reading is returned verbatim.

    ``_on_mainboard`` does not clip or convert ``temperature``: a
    thermally saturated G1 mainboard reports a per-thermistor reading
    above the vendor's guidance, and the verb must not silently clip
    that to a safe-looking value.  A caller reading the reading to
    decide whether to shed load compares the same value the driver's
    decoder wrote.
    """
    cache: dict[str, Any] = {
        "fan_state": [1, 1, 1, 1],
        "temperature": [88.5, 91.25, 92.0],
        "value": [9.5, 9.75, 10.0],
        "state": [2, 2, 2],
        "t": 1_700_000_100.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["present"] is True
    assert result["temperature"] == [88.5, 91.25, 92.0]
    assert result["state"] == [2, 2, 2]


def test_a_missing_field_in_the_cache_reports_none() -> None:
    """A cache dict missing one of the five keys returns ``None`` for it.

    ``_on_mainboard`` reads through ``getattr(msg, name, None)`` at
    every field, so a firmware that renames one of the declared fields
    yields ``None`` in the cache dict for that key.  The verb reads
    through ``dict.get`` so a missing key surfaces as ``None`` at the
    tool boundary rather than a ``KeyError``.  A caller who needs the
    missing field cannot get it from this side and has to reach for
    the driver's own writer, which is where the omission lives.
    """
    cache: dict[str, Any] = {
        "fan_state": [1, 1, 1, 1],
        "temperature": [40.0],
        # value / state / t deliberately absent
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["present"] is True
    assert result["fan_state"] == [1, 1, 1, 1]
    assert result["temperature"] == [40.0]
    assert result["value"] is None
    assert result["state"] is None
    assert result["t"] is None


def test_the_verb_reads_the_snapshot_exactly_once() -> None:
    """The verb calls ``_snapshot`` exactly once per invocation.

    Reading the cache twice on a single verb call would double the
    lock acquisition on the driver's ``_cache_lock`` and (on a driver
    whose ``_snapshot`` had side effects) double the touch.  This test
    counts the calls on a hand-rolled double so the single-read
    contract is graded rather than assumed.
    """
    driver = _StubG1Driver(
        cache={
            "fan_state": [1, 1, 1, 1],
            "temperature": [40.0, 40.0, 40.0],
            "value": [0.0, 0.0, 0.0],
            "state": [0, 0, 0],
            "t": 1_700_000_300.0,
        }
    )
    _call(driver)
    assert driver.calls == ["_mainboard"], f"g1_mainboard must read _mainboard exactly once; got {driver.calls}"


def test_mutating_the_result_does_not_alter_the_cache() -> None:
    """The verb's dict is safe to mutate without racing the DDS thread.

    ``G1Driver._snapshot`` returns a copy of the cache under the
    ``_cache_lock``, so a caller that appends to or overwrites a field
    on the returned dict does not race the DDS thread's next
    ``_on_mainboard`` write.  This cell holds the verb to the same
    contract on the double: mutating the result must leave the source
    cache unchanged (the double copies via ``dict(self._cache)`` for
    the same reason).
    """
    cache: dict[str, Any] = {
        "fan_state": [1, 1, 1, 1],
        "temperature": [40.0, 40.0, 40.0],
        "value": [0.0, 0.0, 0.0],
        "state": [0, 0, 0],
        "t": 1_700_000_400.0,
    }
    driver = _StubG1Driver(cache=cache)
    result = _call(driver)

    result["state"] = [999]
    result["injected"] = "should not appear in source"

    assert cache["state"] == [0, 0, 0], "mutating the verb's dict must not alter the driver's cache"
    assert "injected" not in cache


def test_a_bare_zero_reading_is_still_present() -> None:
    """A cache carrying zero-valued fields still reports ``present=True``.

    An all-zero ``state`` / ``value`` vector is what a board with nothing
    latched reports, and an empty ``fan_state`` / ``temperature`` list is
    what a build with no populated fans / thermistors would report.  Every
    one of those is a *reading*: ``_on_mainboard`` wrote something into
    ``_mainboard``, so ``_snapshot("_mainboard")`` returned a
    non-``None`` dict.  The verb reports the fact of the reading through
    ``present``, not its physical plausibility; falsy field values must
    not silently collapse to ``present=False``.
    """
    cache: dict[str, Any] = {
        "fan_state": [],
        "temperature": [],
        "value": [0.0],
        "state": [0],
        "t": 0.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["present"] is True
    assert result["fan_state"] == []
    assert result["temperature"] == []
    assert result["value"] == [0.0]
    assert result["state"] == [0]
    assert result["t"] == 0.0
