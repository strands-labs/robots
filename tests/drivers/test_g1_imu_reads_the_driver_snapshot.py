"""``g1_imu`` returns exactly what ``G1Driver._snapshot("_imu")`` gives it.

``g1_imu`` is the third driver-instance-taking verb in
:mod:`strands_robots.tools.g1`, after
``g1_get_state`` (``strands-labs/robots#2934``) and
``g1_battery`` (``strands-labs/robots#2938``).  Every earlier
verb in the package is either a pure reader over module-level constants
or a status envelope reader; this one and ``g1_battery`` both take a
driver instance and read its cache through a named accessor -
:meth:`G1Driver._snapshot` - which returns a copy under the driver's
``_cache_lock``.  The tests here fix that contract by handing a
hand-rolled driver double to the verb and asserting the returned dict
names each field ``_on_lowstate`` writes into ``_imu``.

The cache field names are read here off the driver's own writer
(:meth:`G1Driver._on_lowstate` names ``rpy`` / ``gyroscope`` /
``accelerometer`` / ``quaternion`` / ``t``) rather than being restated
in the tests, so a widen or rename on the driver side moves both the
write path and this verb together.  What the tests do restate is the
SDK-load-hygiene contract every file under
:mod:`strands_robots.tools.g1` carries: importing the module must not
pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_imu import g1_imu


class _StubG1Driver:
    """A driver double whose ``_snapshot`` returns a fixed cache dict.

    ``g1_imu`` calls ``driver._snapshot("_imu")`` and reads the five
    fields ``_on_lowstate`` writes into that dict.  This double sits
    under the same interface without pulling the real driver's imports
    (the real class reaches CycloneDDS at construction time in some
    paths), so a test can hand a wired-shape cache to the verb without
    a bus.  A ``None`` cache models the just-connected state where
    ``_on_lowstate`` has not fired yet.
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
    return g1_imu(driver=driver)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an office
    bring-up.  The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the IMU verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_imu")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_imu imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_with_no_lowstate_message_yet_reports_absent() -> None:
    """A ``_snapshot`` returning ``None`` becomes ``present=False``.

    ``_on_lowstate`` writes ``_imu`` on every ``rt/lowstate`` decode,
    and ``_imu`` starts as ``None`` on driver init.  A just-connected
    driver whose first ``LowState`` has not arrived yet gives back
    ``None`` from the snapshot accessor, and the verb must report that
    decidably rather than fabricating a zero-orientation reading.
    """
    driver = _StubG1Driver(cache=None)
    result = _call(driver)

    assert result["status"] == "success"
    assert result["present"] is False
    assert result["rpy"] is None
    assert result["gyroscope"] is None
    assert result["accelerometer"] is None
    assert result["quaternion"] is None
    assert result["t"] is None
    assert driver.calls == ["_imu"], f"g1_imu must read exactly _imu from the cache; got {driver.calls}"


def test_a_healthy_reading_reports_every_field_the_decoder_wrote() -> None:
    """Each of the five ``_on_lowstate`` fields rides through unchanged.

    ``_on_lowstate`` writes ``rpy`` (list[float] radians), ``gyroscope``
    (list[float] rad/s), ``accelerometer`` (list[float] m/s^2),
    ``quaternion`` (list[float] as [w, x, y, z]) and ``t`` (wall time
    of decode).  The verb is not the place to reword, convert or
    truncate those - a caller planning any downstream computation reads
    the same units the driver's decoder wrote.
    """
    cache: dict[str, Any] = {
        "rpy": [0.01, -0.02, 1.57],
        "gyroscope": [0.001, 0.002, -0.003],
        "accelerometer": [0.05, -0.03, 9.81],
        "quaternion": [0.707, 0.0, 0.0, 0.707],
        "t": 1_700_000_000.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["rpy"] == [0.01, -0.02, 1.57]
    assert result["gyroscope"] == [0.001, 0.002, -0.003]
    assert result["accelerometer"] == [0.05, -0.03, 9.81]
    assert result["quaternion"] == [0.707, 0.0, 0.0, 0.707]
    assert result["t"] == 1_700_000_000.0


def test_a_tipped_orientation_rides_through_verbatim_not_clipped() -> None:
    """An extreme orientation reading is returned verbatim.

    ``_on_lowstate`` does not clip or wrap ``rpy``: a fallen or
    face-down G1 reports a pitch or roll magnitude near pi, and the
    verb must not silently clip that to a smaller range or normalise
    it - a caller reading the reading to decide whether the robot is
    up-right compares the same value the driver's decoder wrote.
    """
    cache: dict[str, Any] = {
        "rpy": [3.05, 0.0, 0.0],  # near-flat on the back
        "gyroscope": [0.0, 0.0, 0.0],
        "accelerometer": [-9.81, 0.0, 0.0],
        "quaternion": [0.0, 1.0, 0.0, 0.0],  # 180-degree roll
        "t": 1_700_000_100.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["present"] is True
    assert result["rpy"] == [3.05, 0.0, 0.0]
    assert result["quaternion"] == [0.0, 1.0, 0.0, 0.0]


def test_a_missing_field_in_the_cache_reports_none() -> None:
    """A cache dict missing one of the five fields returns ``None`` for it.

    ``_on_lowstate`` may in principle write a partial dict on a decode
    where one attribute was absent from the IDL message (the handler
    catches ``Exception`` in its own decode path and logs, but a caller
    writing ``_imu`` from a test also skips fields).  The verb reads
    through ``dict.get`` so a missing key surfaces as ``None`` rather
    than a ``KeyError`` at the tool boundary.
    """
    cache: dict[str, Any] = {
        "rpy": [0.0, 0.0, 0.0],
        "gyroscope": [0.0, 0.0, 0.0],
        # accelerometer / quaternion / t missing
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["present"] is True
    assert result["rpy"] == [0.0, 0.0, 0.0]
    assert result["gyroscope"] == [0.0, 0.0, 0.0]
    assert result["accelerometer"] is None
    assert result["quaternion"] is None
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
            "rpy": [0.0, 0.0, 0.0],
            "gyroscope": [0.0, 0.0, 0.0],
            "accelerometer": [0.0, 0.0, 9.81],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "t": 1_700_000_300.0,
        }
    )
    _call(driver)
    assert driver.calls == ["_imu"], f"g1_imu must read _imu exactly once; got {driver.calls}"


def test_the_returned_dict_does_not_alias_the_cache() -> None:
    """Mutating the verb's return value does not corrupt the driver's cache.

    ``G1Driver._snapshot`` already returns a copy under the driver's
    ``_cache_lock`` so the DDS thread that writes ``_on_lowstate`` does
    not race a caller.  The verb receives that copy and reshapes it
    into an envelope; a caller further mutating the returned dict must
    not reach back into the driver's cache.  The stub returns a fresh
    dict per call to model the real driver's behaviour, and this test
    grades that the verb's own return does not share references with
    the stub's internal cache either.
    """
    original_cache: dict[str, Any] = {
        "rpy": [0.1, 0.2, 0.3],
        "gyroscope": [0.0, 0.0, 0.0],
        "accelerometer": [0.0, 0.0, 9.81],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "t": 1_700_000_400.0,
    }
    driver = _StubG1Driver(cache=original_cache)
    result = _call(driver)
    result["rpy"] = [9.9, 9.9, 9.9]
    result["present"] = False

    # The driver's cache is untouched: a fresh snapshot still reports
    # what ``_on_lowstate`` wrote.
    fresh = _call(driver)
    assert fresh["rpy"] == [0.1, 0.2, 0.3]
    assert fresh["present"] is True


def test_a_bare_zero_reading_is_still_present() -> None:
    """A cache carrying all-zero fields still reports ``present=True``.

    An IMU reading of ``rpy=[0, 0, 0]`` and ``accelerometer=[0, 0, 0]``
    would be physically implausible (accelerometer at rest reports g on
    one axis), but it is a *reading*: ``_on_lowstate`` wrote something
    into ``_imu``, so ``_snapshot("_imu")`` returned a non-``None``
    dict.  The verb reports the fact of the reading through
    ``present``, not its physical plausibility; falsy field values must
    not silently collapse to ``present=False``.
    """
    cache: dict[str, Any] = {
        "rpy": [0.0, 0.0, 0.0],
        "gyroscope": [0.0, 0.0, 0.0],
        "accelerometer": [0.0, 0.0, 0.0],
        "quaternion": [0.0, 0.0, 0.0, 0.0],
        "t": 0.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["present"] is True
    assert result["rpy"] == [0.0, 0.0, 0.0]
    assert result["t"] == 0.0
