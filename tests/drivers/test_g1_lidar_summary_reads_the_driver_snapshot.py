"""``g1_lidar_summary`` returns exactly what ``G1Driver._snapshot("_lidar_summary")`` gives it.

``g1_lidar_summary`` is the fifth driver-instance-taking verb in
:mod:`strands_robots.tools.g1`, after ``g1_get_state``
(``strands-labs/robots#2934``), ``g1_battery`` (``strands-labs/robots#2938``),
``g1_imu`` (``strands-labs/robots#2939``) and ``g1_lidar_state``
(``strands-labs/robots#2941``). Every earlier read-only verb in the package
(``g1_joint_reference``, ``g1_list_motion_gates``, ``g1_fsm_admits``) is a
pure reader over module-level constants; this one and its four siblings all
take a driver instance and read its cache through a named accessor -
:meth:`G1Driver._snapshot` for the ``rt/utlidar/cloud_livox_mid360`` summary.
The tests here fix that contract by handing a hand-rolled driver double to
the verb and asserting the returned dict names each field
``_on_lidar_cloud`` writes into ``_lidar_summary``.

The cache field names are read here off the driver's own writer
(:meth:`G1Driver._on_lidar_cloud` names ``count`` / ``width`` / ``height`` /
``point_step`` / ``row_step`` / ``t``) rather than being restated in the
tests, so a widen or rename on the driver side moves both the write path and
this verb together. What the tests do restate is the SDK-load-hygiene
contract every file under :mod:`strands_robots.tools.g1` carries: importing
the module must not pull any ``unitree_sdk2py`` submodule. And that
``count`` is the cloud's uncapped size (``strands-labs/robots#2752``) - a
MID-360 that drops from 24000 points to 3000 is reporting a fault, and
clamping it would hide exactly that.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_lidar_summary import g1_lidar_summary


class _StubG1Driver:
    """A driver double whose ``_snapshot`` returns a fixed cache dict.

    ``g1_lidar_summary`` calls ``driver._snapshot("_lidar_summary")`` and
    reads the six fields ``_on_lidar_cloud`` writes into that dict. This
    double sits under the same interface without pulling the real driver's
    imports (the real class reaches CycloneDDS at construction time in some
    paths), so a test can hand a wired-shape cache to the verb without a
    bus. A ``None`` cache models the just-connected state where
    ``_on_lidar_cloud`` has not fired yet.
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
    return g1_lidar_summary(driver=driver)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable with
    the SDK absent; a module that pulled a submodule at import time would
    break every headless CI runner and Thor before an office bring-up. The
    driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only path
    that loads the SDK); this cell holds the lidar-summary verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_lidar_summary")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_lidar_summary imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_with_no_cloud_message_yet_reports_absent() -> None:
    """A ``_snapshot`` returning ``None`` becomes ``present=False``.

    ``_on_lidar_cloud`` writes ``_lidar_summary`` on every
    ``rt/utlidar/cloud_livox_mid360`` decode, and ``_lidar_summary`` starts
    as ``None`` on driver init. A just-connected driver whose cloud has not
    arrived yet gives back ``None`` from the snapshot accessor, and the
    verb must report that decidably rather than fabricating a ``count`` of
    ``0`` (which a MID-360 whose scan had stopped would also read as) or
    zero point/row steps (which would look like a byte-broken frame).
    """
    driver = _StubG1Driver(cache=None)
    result = _call(driver)

    assert result["status"] == "success"
    assert result["present"] is False
    assert result["count"] is None
    assert result["width"] is None
    assert result["height"] is None
    assert result["point_step"] is None
    assert result["row_step"] is None
    assert result["t"] is None
    assert driver.calls == ["_lidar_summary"], (
        f"g1_lidar_summary must read exactly _lidar_summary from the cache; got {driver.calls}"
    )


def test_a_full_frame_reports_every_field_the_summariser_wrote() -> None:
    """Each of the six ``_on_lidar_cloud`` fields rides through unchanged.

    ``_on_lidar_cloud`` writes ``count`` (the cloud's true size as
    ``width * height``), ``width``, ``height``, ``point_step`` (bytes per
    point), ``row_step`` (bytes per row) and ``t`` (wall time of decode).
    The verb is not the place to reword or convert those - a caller
    reading this summary against a health chip on the mesh reads the same
    integers the driver wrote.
    """
    cache: dict[str, Any] = {
        "count": 24000,
        "width": 24000,
        "height": 1,
        "point_step": 16,
        "row_step": 24000 * 16,
        "t": 1_700_000_000.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["count"] == 24000
    assert result["width"] == 24000
    assert result["height"] == 1
    assert result["point_step"] == 16
    assert result["row_step"] == 24000 * 16
    assert result["t"] == 1_700_000_000.0


def test_a_sparse_frame_reports_the_true_uncapped_count() -> None:
    """A cloud that dropped to 3000 points reports 3000, not a cap.

    ``strands-labs/robots#2752`` landed the rule that this record's
    ``count`` is the cloud's true, uncapped size. A MID-360 whose scan
    dropped from 24000 to 3000 points is reporting a fault, and a
    ``capped_at``-style clamp beside the count would hide exactly that.
    The verb must surface the number ``_on_lidar_cloud`` wrote,
    verbatim, so a health chip reading it against the healthy-frame
    baseline (24000) can raise on the drop.
    """
    cache: dict[str, Any] = {
        "count": 3000,
        "width": 3000,
        "height": 1,
        "point_step": 16,
        "row_step": 3000 * 16,
        "t": 1_700_000_001.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["count"] == 3000, (
        "g1_lidar_summary must report the uncapped count; clamping it would hide the "
        "very fault strands-labs/robots#2752 was landed to surface."
    )
    assert result["width"] == 3000
    assert result["height"] == 1


def test_an_organised_cloud_reports_width_times_height() -> None:
    """A ``height > 1`` cloud round-trips as ``count = width * height``.

    A MID-360 reports ``height=1``, but the driver's summary formula is
    ``width * height`` on purpose so an *organised* cloud (say a depth
    camera republishing on the same topic in a test) is read correctly
    rather than dropping the second dimension. The verb passes the
    driver's number through as-is; the width/height fields ride alongside
    for a caller who wants to know which axis carried what.
    """
    cache: dict[str, Any] = {
        "count": 640 * 480,
        "width": 640,
        "height": 480,
        "point_step": 16,
        "row_step": 640 * 16,
        "t": 1_700_000_002.0,
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["count"] == 640 * 480
    assert result["width"] == 640
    assert result["height"] == 480


def test_a_partial_cache_reports_none_for_missing_fields() -> None:
    """A cache missing a field surfaces ``None`` for it, not a made-up value.

    ``_on_lidar_cloud`` writes every field on every decode, but the verb
    reads through ``dict.get`` rather than direct indexing so a cache that
    a future driver refactor writes as a subset does not raise here. What
    ``None`` names in that case is exactly that the driver did not write
    the field; a caller who needs the missing field cannot get it from
    this side and has to reach for the driver's own writer, which is
    where the omission lives.
    """
    cache: dict[str, Any] = {
        "count": 24000,
        "width": 24000,
        "height": 1,
        # point_step, row_step and t deliberately absent
    }
    result = _call(_StubG1Driver(cache=cache))

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["count"] == 24000
    assert result["width"] == 24000
    assert result["height"] == 1
    assert result["point_step"] is None
    assert result["row_step"] is None
    assert result["t"] is None


def test_the_verb_reads_the_snapshot_exactly_once() -> None:
    """One ``_snapshot("_lidar_summary")`` call per verb invocation.

    ``_snapshot`` holds the driver's ``_cache_lock`` for the copy; a
    verb that read the accessor twice on the same call would take that
    lock twice, and a call that raced the DDS thread between the two
    reads could return two dicts naming different times. The verb reads
    once and reshapes the result, so a caller sees a self-consistent
    snapshot even under a busy bus.
    """
    driver = _StubG1Driver(
        cache={
            "count": 24000,
            "width": 24000,
            "height": 1,
            "point_step": 16,
            "row_step": 24000 * 16,
            "t": 1_700_000_000.0,
        }
    )
    _call(driver)

    assert driver.calls == ["_lidar_summary"], (
        f"expected exactly one _snapshot('_lidar_summary') call; got {driver.calls}"
    )


def test_the_returned_dict_does_not_alias_the_cache() -> None:
    """Mutating the verb's return value does not corrupt the driver cache.

    The driver's ``_snapshot`` copies the underlying dict so the DDS
    thread's next write does not race a reader mutating what it just
    handed out. This test grades the same rule end-to-end through the
    verb: a caller that appends to ``count`` or edits ``t`` on the
    returned dict must not see the driver's next ``_snapshot`` reflect
    the change.
    """
    cache: dict[str, Any] = {
        "count": 24000,
        "width": 24000,
        "height": 1,
        "point_step": 16,
        "row_step": 24000 * 16,
        "t": 1_700_000_000.0,
    }
    driver = _StubG1Driver(cache=cache)
    result = _call(driver)
    result["count"] = 999
    result["t"] = 0.0

    fresh = driver._snapshot("_lidar_summary")
    assert fresh is not None
    assert fresh["count"] == 24000, "verb return aliased the driver cache"
    assert fresh["t"] == 1_700_000_000.0, "verb return aliased the driver cache"
