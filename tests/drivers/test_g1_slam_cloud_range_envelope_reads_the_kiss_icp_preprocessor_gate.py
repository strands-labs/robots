"""The cloud range envelope lookup names what the neon SLAM kiss-icp preprocessor admits.

The neon bundle's SLAM runner
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._make_odometry``)
constructs a ``kiss_icp.config.KISSConfig`` and hands it to
``kiss_icp.kiss_icp.KissICP`` with ``cfg.data.min_range = 1.0`` and
``cfg.data.max_range = 40.0``; kiss-icp's frame preprocessor drops
any point whose Euclidean distance from the sensor origin sits
outside ``[1.0, 40.0]`` before the ICP registration sees the
frame.  The
:mod:`strands_robots.tools.g1.g1_slam_cloud_range_envelope` module
snapshots that ``[1.0, 40.0]`` band into module-level constants
and exposes two agent-facing verbs -
:func:`g1_list_slam_cloud_range_envelope` (name the whole
envelope) and :func:`g1_slam_cloud_range_admits` (decide one
query) - so a caller can decide the refusal decidably before a
future driver-side SLAM ingest wrapper fires.  The tests here fix
that contract without pulling the SDK: the module is loadable on
a host without ``unitree_sdk2py`` and without ``kiss_icp`` (the
same SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off
the module's own snapshot rather than restated in the tests, so a
widen or narrow to the observed band surfaces here as a shape
change rather than as a diverging table this file would need to
manually update.

Two things this file's cells deliberately do not pin:

* The kiss-icp preprocessor's own answer at wire time.  The
  envelope is the neon runner's observed band, not the Livox
  Mid-360's manuf. spec (the sensor's near limit is about 0.1 m
  but the neon runner's ``min_range`` narrows that to 1.0 m).  A
  driver-side wrapper for the SLAM ingest that lands later will
  run the preprocessor itself and interpret the surviving points
  at wire time; the downstream shape checks the preprocessor
  applies (finiteness, deskew) are frame-space refusals that
  need the cloud's per-point buffer on hand and are not decided
  here.
* Whether the driver's SLAM runner has an active odometry state.
  That is a driver-instance state read that belongs on the SLAM
  runner's own ``get_status``-style query; the envelope answers
  the per-point range half only, in the same way
  ``g1_slam_icp_fitness_envelope`` (``strands-labs/robots#3008``)
  answers the fitness half without also reading the driver's
  live ``fsm_id``.
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

import pytest

from strands_robots.tools.g1.g1_slam_cloud_range_envelope import (
    _CLOUD_MAX_RANGE_M,
    _CLOUD_MIN_RANGE_M,
    g1_list_slam_cloud_range_envelope,
    g1_slam_cloud_range_admits,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py`` or ``kiss_icp``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent (refs strands-labs/robots#358);
    a module that pulled a submodule at import time would break
    every headless CI runner and Thor before an office bring-up.
    The ``kiss_icp`` dependency the neon SLAM runner uses at wire
    time is also refused here - a caller who imports this
    envelope should not pay the ``kiss_icp`` weight, because this
    module makes no preprocessor call.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_slam_cloud_range_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower() or "kiss_icp" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_slam_cloud_range_envelope imports "
        f"pulled SDK / kiss_icp submodules: {leaked}. The rule for this "
        f"package is that the SDK loads only inside function bodies "
        f"(refs strands-labs/robots#358)."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """Every scalar is a finite float and the min/max chain is ordered.

    A non-finite bound would let
    :func:`g1_slam_cloud_range_admits` admit every value on that
    dimension; an out-of-order chain (``min > max``) would let a
    caller pass the near-clip gate on a value the far-clip would
    refuse or vice versa.  Pinned so a widen or narrow that
    inverts the chain surfaces here rather than as a silently
    unreachable envelope in production.
    """
    for name, value in (
        ("_CLOUD_MIN_RANGE_M", _CLOUD_MIN_RANGE_M),
        ("_CLOUD_MAX_RANGE_M", _CLOUD_MAX_RANGE_M),
    ):
        assert math.isfinite(value), f"{name} is not finite: {value!r}"

    assert _CLOUD_MIN_RANGE_M < _CLOUD_MAX_RANGE_M, (
        f"range chain inverted: min={_CLOUD_MIN_RANGE_M}, "
        f"max={_CLOUD_MAX_RANGE_M}. g1_slam_cloud_range_admits would "
        f"refuse every value if min >= max."
    )


def test_the_neon_bounds_match_the_observed_snapshot() -> None:
    """The bounds match the observed neon-runner values of ``1.0`` and ``40.0``.

    The neon bundle's ``KISSConfig.data.min_range`` is ``1.0`` and
    ``KISSConfig.data.max_range`` is ``40.0`` at the snapshot this
    port is based on
    (``cagataycali/neon-the-g1/tools/g1_slam.py``).  Pinned here so
    a re-snapshot of the neon bundle that widened or narrowed the
    band surfaces as a diff on this test rather than as a silently
    drifting admission gate.
    """
    assert _CLOUD_MIN_RANGE_M == 1.0
    assert _CLOUD_MAX_RANGE_M == 40.0


def test_g1_list_slam_cloud_range_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every scalar and the single refusal reason.

    ``envelope`` carries both range constants; ``refusals`` names
    the single reason the admits path will surface
    (``range_outside_kiss_icp_preprocessor`` for a value the
    preprocessor drops before the ICP sees the frame).
    """
    result = _call(g1_list_slam_cloud_range_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["cloud_min_range_m"] == _CLOUD_MIN_RANGE_M
    assert env["cloud_max_range_m"] == _CLOUD_MAX_RANGE_M

    refusal_reasons = {r["reason"] for r in result["refusals"]}
    assert refusal_reasons == {"range_outside_kiss_icp_preprocessor"}


def test_g1_slam_cloud_range_admits_a_value_inside_the_band() -> None:
    """A range strictly inside the band is admitted with route=neon_slam_ingest.

    The identity case (``range_m=5.0``, comfortably inside the
    ``[1.0, 40.0]`` band) is admitted; ``route`` names
    ``"neon_slam_ingest"`` to identify the pipeline branch a
    future driver-side wrapper would take.
    """
    result = _call(g1_slam_cloud_range_admits, range_m=5.0)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "neon_slam_ingest"
    assert result["refusals"] == []


def test_g1_slam_cloud_range_admits_at_the_min_boundary() -> None:
    """A range *equal* to the near-clip is admitted, not refused.

    The neon runner's kiss-icp preprocessor comparison against
    ``min_range`` is non-strict at the boundary in every published
    release the neon bundle targets, so equality is admitted;
    refusing at the boundary would drop the neon bundle's own
    admitted edge.  Pinned because an off-by-one that turned the
    non-strict comparison into a strict one would silently reject
    every point at exactly ``1.0`` m.
    """
    result = _call(g1_slam_cloud_range_admits, range_m=_CLOUD_MIN_RANGE_M)
    assert result["admits"] is True
    assert result["route"] == "neon_slam_ingest"
    assert result["refusals"] == []


def test_g1_slam_cloud_range_admits_at_the_max_boundary() -> None:
    """A range *equal* to the far-clip is admitted, not refused.

    The neon runner's kiss-icp preprocessor comparison against
    ``max_range`` is non-strict at the boundary in every published
    release the neon bundle targets, so equality is admitted.
    Pinned so an off-by-one that refused the far-clip edge
    surfaces here.
    """
    result = _call(g1_slam_cloud_range_admits, range_m=_CLOUD_MAX_RANGE_M)
    assert result["admits"] is True
    assert result["route"] == "neon_slam_ingest"
    assert result["refusals"] == []


def test_g1_slam_cloud_range_admits_below_the_min() -> None:
    """A range below the near-clip refuses on ``cloud_min_range_m``.

    The refusal descriptor names ``dimension="range_m"``, the
    offending value, the bound it violated
    (``bound_key="cloud_min_range_m"``), the comparison
    (``"value < bound"``), and the neon-runner reason
    (``"range_outside_kiss_icp_preprocessor"``).  ``route`` is
    ``None`` because a rejected point would not reach the
    ``neon_slam_ingest`` path at wire time.
    """
    below = 0.5
    result = _call(g1_slam_cloud_range_admits, range_m=below)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "range_m"
    assert r["value"] == below
    assert r["bound_key"] == "cloud_min_range_m"
    assert r["bound"] == _CLOUD_MIN_RANGE_M
    assert r["comparison"] == "value < bound"
    assert r["reason"] == "range_outside_kiss_icp_preprocessor"


def test_g1_slam_cloud_range_admits_above_the_max() -> None:
    """A range above the far-clip refuses on ``cloud_max_range_m``.

    Points farther than 40.0 m are outside the neon runner's
    admitted band and refuse with
    ``bound_key="cloud_max_range_m"`` and ``comparison="value >
    bound"``.  Pinned because a silent drift to a wider far-clip
    would let unfiltered far returns through to the ICP.
    """
    over = 50.0
    result = _call(g1_slam_cloud_range_admits, range_m=over)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "range_m"
    assert r["value"] == over
    assert r["bound_key"] == "cloud_max_range_m"
    assert r["bound"] == _CLOUD_MAX_RANGE_M
    assert r["comparison"] == "value > bound"
    assert r["reason"] == "range_outside_kiss_icp_preprocessor"


def test_g1_slam_cloud_range_admits_at_zero_refuses_on_min() -> None:
    """A range of ``0.0`` refuses on ``cloud_min_range_m`` (a point at the origin).

    A point at exactly the sensor origin sits below the neon
    runner's near-clip, so the refusal names
    ``bound_key="cloud_min_range_m"`` — the neon config treats
    the origin as a preprocessor drop, not a shape violation
    against a positive-only clamp.
    """
    result = _call(g1_slam_cloud_range_admits, range_m=0.0)
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["bound_key"] == "cloud_min_range_m"
    assert r["reason"] == "range_outside_kiss_icp_preprocessor"


def test_g1_slam_cloud_range_admits_a_negative_range_refuses_on_min() -> None:
    """A negative range refuses on ``cloud_min_range_m``, not a separate shape bound.

    A negative Euclidean distance is not a value the sensor
    would produce, but the envelope answers the neon runner's
    admission decision only (which is a single band gate); the
    refusal surfaces as a ``cloud_min_range_m`` violation with
    ``comparison="value < bound"``, quoting the same reason the
    preprocessor would.
    """
    under = -1.0
    result = _call(g1_slam_cloud_range_admits, range_m=under)
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "range_m"
    assert r["value"] == under
    assert r["bound_key"] == "cloud_min_range_m"
    assert r["bound"] == _CLOUD_MIN_RANGE_M
    assert r["comparison"] == "value < bound"
    assert r["reason"] == "range_outside_kiss_icp_preprocessor"


@pytest.mark.parametrize("bad_range", [math.inf, -math.inf, math.nan])
def test_g1_slam_cloud_range_admits_refuses_non_finite_input(
    bad_range: float,
) -> None:
    """``math.inf`` / ``-math.inf`` / ``math.nan`` refuse with ``comparison="non-finite"``.

    A NaN cannot be compared decidably (``nan < 1.0`` is
    ``False`` but so is ``nan > 40.0``), and an infinity is not
    a distance the preprocessor would ever admit either.  Named
    on the refusal descriptor so a caller distinguishes a bounds
    violation from a shape violation.  The bound the refusal
    names is ``cloud_max_range_m`` by convention — the same
    convention ``g1_slam_icp_fitness_envelope``
    (``strands-labs/robots#3008``) uses for its non-finite refusals.
    """
    result = _call(g1_slam_cloud_range_admits, range_m=bad_range)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "range_m"
    assert r["comparison"] == "non-finite"
    assert r["reason"] == "range_outside_kiss_icp_preprocessor"
    assert r["bound_key"] == "cloud_max_range_m"


def test_g1_slam_cloud_range_admits_default_call_is_the_min_boundary() -> None:
    """Calling with no arguments admits at the near-clip boundary.

    The verb's default ``range_m=1.0`` sits at the exact
    neon-runner near-clip, so a zero-arg invocation ("does the
    minimum-admitted range admit?") returns ``admits=True,
    route="neon_slam_ingest"`` with an empty refusals list.
    Pinned so a change to the default that silently shifted the
    boundary-query answer surfaces here.
    """
    result = _call(g1_slam_cloud_range_admits)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "neon_slam_ingest"
    assert result["refusals"] == []


def test_the_admits_envelope_matches_the_list_envelope() -> None:
    """The two verbs surface the same envelope descriptor.

    A widen to the envelope in one verb but not the other would
    hand a caller comparing an admitted-path payload against the
    list verb's payload two diverging tables.  Pinned so any
    future change lands in :func:`_envelope` (the shared builder)
    rather than as a per-verb inlined copy.
    """
    listed = _call(g1_list_slam_cloud_range_envelope)["envelope"]
    admitted = _call(g1_slam_cloud_range_admits, range_m=5.0)["envelope"]
    assert listed == admitted
