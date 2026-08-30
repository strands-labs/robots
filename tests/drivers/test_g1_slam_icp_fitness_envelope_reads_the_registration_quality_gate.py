"""The ICP fitness envelope lookup names what the neon SLAM relocaliser admits.

The neon bundle's SLAM relocaliser
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._do_relocalize``)
runs Open3D point-to-point ICP against the currently-loaded map and
refuses the returned transform when ``result.fitness`` drops below
``0.3``.  Open3D's ``fitness`` is a dimensionless fraction in ``[0.0,
1.0]`` naming the share of source correspondences matched inside the
ICP inlier threshold.  The
:mod:`strands_robots.tools.g1.g1_slam_icp_fitness_envelope` module
snapshots that ``0.3`` threshold (plus the ``[0.0, 1.0]`` shape
bounds Open3D produces) into module-level constants and exposes two
agent-facing verbs -
:func:`g1_list_slam_icp_fitness_envelope` (name the whole envelope)
and :func:`g1_slam_icp_fitness_admits` (decide one query) - so a
caller can decide the refusal decidably before a future driver-side
SLAM relocalise wrapper fires.  The tests here fix that contract
without pulling the SDK: the module is loadable on a host without
``unitree_sdk2py`` and without ``open3d`` (the same SDK-load-hygiene
rule every other file under :mod:`strands_robots.tools.g1` carries,
refs strands-labs/robots#358), and every membership answer is read
off the module's own snapshot rather than restated in the tests, so
a widen or narrow to the observed threshold surfaces here as a shape
change rather than as a diverging table this file would need to
manually update.

Two things this file's cells deliberately do not pin:

* The Open3D ICP's own answer at wire time.  The envelope is the
  neon runner's observed threshold, not Open3D's own bounds (Open3D
  reports fitness in ``[0.0, 1.0]`` unconditionally).  A driver-side
  wrapper for the SLAM relocaliser that lands later will run the ICP
  itself and interpret ``result.fitness`` at wire time; the two
  additional transform-space refusals the neon runner applies
  (translation magnitude ``> 50.0`` m and negative rotation-trace)
  are not decided here because they are transform-space refusals
  that need the ICP output on hand.
* Whether the driver's map cache has an active map loaded.  That is
  a driver-instance state read that belongs on the SLAM runner's
  own ``get_status``-style query; the envelope answers the fitness
  half only, in the same way
  :mod:`~strands_robots.tools.g1.g1_swing_height_envelope` answers
  the value half without also reading the driver's live ``fsm_id``.
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

import pytest

from strands_robots.tools.g1.g1_slam_icp_fitness_envelope import (
    _ICP_FITNESS_MAX,
    _ICP_FITNESS_MIN,
    _ICP_FITNESS_THRESHOLD,
    g1_list_slam_icp_fitness_envelope,
    g1_slam_icp_fitness_admits,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py`` or ``open3d``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent (refs strands-labs/robots#358); a
    module that pulled a submodule at import time would break every
    headless CI runner and Thor before an office bring-up.  The
    ``open3d`` dependency the neon SLAM runner uses at wire time is
    also refused here - a caller who imports this envelope should
    not pay the ``open3d`` weight, because this module makes no ICP
    call.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_slam_icp_fitness_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower() or "open3d" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_slam_icp_fitness_envelope imports "
        f"pulled SDK / open3d submodules: {leaked}. The rule for this "
        f"package is that the SDK loads only inside function bodies "
        f"(refs strands-labs/robots#358)."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """Every scalar is a finite float and the min/threshold/max chain is ordered.

    A non-finite bound would let
    :func:`g1_slam_icp_fitness_admits` admit every value on that
    dimension; an out-of-order chain would let a caller pass the
    threshold gate on a value the ICP shape check would refuse or
    vice versa.  Pinned so a widen or narrow that inverts the chain
    surfaces here rather than as a silently unreachable envelope in
    production.
    """
    for name, value in (
        ("_ICP_FITNESS_MIN", _ICP_FITNESS_MIN),
        ("_ICP_FITNESS_MAX", _ICP_FITNESS_MAX),
        ("_ICP_FITNESS_THRESHOLD", _ICP_FITNESS_THRESHOLD),
    ):
        assert math.isfinite(value), f"{name} is not finite: {value!r}"

    assert _ICP_FITNESS_MIN <= _ICP_FITNESS_THRESHOLD <= _ICP_FITNESS_MAX, (
        f"fitness chain inverted: min={_ICP_FITNESS_MIN}, "
        f"threshold={_ICP_FITNESS_THRESHOLD}, max={_ICP_FITNESS_MAX}. "
        f"g1_slam_icp_fitness_admits would surface disjoint refusal "
        f"reasons for values on the same side of the chain."
    )


def test_the_open3d_bounds_match_the_dimensionless_fraction_range() -> None:
    """Open3D's fitness is a dimensionless fraction in [0.0, 1.0].

    The envelope quotes ``0.0`` and ``1.0`` as the bounds of Open3D's
    ``result.fitness``.  Pinned here because a change to those bounds
    would either be an Open3D API contract change (unlikely - fitness
    is a fraction by construction) or a bug in this envelope, and
    either way the failure surface here is the right place to catch
    it.
    """
    assert _ICP_FITNESS_MIN == 0.0
    assert _ICP_FITNESS_MAX == 1.0


def test_the_neon_threshold_matches_the_observed_snapshot() -> None:
    """The threshold matches the observed neon-runner value of ``0.3``.

    The neon bundle's ``_ICP_FITNESS_THRESHOLD`` is ``0.3`` at the
    snapshot this port is based on
    (``cagataycali/neon-the-g1/tools/g1_slam.py``).  Pinned here so
    a re-snapshot of the neon bundle that widened or narrowed the
    threshold surfaces as a diff on this test rather than as a
    silently drifting admission gate.
    """
    assert _ICP_FITNESS_THRESHOLD == 0.3


def test_g1_list_slam_icp_fitness_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every scalar and the two refusal reasons.

    ``envelope`` carries every fitness constant; ``refusals`` names
    the two refusal reasons the admits path will surface
    (``fitness_below_threshold`` for a value inside the ICP's own
    bounds but below the neon-runner admission threshold, and
    ``fitness_outside_open3d_range`` for a value the ICP itself
    does not produce).
    """
    result = _call(g1_list_slam_icp_fitness_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["icp_fitness_min"] == _ICP_FITNESS_MIN
    assert env["icp_fitness_max"] == _ICP_FITNESS_MAX
    assert env["icp_fitness_threshold"] == _ICP_FITNESS_THRESHOLD

    refusal_reasons = {r["reason"] for r in result["refusals"]}
    assert refusal_reasons == {
        "fitness_below_threshold",
        "fitness_outside_open3d_range",
    }


def test_g1_slam_icp_fitness_admits_a_value_above_the_threshold() -> None:
    """A fitness above the threshold is admitted with route=neon_relocalize.

    The identity case (``fitness=0.5``, comfortably above the ``0.3``
    admission threshold and well inside the ``[0.0, 1.0]`` shape
    bounds) is admitted; ``route`` names ``"neon_relocalize"`` to
    identify the pipeline branch a future driver-side wrapper would
    take.
    """
    result = _call(g1_slam_icp_fitness_admits, fitness=0.5)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "neon_relocalize"
    assert result["refusals"] == []


def test_g1_slam_icp_fitness_admits_at_the_exact_threshold_boundary() -> None:
    """A fitness *equal* to the threshold is admitted, not refused.

    The neon runner's own comparison is ``result.fitness <
    _ICP_FITNESS_THRESHOLD`` (strict-less-than), so equality is
    admitted; refusing at the boundary would drop the neon bundle's
    own admitted edge.  Pinned because an off-by-one that turned
    the strict comparison into a non-strict one would silently
    reject every fitness at exactly ``0.3``.
    """
    result = _call(g1_slam_icp_fitness_admits, fitness=_ICP_FITNESS_THRESHOLD)
    assert result["admits"] is True
    assert result["route"] == "neon_relocalize"
    assert result["refusals"] == []


def test_g1_slam_icp_fitness_admits_at_the_open3d_upper_bound() -> None:
    """A fitness *equal* to ``icp_fitness_max`` (``1.0``) is admitted.

    ``1.0`` names every source correspondence matched by Open3D's
    inlier threshold - the strongest ICP registration the runner
    could ever produce, and the neon runner's threshold gate admits
    it (``1.0 >= 0.3``).  Pinned so an off-by-one that refused the
    strongest possible registration surfaces here.
    """
    result = _call(g1_slam_icp_fitness_admits, fitness=_ICP_FITNESS_MAX)
    assert result["admits"] is True
    assert result["route"] == "neon_relocalize"
    assert result["refusals"] == []


def test_g1_slam_icp_fitness_admits_below_the_threshold() -> None:
    """A fitness below the neon threshold refuses on ``icp_fitness_threshold``.

    The refusal descriptor names ``dimension="fitness"``, the
    offending value, the bound it violated
    (``bound_key="icp_fitness_threshold"``), the comparison
    (``"value < bound"``), and the neon-runner reason
    (``"fitness_below_threshold"``).  ``route`` is ``None`` because
    a rejected value would not reach the ``neon_relocalize`` path
    at wire time.
    """
    below = 0.1
    result = _call(g1_slam_icp_fitness_admits, fitness=below)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "fitness"
    assert r["value"] == below
    assert r["bound_key"] == "icp_fitness_threshold"
    assert r["bound"] == _ICP_FITNESS_THRESHOLD
    assert r["comparison"] == "value < bound"
    assert r["reason"] == "fitness_below_threshold"


def test_g1_slam_icp_fitness_admits_at_open3d_zero() -> None:
    """A fitness of ``0.0`` refuses on ``icp_fitness_threshold``, not on the shape bound.

    ``0.0`` is a value Open3D produces (no correspondences matched
    within the inlier threshold), so it sits *inside* the ``[0.0,
    1.0]`` shape bounds; the refusal is a *quality* refusal against
    the neon runner's threshold, not a shape refusal against
    Open3D's own range.  Pinned so the reason surfaces as
    ``"fitness_below_threshold"`` rather than the shape
    ``"fitness_outside_open3d_range"`` at the edge case.
    """
    result = _call(g1_slam_icp_fitness_admits, fitness=_ICP_FITNESS_MIN)
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["reason"] == "fitness_below_threshold"
    assert r["bound_key"] == "icp_fitness_threshold"


def test_g1_slam_icp_fitness_admits_above_the_open3d_ceiling() -> None:
    """A fitness above ``icp_fitness_max`` refuses on the shape bound.

    Open3D does not produce fitness values above ``1.0``, so a
    caller who passed one supplied a shape the ICP would never
    surface.  The refusal descriptor names
    ``bound_key="icp_fitness_max"`` and
    ``reason="fitness_outside_open3d_range"`` so a caller
    distinguishes a shape violation from a quality one.
    """
    over = _ICP_FITNESS_MAX + 0.5
    result = _call(g1_slam_icp_fitness_admits, fitness=over)
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "fitness"
    assert r["value"] == over
    assert r["bound_key"] == "icp_fitness_max"
    assert r["bound"] == _ICP_FITNESS_MAX
    assert r["comparison"] == "value > bound"
    assert r["reason"] == "fitness_outside_open3d_range"


def test_g1_slam_icp_fitness_admits_below_the_open3d_floor() -> None:
    """A strictly-negative fitness refuses on the shape bound, not the threshold.

    A negative fitness is not a value Open3D produces (the fraction
    is bounded below by zero).  The refusal descriptor names
    ``bound_key="icp_fitness_min"`` and
    ``reason="fitness_outside_open3d_range"`` because the shape
    violation takes priority over the quality one - a value the
    ICP would never surface is not a value that has a
    quality-vs-threshold question to answer.
    """
    under = -0.1
    result = _call(g1_slam_icp_fitness_admits, fitness=under)
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "fitness"
    assert r["value"] == under
    assert r["bound_key"] == "icp_fitness_min"
    assert r["bound"] == _ICP_FITNESS_MIN
    assert r["comparison"] == "value < bound"
    assert r["reason"] == "fitness_outside_open3d_range"


@pytest.mark.parametrize("bad_fitness", [math.inf, -math.inf, math.nan])
def test_g1_slam_icp_fitness_admits_refuses_non_finite_input(
    bad_fitness: float,
) -> None:
    """``math.inf`` / ``-math.inf`` / ``math.nan`` refuse with ``comparison="non-finite"``.

    A NaN cannot be compared decidably (``nan < 0.3`` is ``False``
    but so is ``nan >= 0.3``), and an infinity is not a valid
    fraction - both are shape violations rather than value ones.
    Named on the refusal descriptor so a caller distinguishes a
    bounds violation from a shape violation.
    """
    result = _call(g1_slam_icp_fitness_admits, fitness=bad_fitness)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "fitness"
    assert r["comparison"] == "non-finite"
    assert r["reason"] == "fitness_outside_open3d_range"
    assert r["bound_key"] == "icp_fitness_max"


def test_g1_slam_icp_fitness_admits_default_call_is_the_threshold_boundary() -> None:
    """Calling with no arguments admits at the threshold boundary.

    The verb's default ``fitness=0.3`` sits at the exact neon-runner
    admission threshold, so a zero-arg invocation ("does the
    minimum-admitted fitness admit?") returns ``admits=True,
    route="neon_relocalize"`` with an empty refusals list.  Pinned
    so a change to the default that silently shifted the
    boundary-query answer surfaces here.
    """
    result = _call(g1_slam_icp_fitness_admits)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "neon_relocalize"
    assert result["refusals"] == []


def test_the_admits_envelope_matches_the_list_envelope() -> None:
    """The two verbs surface the same envelope descriptor.

    A widen to the envelope in one verb but not the other would
    hand a caller comparing an admitted-path payload against the
    list verb's payload two diverging tables.  Pinned so any
    future change lands in :func:`_envelope` (the shared builder)
    rather than as a per-verb inlined copy.
    """
    listed = _call(g1_list_slam_icp_fitness_envelope)["envelope"]
    admitted = _call(g1_slam_icp_fitness_admits, fitness=0.5)["envelope"]
    assert listed == admitted
