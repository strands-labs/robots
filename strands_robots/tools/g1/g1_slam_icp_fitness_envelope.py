"""Agent-facing lookup for the ICP fitness envelope the neon SLAM relocaliser admits.

The neon bundle's SLAM relocaliser
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._do_relocalize``)
runs Open3D point-to-point ICP against the currently-loaded map and
refuses the returned transform when the registration's ``fitness``
scalar drops below ``0.3``.  Open3D's ``fitness`` is the fraction of
source correspondences inside the ICP inlier threshold -- a
dimensionless scalar in ``[0.0, 1.0]`` where ``1.0`` names every
sample correspondence matched and ``0.0`` names none matched.  The
neon runner also caps the returned transform on two additional
geometric shapes (translation magnitude ``> 50.0`` m and negative
trace of the rotation block), but only ``fitness`` is a scalar the
caller can decide on before the ICP even fires; the two geometric
caps are transform-space refusals that need the ICP output on hand.
This module snapshots the fitness threshold into a module-level
constant and exposes two agent-facing verbs so a caller can decide
the refusal decidably before a future driver-side SLAM relocalise
wrapper is called, rather than pinning the value inside the ICP
path where the refusal is invisible to the planner.

Twin of ``g1_slam_relocalize_envelope`` (the ICP *voxel-size*
dimension on the same relocalise surface, open for review as
strands-labs/robots#3006).  That sibling is named as a literal rather
than cross-referenced, because a Sphinx role promises an importable
dotted path and this tree does not carry that module until #3006
lands.  Twin also of
:mod:`~strands_robots.tools.g1.g1_lidar_max_points_envelope` (the
*downsample cap* on the frame the ICP consumes).  The three modules
stay separate because a fitness refusal is a *quality* judgement on
an ICP that already ran, a voxel size is a *pre-ICP subsample*
argument on the cloud, and a max-points cap is a *pre-parser
subsample* on the raw Livox frame; three different surfaces with
disjoint refusal shapes.  Colocating them would hand an agent planner
a single refusal payload that mixed the quality, subsample and
downsample remedies and would tie any future firmware or Open3D
revision to a caller-side pipeline revision the neon bundle does not
couple.

Two things this module is deliberately *not*:

* An execution path.  The neon bundle's SLAM relocaliser ran the
  actual ICP against ``self._map_dedup`` and interpreted the
  ``result.fitness`` scalar; the write end of that pipeline is
  reached today by ``g1_slam_relocalize_envelope`` on the
  caller-side subsample dimension (strands-labs/robots#3006) and by
  future driver-side work on the ICP dispatch itself.  This module
  ports the read-only envelope half without also introducing a
  second ICP writer path the driver does not yet own.  Refs
  strands-labs/robots#358.
* An SDK / Open3D re-import.  The fitness threshold is captured
  here as a module-level constant so ``import
  strands_robots.tools.g1.g1_slam_icp_fitness_envelope`` pulls no
  ``unitree_sdk2py`` submodule and no ``open3d`` submodule -- the
  import-hygiene contract every other file in this package carries
  (refs strands-labs/robots#358).  The ``open3d`` module ships as an
  optional dependency of the neon SLAM runner; a caller who wants
  to run the actual ICP reaches the neon bundle's own runner path,
  which imports ``open3d`` inside the runner's function bodies and
  not at module load.

What this module does not decide.

* Whether the ICP will actually fire.  A pre-flight fitness
  admission is a check the caller can make against the threshold
  before dispatching the relocalise; the ICP itself may still fail
  on the geometric transform caps (``translation > 50.0`` m or
  negative rotation trace) which are transform-space refusals that
  need the ICP output on hand.  ``g1_slam_icp_fitness_admits``
  answers the fitness half only, in the same way
  :mod:`~strands_robots.tools.g1.g1_swing_height_envelope` answers
  the value half without also reading the driver's live ``fsm_id``.
* Whether the caller-supplied fitness value is one the ICP would
  ever produce.  Open3D reports fitness in ``[0.0, 1.0]``; a
  caller who passes a value outside that range receives a
  ``non-finite`` or bounds refusal, but the verb does not
  otherwise second-guess the caller's arithmetic.  The bounds
  check is the same shape the ICP itself would surface: a fitness
  outside ``[0.0, 1.0]`` is not a value the ICP produced and would
  not be admitted by the relocaliser's threshold comparison anyway.
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool

#: The inclusive lower bound Open3D's ``result.fitness`` scalar
#: takes on.  ``0.0`` names no correspondences matched between the
#: source and the target within the ICP inlier threshold; a value
#: below zero is not a value Open3D produces and is refused as a
#: shape violation.
_ICP_FITNESS_MIN: float = 0.0

#: The inclusive upper bound Open3D's ``result.fitness`` scalar
#: takes on.  ``1.0`` names every source correspondence matched
#: to a target within the ICP inlier threshold; a value above one
#: is not a value Open3D produces and is refused as a shape
#: violation.
_ICP_FITNESS_MAX: float = 1.0

#: The neon SLAM relocaliser's refusal threshold on the ICP
#: fitness scalar.  A registration whose ``result.fitness`` drops
#: below ``0.3`` is returned as ``None`` by the neon runner's
#: ``_do_relocalize`` (``cagataycali/neon-the-g1/tools/g1_slam.py``),
#: which the caller reads as "the map does not match the current
#: cloud well enough to relocalise on".  Named as an *inclusive*
#: minimum-admitted fitness: a fitness *equal to* the threshold is
#: still admitted (the neon runner's own comparison is ``result.
#: fitness < _ICP_FITNESS_THRESHOLD``, strict-less-than, so equality
#: is admitted); refusing at the boundary would drop the neon
#: bundle's own admitted edge.
_ICP_FITNESS_THRESHOLD: float = 0.3


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_slam_icp_fitness_envelope` so
    :func:`g1_slam_icp_fitness_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place.  Every field is a snapshot read; no bus is touched.
    """
    return {
        "icp_fitness_min": _ICP_FITNESS_MIN,
        "icp_fitness_max": _ICP_FITNESS_MAX,
        "icp_fitness_threshold": _ICP_FITNESS_THRESHOLD,
    }


@tool
def g1_list_slam_icp_fitness_envelope() -> dict[str, Any]:
    """Return the ICP fitness envelope the neon SLAM relocaliser admits.

    Read-only.  No driver instance, no DDS, no SDK, no Open3D
    submodule: every field is a module-level constant.  Useful
    before a future driver-side wrapper for the SLAM relocaliser
    is called, so a caller can compare an intended fitness value
    (either the last observed relocalise's fitness read off a
    ``get_status``-style query, or a threshold the caller wants to
    plan against) to the neon bundle's own admission threshold.

    The envelope names three scalars.  ``icp_fitness_min`` and
    ``icp_fitness_max`` are the bounds Open3D's ``result.fitness``
    takes on ([0.0, 1.0] -- a dimensionless fraction of source
    correspondences matched within the ICP inlier threshold).
    ``icp_fitness_threshold`` is the neon SLAM relocaliser's
    minimum-admitted fitness (``0.3``); a registration whose fitness
    drops below this is returned as ``None`` by the runner's
    ``_do_relocalize``.  Values ``>= icp_fitness_threshold`` are
    admitted; values strictly less than the threshold are refused
    at the fitness half of the pipeline (the ICP itself still owns
    the transform-space refusals on translation magnitude and
    rotation-trace shape).

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        every fitness scalar (``icp_fitness_min``,
        ``icp_fitness_max``, ``icp_fitness_threshold``); a
        ``refusals`` list carrying the two shape / bounds refusal
        strings this envelope surfaces
        (``fitness_below_threshold`` for a value inside the ICP's
        own bounds but below the neon-runner admission threshold,
        and ``fitness_outside_open3d_range`` for a value the ICP
        itself does not produce).  Every field is a snapshot of
        an observed neon-runner constant; no dynamic decode runs
        here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {
                "reason": "fitness_below_threshold",
                "text": (
                    "ICP fitness below the neon SLAM relocaliser's "
                    "minimum admission threshold; the runner's "
                    "_do_relocalize returns None on this path"
                ),
            },
            {
                "reason": "fitness_outside_open3d_range",
                "text": (
                    "Open3D's result.fitness is a dimensionless "
                    "fraction in [0.0, 1.0]; a value outside this "
                    "range is a shape violation, not a value the "
                    "ICP produced"
                ),
            },
        ],
    }


def _finite(value: float) -> bool:
    """Return whether ``value`` is a finite float.

    Kept here rather than pulled from ``strands_robots.utils``
    because the envelope check needs only the finiteness half of a
    validator; the positivity half is already answered by the
    explicit ``_ICP_FITNESS_MIN`` clamp (``0.0``), and a shared
    positive-finite validator would refuse ``0.0`` which is a
    legitimate ICP output (no correspondences matched).  A future
    consolidation with the shared validator lands when the
    driver-side write verb reuses this admits function.
    """
    return math.isfinite(float(value))


@tool
def g1_slam_icp_fitness_admits(fitness: float = 0.3) -> dict[str, Any]:
    """Decide whether an ICP fitness sits at or above the neon runner's threshold.

    Read-only.  Compares the argument against
    :data:`_ICP_FITNESS_THRESHOLD` and reports whether the neon
    bundle's SLAM relocaliser would accept the registration or
    return ``None`` on it.  No driver instance, no DDS, no SDK, no
    Open3D submodule: the decision reads only module-level
    constants and the argument itself.

    Two refusal shapes are decided:

    * ``fitness`` non-finite (``math.inf``, ``math.nan``) or
      outside ``[icp_fitness_min, icp_fitness_max]`` (``[0.0,
      1.0]``) -- refused with ``reason="fitness_outside_open3d_range"``.
      Open3D does not produce fitness values outside its
      dimensionless-fraction range, so a caller who passed one
      supplied a shape the ICP would never surface.  A NaN cannot
      be compared decidably (``nan < 0.3`` is ``False`` but so is
      ``nan >= 0.3``), and an infinity is not a valid fraction.
    * ``fitness < icp_fitness_threshold`` (strictly less than) --
      refused with ``reason="fitness_below_threshold"``.  The neon
      runner's own comparison is ``result.fitness <
      _ICP_FITNESS_THRESHOLD`` (strict-less-than), so equality is
      admitted; refusing at the boundary would drop the neon
      bundle's own admitted edge.

    A value ``>= icp_fitness_threshold`` and inside the ICP's own
    bounds is admitted (the neon runner's threshold gate passes);
    the returned ``route`` names ``"neon_relocalize"`` on an
    admission and ``None`` on a refusal so a caller distinguishes
    the pipeline branch a future write would take.

    Args:
        fitness: An ICP fitness scalar (dimensionless fraction of
            source correspondences matched within Open3D's inlier
            threshold).  The default ``0.3`` sits at the neon
            runner's admission threshold -- a zero-arg invocation
            asks "does the minimum-admitted fitness admit?", and
            the answer is ``True`` (the neon runner uses
            strict-less-than, so the boundary is admitted).

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the value would let the neon runner return a transform
        rather than ``None``; a ``route`` string naming the
        pipeline branch a future write would take
        (``"neon_relocalize"`` on admission, ``None`` on refusal);
        a ``refusals`` list carrying the refusal descriptors on a
        rejected value, each with the offending value, the bound
        it violated (``bound_key``), the comparison, and the
        neon-runner reason string; and the same ``envelope``
        sub-dict :func:`g1_list_slam_icp_fitness_envelope` returns.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []
    route: str | None = None

    def _reject(
        value: float,
        bound_key: str,
        bound: float,
        cmp: str,
        reason: str,
    ) -> None:
        refusals.append(
            {
                "dimension": "fitness",
                "value": float(value) if _finite(value) else value,
                "bound_key": bound_key,
                "bound": bound,
                "comparison": cmp,
                "reason": reason,
            }
        )

    if not _finite(fitness):
        _reject(
            fitness,
            "icp_fitness_max",
            _ICP_FITNESS_MAX,
            "non-finite",
            "fitness_outside_open3d_range",
        )
    else:
        f = float(fitness)
        if f < _ICP_FITNESS_MIN:
            _reject(
                fitness,
                "icp_fitness_min",
                _ICP_FITNESS_MIN,
                "value < bound",
                "fitness_outside_open3d_range",
            )
        elif f > _ICP_FITNESS_MAX:
            _reject(
                fitness,
                "icp_fitness_max",
                _ICP_FITNESS_MAX,
                "value > bound",
                "fitness_outside_open3d_range",
            )
        elif f < _ICP_FITNESS_THRESHOLD:
            _reject(
                fitness,
                "icp_fitness_threshold",
                _ICP_FITNESS_THRESHOLD,
                "value < bound",
                "fitness_below_threshold",
            )
        else:
            route = "neon_relocalize"

    return {
        "status": "success",
        "admits": not refusals,
        "route": route,
        "refusals": refusals,
        "envelope": envelope,
    }
