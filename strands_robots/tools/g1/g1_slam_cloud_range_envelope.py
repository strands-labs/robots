"""Agent-facing lookup for the per-point range envelope the neon SLAM ingest admits.

The neon bundle's SLAM runner
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._make_odometry``)
constructs a ``kiss_icp.config.KISSConfig`` and hands it to
``kiss_icp.kiss_icp.KissICP`` before any Livox frame is processed.
Two of that config's ``data`` fields are per-point radial gates the
kiss-icp frame preprocessor applies before it hands the cloud to the
ICP: ``cfg.data.min_range = 1.0`` (metres, inclusive lower bound on
``||p||_2`` for a point ``p`` to be kept) and ``cfg.data.max_range =
40.0`` (metres, inclusive upper bound).  A point whose Euclidean
distance from the sensor origin sits outside ``[1.0, 40.0]`` is
dropped by kiss-icp's preprocessor before the ICP registration ever
sees it.  This module snapshots the two bounds into module-level
constants and exposes two agent-facing verbs so a caller can decide
the refusal decidably before a future driver-side wrapper for the
SLAM ingest is called, rather than pinning the value inside the
frame-preprocessor path where the refusal is invisible to the
planner.

Twin of ``g1_slam_icp_fitness_envelope``
(``strands-labs/robots#3008``, the *registration-quality* dimension
on the same SLAM surface -- the fitness scalar the ICP returns after
the frame has already been range-filtered; the sibling is still open
for review, therefore named as a literal, not a dotted path role) and
:mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope`
(``strands-labs/robots#3006``, the ICP *voxel-size* dimension on the
map-side relocalise path).  The three modules stay separate because
a per-point range filter is a *pre-ICP frame-preprocessor* gate on
the raw cloud, a voxel size is a *pre-ICP subsample* argument on
the source cloud, and a fitness is a *post-ICP quality* judgement
on a registration that already ran; three different surfaces with
disjoint refusal shapes.  Colocating them would hand an agent
planner a single refusal payload that mixed the frame-preprocessor,
subsample and quality remedies and would tie any future kiss-icp
release to a caller-side pipeline revision the neon bundle does not
couple.

Two things this module is deliberately *not*:

* An execution path.  The neon bundle's SLAM runner ran the actual
  frame preprocessor inside kiss-icp against the Livox cloud stream
  and interpreted the surviving points; the write end of that
  pipeline is reached today by
  ``g1_slam_icp_fitness_admits`` (``strands-labs/robots#3008``) on
  the post-registration quality dimension and by future driver-side
  work on the frame-ingest dispatch itself.  This module ports the
  read-only envelope half without also introducing a second SLAM
  ingest writer path the driver does not yet own.  Refs
  strands-labs/robots#358.
* An SDK / kiss-icp re-import.  The two range bounds are captured
  here as module-level constants so ``import
  strands_robots.tools.g1.g1_slam_cloud_range_envelope`` pulls no
  ``unitree_sdk2py`` submodule and no ``kiss_icp`` submodule -- the
  import-hygiene contract every other file in this package carries
  (refs strands-labs/robots#358).  The ``kiss_icp`` module ships as
  an optional dependency of the neon SLAM runner; a caller who
  wants to run the actual frame preprocessor reaches the neon
  bundle's own runner path, which imports ``kiss_icp`` inside the
  runner's function bodies and not at module load.

What this module does not decide.

* Whether the ICP will actually admit the frame's *surviving*
  points.  A pre-flight range admission is a check the caller can
  make against each point's distance before dispatching a frame;
  kiss-icp's frame preprocessor may still drop a surviving point
  on downstream shape checks (finiteness, deskew) which are
  frame-space refusals that need the cloud's per-point buffer on
  hand.  ``g1_slam_cloud_range_admits`` answers the *radial
  distance* half only, in the same way
  ``g1_slam_icp_fitness_envelope`` (``strands-labs/robots#3008``)
  answers the fitness half without also reading the ICP's live
  transform.
* Whether the caller-supplied range value is one a Livox Mid-360
  cloud would ever carry.  The Livox Mid-360 has its own manuf.
  spec range (about 0.1 m near limit and about 40 m far limit);
  the envelope here is the *kiss-icp preprocessor's* admitted
  range and is narrower on the near end (1.0 m vs. 0.1 m) --
  points inside the sensor's near limit but below kiss-icp's
  ``min_range`` are refused as *quality* violations, not shape
  ones, because they are values the sensor produced but that the
  neon runner's config drops.  A distinction future driver-side
  work reads at the ingest boundary; this envelope quotes the
  neon-runner config, not the sensor spec.
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool

#: The inclusive lower bound on ``||p||_2`` (Euclidean distance from
#: the sensor origin, metres) a point must satisfy to be kept by
#: kiss-icp's frame preprocessor.  ``1.0`` names the near-clip the
#: neon runner's ``KISSConfig.data.min_range`` sets; a point closer
#: than 1.0 m to the sensor is dropped before the ICP registration
#: ever sees it.  Named as an *inclusive* minimum-admitted range: a
#: distance *equal to* ``1.0`` is admitted (the kiss-icp
#: preprocessor's own comparison against ``min_range`` is
#: non-strict at the boundary in every published release the neon
#: bundle targets); refusing at the boundary would drop the neon
#: bundle's own admitted edge.
_CLOUD_MIN_RANGE_M: float = 1.0

#: The inclusive upper bound on ``||p||_2`` a point must satisfy to
#: be kept by kiss-icp's frame preprocessor.  ``40.0`` names the
#: far-clip the neon runner's ``KISSConfig.data.max_range`` sets; a
#: point farther than 40.0 m from the sensor is dropped before the
#: ICP registration ever sees it.  Named as an *inclusive*
#: maximum-admitted range: a distance *equal to* ``40.0`` is
#: admitted (the kiss-icp preprocessor's own comparison against
#: ``max_range`` is non-strict at the boundary in every published
#: release the neon bundle targets); refusing at the boundary would
#: drop the neon bundle's own admitted edge.
_CLOUD_MAX_RANGE_M: float = 40.0


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_slam_cloud_range_envelope` so
    :func:`g1_slam_cloud_range_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place.  Every field is a snapshot read; no bus is touched.
    """
    return {
        "cloud_min_range_m": _CLOUD_MIN_RANGE_M,
        "cloud_max_range_m": _CLOUD_MAX_RANGE_M,
    }


@tool
def g1_list_slam_cloud_range_envelope() -> dict[str, Any]:
    """Return the per-point range envelope the neon SLAM ingest admits.

    Read-only.  No driver instance, no DDS, no SDK, no kiss-icp
    submodule: every field is a module-level constant.  Useful
    before a future driver-side wrapper for the SLAM frame ingest
    is called, so a caller can compare an intended per-point
    distance (for example against a Livox return the caller wants
    to plan against, or the median range of a captured frame) to
    the neon bundle's own admission range.

    The envelope names two scalars.  ``cloud_min_range_m`` is the
    kiss-icp preprocessor's near-clip (``1.0`` m, inclusive), and
    ``cloud_max_range_m`` is the far-clip (``40.0`` m, inclusive);
    a point whose Euclidean distance from the sensor origin sits
    outside ``[cloud_min_range_m, cloud_max_range_m]`` is dropped
    by the preprocessor before the ICP registration is fed the
    frame.  Values inside the range are admitted; values outside
    are refused with the neon-runner reason string
    ``range_outside_kiss_icp_preprocessor``.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        the two range scalars (``cloud_min_range_m``,
        ``cloud_max_range_m``); a ``refusals`` list carrying the
        single refusal reason this envelope surfaces
        (``range_outside_kiss_icp_preprocessor`` for a value the
        preprocessor drops before the ICP sees the frame).  Every
        field is a snapshot of an observed neon-runner constant;
        no dynamic decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {
                "reason": "range_outside_kiss_icp_preprocessor",
                "text": (
                    "Per-point range outside the neon SLAM runner's "
                    "kiss-icp preprocessor admitted range; the "
                    "preprocessor drops the point before the ICP "
                    "registration sees the frame"
                ),
            },
        ],
    }


def _finite(value: float) -> bool:
    """Return whether ``value`` is a finite float.

    Kept here rather than pulled from ``strands_robots.utils``
    because the envelope check needs only the finiteness half of a
    validator; the positivity half would refuse ``0.0`` on a
    kiss-icp preprocessor whose own comparison at the near-clip is
    a non-strict ``>=`` (a point at exactly the origin is refused
    as a *quality* violation against ``min_range``, not as a shape
    one against a positive-only clamp).  A future consolidation
    with the shared validator lands when the driver-side write
    verb reuses this admits function.
    """
    return math.isfinite(float(value))


@tool
def g1_slam_cloud_range_admits(range_m: float = 1.0) -> dict[str, Any]:
    """Decide whether a per-point range sits inside the neon preprocessor's admitted band.

    Read-only.  Compares the argument against
    :data:`_CLOUD_MIN_RANGE_M` and :data:`_CLOUD_MAX_RANGE_M` and
    reports whether the neon bundle's SLAM runner's kiss-icp
    preprocessor would keep the point or drop it before the ICP
    registration sees the frame.  No driver instance, no DDS, no
    SDK, no kiss-icp submodule: the decision reads only
    module-level constants and the argument itself.

    Two refusal shapes are decided against a single reason:

    * ``range_m`` non-finite (``math.inf``, ``math.nan``,
      ``-math.inf``) -- refused with
      ``reason="range_outside_kiss_icp_preprocessor"`` and
      ``comparison="non-finite"``.  A NaN cannot be compared
      decidably (``nan < 1.0`` is ``False`` but so is ``nan >
      40.0``), and an infinity is not a distance the preprocessor
      would ever admit either.
    * ``range_m`` outside ``[cloud_min_range_m, cloud_max_range_m]``
      -- refused with
      ``reason="range_outside_kiss_icp_preprocessor"`` and a
      comparison naming which bound the value violated
      (``"value < bound"`` for a distance under the near-clip,
      ``"value > bound"`` for a distance above the far-clip).

    A value inside ``[cloud_min_range_m, cloud_max_range_m]`` is
    admitted (the kiss-icp preprocessor keeps the point); the
    returned ``route`` names ``"neon_slam_ingest"`` on an
    admission and ``None`` on a refusal so a caller distinguishes
    the pipeline branch a future write would take.

    Args:
        range_m: A per-point Euclidean distance from the sensor
            origin (metres).  The default ``1.0`` sits at the
            near-clip boundary -- a zero-arg invocation asks "does
            the minimum-admitted range admit?", and the answer is
            ``True`` (the kiss-icp preprocessor's comparison is
            non-strict at the boundary, so the boundary is
            admitted).

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the value would let the neon SLAM runner keep the point;
        a ``route`` string naming the pipeline branch a future
        write would take (``"neon_slam_ingest"`` on admission,
        ``None`` on a refusal); a ``refusals`` list carrying the
        refusal descriptors on a rejected value, each with the
        offending value, the bound it violated (``bound_key``),
        the comparison, and the neon-runner reason string; and
        the same ``envelope`` sub-dict
        :func:`g1_list_slam_cloud_range_envelope` returns.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []
    route: str | None = None

    def _reject(
        value: float,
        bound_key: str,
        bound: float,
        cmp: str,
    ) -> None:
        refusals.append(
            {
                "dimension": "range_m",
                "value": float(value) if _finite(value) else value,
                "bound_key": bound_key,
                "bound": bound,
                "comparison": cmp,
                "reason": "range_outside_kiss_icp_preprocessor",
            }
        )

    if not _finite(range_m):
        _reject(
            range_m,
            "cloud_max_range_m",
            _CLOUD_MAX_RANGE_M,
            "non-finite",
        )
    else:
        r = float(range_m)
        if r < _CLOUD_MIN_RANGE_M:
            _reject(
                range_m,
                "cloud_min_range_m",
                _CLOUD_MIN_RANGE_M,
                "value < bound",
            )
        elif r > _CLOUD_MAX_RANGE_M:
            _reject(
                range_m,
                "cloud_max_range_m",
                _CLOUD_MAX_RANGE_M,
                "value > bound",
            )
        else:
            route = "neon_slam_ingest"

    return {
        "status": "success",
        "admits": not refusals,
        "route": route,
        "refusals": refusals,
        "envelope": envelope,
    }
