"""Agent-facing lookup for the ICP-relocalise gate the neon SLAM bundle admits a snap through.

The neon bundle's SLAM runner
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._try_relocalize``)
takes a candidate ICP registration between the live LiDAR frame and a
loaded map and DECIDES three things at once before it lets the fitted
transform ``T`` snap the runner's pose offset onto the map frame: the
open3d fitness score has to clear a floor
(``_ICP_FITNESS_THRESHOLD = 0.3``), the fitted translation magnitude
must not cross a sanity ceiling (``np.linalg.norm(T[:3, 3]) > 50.0``),
and the rotation trace must be non-negative
(``(T[0, 0] + T[1, 1] + T[2, 2]) < 0.0``).  Each is a distinct
refusal shape: a fitness below the floor means the ICP did not find
enough correspondence to trust, a translation above the ceiling means
the fit is aliased onto a distant part of the map (or is a numerically
degenerate transform), and a negative rotation trace means the fit is
a reflection or a large enough rotation that at least two of the
diagonal cosines went negative -- ``trace(R) = 1 + 2*cos(theta)`` for a
proper rotation, so ``trace < 0`` means ``cos(theta) < -0.5`` i.e.
``|theta| > 120 deg``, and that shape is exactly what a symmetric
alignment axis (an ICP flipping the map's front and back walls) or an
improper rotation (a reflection) produces.  The neon runner refuses
outside any one of them; a caller planning a relocalise wants each
answered on its own so the refusal string names which of the three
went wrong.

This module snapshots the observed envelope into module-level
constants and exposes two agent-facing verbs so a caller can decide
the refusal decidably before a future driver-side SLAM relocalise
path is called, rather than pinning the range inside the write path
where the refusal is invisible to the planner.

Twin of a future ``g1_slam_map_names`` lookup that surfaces the *name*
dimension on the same SLAM surface rather than the *match-quality*
dimension.  The two modules stay separate because a name is authored
ahead of a save/load and the match is a decision against a live ICP
result: two different surfaces with disjoint refusal shapes.
Colocating them here would hand an agent planner a single refusal
payload that mixed the two remedies -- \"choose a different name\"
versus \"rebuild the map or move closer\" -- and would tie a future
ICP-parameter revision (open3d version, voxel-size, correspondence
distance) to a name-rule revision the neon bundle does not couple.

Two things this module is deliberately *not*:

* An execution path.  The neon bundle's ``_try_relocalize`` runs a
  full open3d ICP registration on the deciding host's GPU/CPU under
  ``_SlamRunner._process_frame``'s thread; that write is the
  SLAM-runner-side of the neon bundle's pose pipeline, which today's
  :class:`~strands_robots.drivers.g1.G1Driver` does not front.  A
  future driver method that fronts SLAM relocalise will land the
  transition verb; refs strands-labs/robots#358 for the SDK-facing
  gate work that write belongs on.  This module ports the read-only
  envelope half without also introducing a second SLAM path the
  driver does not yet own.
* An SDK re-import.  The envelope is captured here as module-level
  constants so ``import strands_robots.tools.g1.g1_slam_relocalize_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no ``numpy``,
  ``open3d``, or ``kiss_icp`` submodule at import time -- the
  import-hygiene contract every other file in this package carries,
  refs strands-labs/robots#358.  A caller authoring a relocalise plan
  before any SLAM extra is installed on their host still gets the
  envelope back verbatim.  A revision of the observed bounds is a
  driver-side update; when the driver's SLAM relocalise method
  lands, its refusal will surface the same module-local
  :data:`_REFUSAL_TEXT` this module names for a match-quality
  violation.

Why this module does not quote a driver-side ``rc``.

The G1 driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``); its FSM
rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``\"Invalid FSM id - need FSM in {500, 501, 801}\"``).  The SLAM
relocalise decision runs on the neon runner's own thread against an
in-memory ICP result -- it never touches ``rt/lowcmd``, never talks to
the locomotion controller, and reaches no SDK RPC service that ships
an rc table for a match-quality refusal.  Borrowing ``7404`` on a
fitness-below-floor refusal would hand an agent planner a
motion-FSM remedy (``\"need FSM in {500, 501, 801}\"``) for a match
that has nothing to do with the locomotion FSM.  The refusal shape
this module returns names the numeric bound violation in
module-local text so a planner reads a remedy that matches the
surface, and a future driver-side SLAM relocalise wrapper will
surface the same module-local text -- not a re-borrowed motion code.
This mirrors the same-surface refusal rule
:mod:`~strands_robots.tools.g1.g1_dds_max_buffer_envelope` names for
``g1_dds_subscribe``'s ``max_buffer`` argument, refs
strands-labs/robots#358.

What this module does not decide.

* Whether the map that the ICP is fitting to is loaded, non-empty,
  or freshly filtered.  The neon runner's own guard refuses ICP on
  a map of fewer than 100 points (``map_pts is None or
  len(map_pts) < 100``), which is a live-runner state read; this
  envelope names the *result*-side gate on top of that liveness
  check.  A caller planning a relocalise combines both -- the map
  liveness (from a future SLAM-liveness verb) and this envelope --
  before the driver's write path is entered.
* Whether the fitted translation is *plausible* given the robot's
  own motion history.  A 30 m jump when the robot has moved 0.5 m
  is a caller-side plausibility question (drift vs teleport) that
  this envelope does not answer; the ``50 m`` translation ceiling
  refuses only fits that are obviously numerically degenerate or
  aliased across the room, not the softer \"is this a plausible
  drift correction\" one.  A future runner-side plausibility check
  is a separate surface with its own envelope.
* What ICP parameters the neon runner used.  The neon
  ``_try_relocalize`` runs a fixed pipeline (voxel-downsample to
  0.5 m, ``TransformationEstimationPointToPoint``, max
  correspondence distance ``1.0``) that produces the ``fitness``
  this module tests against.  A future runner rewrite that swapped
  those parameters would produce a different fitness distribution
  and would motivate a re-tune of :data:`_FITNESS_MIN`; the
  envelope here names the fitness floor the *current* pipeline
  admits, not a scale-invariant match-quality bound.
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool

#: The lower clamp on the ICP fitness score (inclusive float).  The
#: open3d :class:`~open3d.pipelines.registration.RegistrationResult`'s
#: ``fitness`` field reports the fraction of source points with a
#: correspondence inside ``max_correspondence_distance`` in the
#: target, so a value of ``0.3`` means at least 30% of the live
#: LiDAR downsample matched something in the loaded map at 1 m.
#: The neon bundle's ``g1_slam.py`` names this floor as
#: ``_ICP_FITNESS_THRESHOLD = 0.3`` and refuses any registration
#: below it before the fitted transform is applied.  Named as an
#: inclusive lower bound (``fitness >= 0.3`` admits) because the
#: neon check reads ``result.fitness < _ICP_FITNESS_THRESHOLD`` and
#: refuses only on strictly less, so a fit at exactly the floor is
#: the boundary case the runner admits and this envelope quotes
#: verbatim.
_FITNESS_MIN: float = 0.3

#: The upper clamp on the ICP fitness score.  Fitness is a fraction
#: bounded above by ``1.0`` (every source point has a
#: correspondence), so a value above ``1.0`` is a shape violation
#: rather than a match-quality one -- the neon runner's own
#: fitness comparison would admit it (``0.3`` floor only), but the
#: value is out of the score's domain and a caller passing it is
#: making a shape mistake.  Named here so :func:`g1_slam_relocalize_admits`
#: refuses ``fitness=1.5`` at the shape boundary before the
#: match-quality question is asked, rather than admitting a value
#: that no real ICP result could produce and later trip on the
#: unrelated translation or trace check.
_FITNESS_MAX: float = 1.0

#: The upper clamp on the fitted translation magnitude, in metres.
#: The neon runner reads
#: ``np.linalg.norm(T[:3, 3]) > 50.0`` as \"the ICP snapped the pose
#: to a point 50+ metres away from the runner's own pose origin,
#: which is either aliased across a room-sized map or a numerically
#: degenerate transform\".  ``50 m`` is far outside the neon-bundle-
#: observed indoor mapping range (kiss-icp's own ``max_range`` in
#: the runner is ``40.0`` m, so a fit that moved the pose further
#: than any single LiDAR return could have observed is definitionally
#: an alias).  Named as an inclusive upper bound (``translation_m
#: <= 50.0`` admits, ``> 50.0`` refuses) to match the neon check's
#: strict-greater refusal.
_TRANSLATION_MAX_M: float = 50.0

#: The lower clamp on the fitted rotation trace, dimensionless.  The
#: neon runner reads
#: ``(T[0, 0] + T[1, 1] + T[2, 2]) < 0.0`` as \"the rotation
#: component's trace is negative\".  For a proper rotation matrix
#: the trace is ``1 + 2 * cos(theta)`` where ``theta`` is the angle
#: of rotation, so ``trace >= 0`` admits every rotation up to
#: ``120 deg`` and ``trace < 0`` refuses anything larger -- and a
#: reflection (a numerically degenerate T with ``det = -1``) also
#: shows up here because the diagonal cosines flip sign on the
#: reflected axes.  Named as an inclusive lower bound
#: (``rotation_trace >= 0.0`` admits) to match the neon check's
#: strict-less refusal.
_ROTATION_TRACE_MIN: float = 0.0

#: The upper clamp on the rotation trace.  A proper rotation
#: matrix's trace is bounded above by ``3.0`` (the identity
#: matrix's trace, i.e. ``theta = 0``), so a value above ``3.0`` is
#: a shape violation rather than a rotation-magnitude one.  Named
#: here so :func:`g1_slam_relocalize_admits` refuses
#: ``rotation_trace=5.0`` at the shape boundary rather than
#: admitting a value that no real rotation matrix could produce.
_ROTATION_TRACE_MAX: float = 3.0

#: The module-local refusal text every ``g1_slam_relocalize_admits``
#: refusal quotes when the caller's argument sits outside one of
#: the three neon-runner-observed envelopes.  Named here rather
#: than borrowed from
#: :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` because
#: the SLAM relocalise path ships no distinct rc -- the neon
#: runner just returns ``None`` from ``_try_relocalize`` on a
#: refused match, which the caller reads as \"no offset applied
#: this frame\".  The motion-FSM ``7404`` entry (its nearest
#: neighbour) reads ``\"Invalid FSM id - need FSM in {500, 501,
#: 801}\"`` -- a remedy that points a planner at locomotion FSM
#: transitions to fix a match-quality argument.  Surfacing the
#: module-local text keeps the refusal payload's remedy on the
#: same surface the write belongs on, and a future driver-side
#: SLAM relocalise wrapper will surface this same text rather
#: than re-borrowing a motion code.
_REFUSAL_TEXT: str = (
    "relocalise gate refused - one or more of fitness/translation/trace "
    "sits outside the neon-runner-observed envelope. Refs "
    "strands-labs/robots#358."
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_slam_relocalize_envelope` so
    :func:`g1_slam_relocalize_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place.  Every field is a snapshot read; no bus is touched.
    """
    return {
        "fitness_min": _FITNESS_MIN,
        "fitness_max": _FITNESS_MAX,
        "translation_max_m": _TRANSLATION_MAX_M,
        "rotation_trace_min": _ROTATION_TRACE_MIN,
        "rotation_trace_max": _ROTATION_TRACE_MAX,
    }


@tool
def g1_list_slam_relocalize_envelope() -> dict[str, Any]:
    """Return the ICP-relocalise envelope the neon SLAM runner admits a snap through.

    Read-only.  No driver instance, no DDS, no SDK, no ``numpy`` /
    ``open3d`` / ``kiss_icp`` submodule import at load time: every
    field is a module-level constant.  Useful before a future
    driver-side wrapper for SLAM relocalise is called, so a caller
    can compare an intended (fitness, translation, trace) triple
    against the envelope the neon runner's ``_try_relocalize``
    refuses outside of and can carry the module-local refusal text
    a driver-side wrapper would surface on any bound violation.

    Three dimensions ride the same envelope because the neon runner
    reads them together and returns ``None`` on any one being
    outside its range (ICP fitness below floor, fitted translation
    magnitude above ceiling, rotation matrix trace below floor).
    Colocating the three in one envelope lets a caller planning a
    relocalise ask one verb and receive one payload that names all
    three refusal shapes, mirroring how the neon runner itself
    reads them.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        every clamp the neon runner observed
        (``fitness_min``, ``fitness_max``, ``translation_max_m``,
        ``rotation_trace_min``, ``rotation_trace_max``); and a
        ``refusals`` list carrying a single descriptor with the
        module-local :data:`_REFUSAL_TEXT` a future write verb
        would surface on a bounds violation.  Every field is a
        snapshot of an observed bound or a module-local text; no
        dynamic decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"text": _REFUSAL_TEXT},
        ],
    }


def _finite(value: float) -> bool:
    """Return whether ``value`` is a finite float.

    Kept here rather than pulled from ``strands_robots.utils``
    because the envelope check needs only the finiteness half of a
    validator; the positivity half does not apply to the rotation
    trace (which is a signed quantity, admitted at zero) or to the
    fitness lower clamp (which is a fraction, also admitted at zero
    but refused there by the match-quality floor, not by
    positivity).  A future consolidation with the shared validator
    lands when the driver-side write verb reuses this admits
    function.
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _reject(
    refusals: list[dict[str, Any]],
    dimension: str,
    value: Any,
    bound_key: str,
    bound: float,
    cmp: str,
) -> None:
    """Append one refusal descriptor to ``refusals``.

    Kept here rather than inlined at each rejection site so the
    refusal shape is one dict-literal in one place and a widen to
    it (a new field, a rewording of the module-local text) lands in
    one place.  The refusal names the dimension, the offending
    value, the clamp it violated, and the shared :data:`_REFUSAL_TEXT`
    a driver-side wrapper would surface.
    """
    refusals.append(
        {
            "dimension": dimension,
            "value": value,
            "bound_key": bound_key,
            "bound": bound,
            "comparison": cmp,
            "text": _REFUSAL_TEXT,
        }
    )


@tool
def g1_slam_relocalize_admits(
    fitness: float = 1.0,
    translation_m: float = 0.0,
    rotation_trace: float = 3.0,
) -> dict[str, Any]:
    """Decide whether an ICP result sits inside the relocalise gate.

    Read-only.  Compares the three arguments against the clamps
    :func:`g1_list_slam_relocalize_envelope` returns and reports
    every bound the triple violates.  No driver instance, no DDS,
    no SDK, no ``numpy`` / ``open3d`` / ``kiss_icp`` submodule
    import: the decision reads only module-level constants and
    the arguments themselves.

    A triple inside the envelope is *not* the same as an admitted
    snap: the neon runner also refuses on the map-liveness check
    (``map_pts is None or len(map_pts) < 100``), which is a
    live-runner state read this verb does not answer (that is a
    future SLAM-liveness companion verb).  The returned payload
    names only the numeric bound decision.

    All three dimensions are graded, so a triple with two
    violations reports both refusals; a caller reading only the
    first would re-plan against a fitness floor and still trip on
    the translation ceiling.  The neon runner's own check
    short-circuits on the first refusal, but a caller planning a
    relocalise wants every violation named so their next attempt
    fixes the whole triple.

    Args:
        fitness: open3d ICP fitness score in ``[0.0, 1.0]``.  The
            default ``1.0`` (perfect fit) matches an ICP result
            that would definitively admit, so a caller who does
            not pass an explicit argument for this dimension lands
            on the admitted side of the fitness clamp -- letting
            them probe the other two dimensions in isolation.
            Refused below ``fitness_min`` (the neon-runner-observed
            match-quality floor; below it the ICP did not find
            enough correspondence to trust the fit) and above
            ``fitness_max`` (a shape violation; a fraction cannot
            exceed ``1.0``).  Boolean values are refused explicitly
            because Python's ``bool`` is a subclass of ``int``, so
            a caller passing ``True`` would otherwise silently look
            up ``1.0`` (a legitimate perfect-fit fitness) and hide
            the type mistake.  Non-finite floats
            (``math.inf``, ``math.nan``) are refused with a
            ``non-finite`` comparison because a NaN cannot be
            compared decidably (``nan < 0.3`` is ``False`` but so
            is ``nan >= 0.3``) and an infinity would land outside
            the score's domain.
        translation_m: fitted translation magnitude in metres,
            i.e. ``||T[:3, 3]||``.  The default ``0.0`` (no
            translation, i.e. the identity fit) admits, so a
            caller who does not pass an explicit argument for
            this dimension lands on the admitted side of the
            translation clamp.  Refused above ``translation_max_m``
            (the neon-runner-observed alias ceiling; above it the
            ICP snapped to a point further than any single LiDAR
            return could have observed).  Negative values are
            refused with the same shape because a magnitude
            cannot be negative -- a caller who computed
            ``T[:3, 3]`` and forgot the ``norm`` would otherwise
            probe the ceiling with a signed value.
        rotation_trace: fitted rotation matrix trace, i.e.
            ``T[0, 0] + T[1, 1] + T[2, 2]``.  The default ``3.0``
            (the identity rotation) admits, so a caller who does
            not pass an explicit argument for this dimension
            lands on the admitted side of the trace clamp.
            Refused below ``rotation_trace_min`` (the
            neon-runner-observed sign floor; below it the fit is
            either a rotation past ``120 deg`` or a reflection,
            both of which the runner treats as degenerate) and
            above ``rotation_trace_max`` (a shape violation; a
            proper rotation matrix's trace cannot exceed ``3.0``).
            Boolean and non-finite values are refused for the same
            reason as ``fitness``.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        every dimension is inside its clamp pair; a ``refusals``
        list of refusal descriptors, each carrying the dimension
        name, the offending value, the clamp it violated, the
        comparison ("value < bound", "value > bound", or
        "non-finite"), and the module-local :data:`_REFUSAL_TEXT`
        a driver-side wrapper would surface if the relocalise
        were attempted while any dimension is outside the
        envelope; the same ``envelope`` sub-dict
        :func:`g1_list_slam_relocalize_envelope` returns.  On an
        admitted triple the ``refusals`` list is empty; on a
        rejected triple every violated bound is named so the
        caller sees the whole shape of the refusal at once.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    # Fitness dimension: refuse bool first (subclasses int/float via
    # __bool__ = int subclass), then non-finite, then bounds.
    if isinstance(fitness, bool):
        _reject(refusals, "fitness", fitness, "fitness_min", _FITNESS_MIN, "non-int")
    elif not _finite(fitness):
        _reject(refusals, "fitness", fitness, "fitness_min", _FITNESS_MIN, "non-finite")
    else:
        f = float(fitness)
        if f < _FITNESS_MIN:
            _reject(refusals, "fitness", fitness, "fitness_min", _FITNESS_MIN, "value < bound")
        elif f > _FITNESS_MAX:
            _reject(refusals, "fitness", fitness, "fitness_max", _FITNESS_MAX, "value > bound")

    # Translation dimension: same order.
    if isinstance(translation_m, bool):
        _reject(
            refusals,
            "translation_m",
            translation_m,
            "translation_max_m",
            _TRANSLATION_MAX_M,
            "non-int",
        )
    elif not _finite(translation_m):
        _reject(
            refusals,
            "translation_m",
            translation_m,
            "translation_max_m",
            _TRANSLATION_MAX_M,
            "non-finite",
        )
    else:
        t = float(translation_m)
        # A magnitude cannot be negative; refuse strictly-negative
        # values as shape violations with a distinct bound key so
        # the refusal reads decidably against the ceiling refusal.
        if t < 0.0:
            _reject(
                refusals,
                "translation_m",
                translation_m,
                "translation_magnitude_floor",
                0.0,
                "value < bound",
            )
        elif t > _TRANSLATION_MAX_M:
            _reject(
                refusals,
                "translation_m",
                translation_m,
                "translation_max_m",
                _TRANSLATION_MAX_M,
                "value > bound",
            )

    # Rotation-trace dimension: same order.
    if isinstance(rotation_trace, bool):
        _reject(
            refusals,
            "rotation_trace",
            rotation_trace,
            "rotation_trace_min",
            _ROTATION_TRACE_MIN,
            "non-int",
        )
    elif not _finite(rotation_trace):
        _reject(
            refusals,
            "rotation_trace",
            rotation_trace,
            "rotation_trace_min",
            _ROTATION_TRACE_MIN,
            "non-finite",
        )
    else:
        r = float(rotation_trace)
        if r < _ROTATION_TRACE_MIN:
            _reject(
                refusals,
                "rotation_trace",
                rotation_trace,
                "rotation_trace_min",
                _ROTATION_TRACE_MIN,
                "value < bound",
            )
        elif r > _ROTATION_TRACE_MAX:
            _reject(
                refusals,
                "rotation_trace",
                rotation_trace,
                "rotation_trace_max",
                _ROTATION_TRACE_MAX,
                "value > bound",
            )

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
