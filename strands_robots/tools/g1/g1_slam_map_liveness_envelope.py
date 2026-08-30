"""Agent-facing lookup for the map-liveness floor the neon SLAM runner admits a relocalise through.

The neon bundle's SLAM runner
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._try_relocalize``)
decides three things at once on a candidate ICP registration before
it lets the fitted transform snap the runner's pose offset onto the
map frame -- the ICP fitness must clear a floor, the fitted
translation must not cross a ceiling, and the rotation trace must
be non-negative (the three-dimension envelope
:mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope` names).
Before any of those three decisions runs, the runner first refuses
on a *liveness* precondition:
``if _o3d is None or map_pts is None or len(map_pts) < 100: return None``.
The ``map_pts`` argument is the (N, K) LiDAR-return array the loaded
map carries; a value of ``None`` names an unloaded / unsaved map,
and a length below ``100`` names a map with too few point returns
for an ICP registration to have a chance at correspondence.  The
neon runner returns ``None`` on any of those three, which the
caller reads as "no offset applied this frame".

This module surfaces the point-count half of that liveness gate --
the ``len(map_pts) >= 100`` floor -- as an agent-facing lookup pair
so a caller planning a relocalise can decide the point-count
precondition decidably before a future driver-side wrapper is
called, rather than pinning the floor inside the write path where
the refusal is invisible to the planner.  The ``_o3d is None`` and
``map_pts is None`` halves belong on a future SLAM-liveness runner
surface (whether the runner is up, whether a map is loaded) rather
than this numeric envelope, because their refusals answer
different remedies -- "install the SLAM extra" and "load a map"
against "the loaded map is too sparse to fit against".

Twin of :mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope`
(the merged strands-labs/robots#3006), which names the
match-quality dimensions on the same
``_SlamRunner._try_relocalize`` gate.  The two modules stay
separate because ``_try_relocalize`` reads them in order: this
module's floor is a *precondition* on the caller's map argument
(refused before any ICP dispatch, without needing a driver
handle) and the twin's envelope is a *result*-side judgement on
what the ICP produced (a live open3d registration result the
runner reads under the SLAM extra).  Colocating them would hand
an agent planner a single refusal payload that mixed a
"build a bigger map before you try" remedy against a
"the fit was aliased across the room" remedy.  Colocating them
would also tie a future map-cardinality revision (a runner
patch that lowered the ``100`` floor because ``o3d`` grew a
sparser-cloud registration mode) to a match-quality revision the
neon runner does not couple.

Two things this module is deliberately *not*:

* An execution path.  The neon runner's ``_try_relocalize`` runs
  a full open3d ICP registration under
  ``_SlamRunner._process_frame``'s thread; today's
  :class:`~strands_robots.drivers.g1.G1Driver` does not front
  that write, so no motion admission is at stake at this
  precondition.  A future driver method that fronts SLAM
  relocalise will surface the same module-local
  :data:`_REFUSAL_TEXT` on a sparse-map refusal; this module
  ports the read-only precondition half without also introducing
  a second SLAM path the driver does not yet own, refs
  strands-labs/robots#358.
* An SDK re-import.  The floor is captured here as a module-level
  constant so ``import strands_robots.tools.g1.g1_slam_map_liveness_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no ``numpy``,
  ``open3d``, or ``kiss_icp`` submodule at import -- the
  import-hygiene contract every other file in this package
  carries, refs strands-labs/robots#358.  A caller authoring a
  relocalise plan before any SLAM extra is installed on their
  host still gets the floor back verbatim.

Why this module does not quote a driver-side ``rc``.

The G1 driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``); its
FSM rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``).  The SLAM
map-liveness decision runs on the neon runner's own thread
against an in-memory map array -- it never touches ``rt/lowcmd``,
never talks to the locomotion controller, and reaches no SDK RPC
service that ships an rc table for a sparse-map refusal.
Borrowing ``7404`` on a below-floor refusal would hand an agent
planner a motion-FSM remedy for a map-cardinality argument.  The
refusal shape this module returns names the numeric bound
violation in module-local text so a planner reads a remedy that
matches the surface, and a future driver-side SLAM relocalise
wrapper will surface the same module-local text.  This mirrors
the same-surface refusal rule
:mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope` names
for the ``_try_relocalize`` match-quality gate, refs
strands-labs/robots#358.

What this module does not decide.

* Whether the map is loaded at all.  ``map_pts is None`` and
  ``_o3d is None`` are the two other refusal shapes the neon
  runner's precondition names, and each carries a different
  remedy ("load a map" and "install the SLAM extra")
  than the point-count floor's ("build a bigger map").  A future
  SLAM-liveness verb that answers those two off a live driver
  or runner handle will surface them; this envelope names only
  the numeric precondition so a caller planning against an
  intended map cardinality can decide the count half without a
  driver handle or a SLAM extra installed.
* Whether the loaded map's point count is a good match for the
  runner's *live* frame density.  A map with exactly ``100``
  points admits at this floor but would produce an ICP fit with
  poor fitness against a ``50_000``-point live frame; the
  match-quality dimension is exactly what the twin envelope
  :mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope`
  names (its ``fitness_min = 0.3`` floor).  The two envelopes
  together answer the whole ``_try_relocalize`` decision -- this
  one on the precondition, the twin on the result.
* What the runner's *save* threshold is.  The neon runner's
  save path (``_SlamRunner._save``) refuses on an empty
  ``chunks`` list (``if not chunks: return {"ok": False, "error":
  "no map data accumulated"}``); that is a distinct
  refusal on a distinct surface (write-time save, not read-time
  relocalise) and a future ``g1_slam_save_envelope`` lookup pair
  will name that ``chunks`` floor.  A map that saves is not the
  same as a map that relocalises against: the save path's floor
  is on the accumulated chunk list, not on the point count that
  a later load reproduces.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.utils import positive_count_error

#: The inclusive lower bound on the loaded map's point count that
#: the neon SLAM runner's ``_try_relocalize`` admits an ICP
#: registration through.  The neon bundle reads
#: ``len(map_pts) < 100`` and returns ``None`` on strict-less, so a
#: map of exactly ``100`` points is the boundary case the runner
#: admits and this envelope quotes verbatim.  Named as an
#: inclusive lower bound so :func:`g1_slam_map_liveness_admits`
#: refuses a caller-supplied ``99`` and admits a caller-supplied
#: ``100``, mirroring the runner's ``<`` refusal exactly.
_MAP_LIVENESS_MIN: int = 100

#: The module-local refusal text every ``g1_slam_map_liveness_admits``
#: refusal quotes when the caller-supplied point count sits below
#: the neon-runner-observed floor.  Named here rather than
#: borrowed from
#: :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` because
#: the SLAM relocalise path ships no distinct rc -- the neon
#: runner just returns ``None`` from ``_try_relocalize`` on a
#: refused map, which the caller reads as "no offset applied this
#: frame".  The motion-FSM ``7404`` entry (its nearest neighbour)
#: reads ``"Invalid FSM id - need FSM in {500, 501, 801}"`` -- a
#: remedy that points a planner at locomotion FSM transitions to
#: fix a map-cardinality argument.  Surfacing the module-local
#: text keeps the refusal payload's remedy on the same surface the
#: write belongs on; a future driver-side SLAM relocalise wrapper
#: will surface this same text rather than re-borrowing a motion
#: code.
_REFUSAL_TEXT: str = (
    "map liveness gate refused - the loaded map's point count sits "
    "below the neon-runner-observed floor for an ICP relocalise. "
    "Refs strands-labs/robots#358."
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_slam_map_liveness_envelope` so
    :func:`g1_slam_map_liveness_admits` names the same field on
    its admitted-path payload and so a widen to the descriptor
    lands in one place.  Every field is a snapshot read; no bus
    is touched.
    """
    return {
        "point_count_min": _MAP_LIVENESS_MIN,
    }


@tool
def g1_list_slam_map_liveness_envelope() -> dict[str, Any]:
    """Return the point-count floor the neon SLAM runner admits a relocalise-map through.

    Read-only.  No driver instance, no DDS, no SDK, no ``numpy`` /
    ``open3d`` / ``kiss_icp`` submodule import at load time: the
    field is a module-level constant.  Useful before a future
    driver-side wrapper for SLAM relocalise is called, so a caller
    can compare an intended map's point count against the floor
    the neon runner's ``_try_relocalize`` refuses below and can
    carry the module-local refusal text a driver-side wrapper
    would surface on a bounds violation.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        the neon-runner-observed floor (``point_count_min``); and
        a ``refusals`` list carrying a single descriptor with the
        module-local :data:`_REFUSAL_TEXT` a future write verb
        would surface on a below-floor map.  Every field is a
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


@tool
def g1_slam_map_liveness_admits(
    point_count: int = _MAP_LIVENESS_MIN,
) -> dict[str, Any]:
    """Decide whether a candidate map's point count clears the relocalise floor.

    Read-only.  Compares ``point_count`` against the floor
    :func:`g1_list_slam_map_liveness_envelope` returns and reports
    the bound the argument violates.  No driver instance, no DDS,
    no SDK, no ``numpy`` / ``open3d`` / ``kiss_icp`` submodule
    import: the decision reads only a module-level constant and
    the argument itself.

    A count above the floor is *not* the same as an admitted
    snap: the neon runner also refuses on the match-quality
    dimensions (fitness, translation, trace) which the twin
    envelope :mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope`
    names, and on the two other precondition halves
    (``_o3d is None``, ``map_pts is None``) which are live-runner
    reads this verb does not answer.  The returned payload names
    only the numeric point-count decision.

    Args:
        point_count: The candidate map's point count, i.e.
            ``len(map_pts)`` on the neon runner side.  The default
            ``100`` (the observed floor) admits, so a caller who
            does not pass an explicit argument lands on the
            admitted boundary case the runner itself admits.  The
            shared :func:`~strands_robots.utils.positive_count_error`
            domain refuses non-``int`` inputs (including ``bool``,
            which is an ``int`` subclass whose ``True`` would
            otherwise be a silent ``1``), values below ``1``, and
            any type coercion that could hide a floating-point
            argument.  A count of ``0`` (an empty map) is refused
            by the shared domain rather than by the map-liveness
            floor because the runner's own precondition reads
            ``map_pts is None or len(map_pts) < 100`` and treats
            an empty array under the ``None`` half; this envelope
            answers only the numeric decision, so a caller who
            probed ``point_count=0`` receives the shared-domain
            refusal that names the shape mistake decidably before
            the ``100`` floor is asked.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the point count sits at or above the floor; a ``refusals``
        list of refusal descriptors, each carrying the dimension
        name, the offending value, the clamp it violated, the
        comparison ("value < bound" or the shared-domain
        "shared-domain" descriptor for a shape mistake), and the
        module-local :data:`_REFUSAL_TEXT` a driver-side wrapper
        would surface if the relocalise were attempted while the
        map was below the floor; the same ``envelope`` sub-dict
        :func:`g1_list_slam_map_liveness_envelope` returns.  On
        an admitted count the ``refusals`` list is empty.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    # Shared-domain shape check first: the shared
    # positive_count_error refuses bool, non-int, and value < 1.
    # This lands before the map-liveness floor check so a shape
    # mistake reads decidably against the module-local floor.
    domain_err = positive_count_error(point_count, "point_count", "g1_slam_map_liveness_admits")
    if domain_err is not None:
        refusals.append(
            {
                "dimension": "point_count",
                "value": point_count,
                "bound_key": "point_count_min",
                "bound": _MAP_LIVENESS_MIN,
                "comparison": "shared-domain",
                "domain_error": domain_err,
                "text": _REFUSAL_TEXT,
            }
        )
    else:
        # The shared domain has admitted the shape; now grade the
        # map-liveness floor.  The runner reads len(map_pts) < 100
        # and refuses on strict-less, so exactly 100 admits.
        if point_count < _MAP_LIVENESS_MIN:
            refusals.append(
                {
                    "dimension": "point_count",
                    "value": point_count,
                    "bound_key": "point_count_min",
                    "bound": _MAP_LIVENESS_MIN,
                    "comparison": "value < bound",
                    "text": _REFUSAL_TEXT,
                }
            )

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
