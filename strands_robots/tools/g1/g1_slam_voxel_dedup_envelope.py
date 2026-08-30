"""Agent-facing lookup for the map-dedup voxel edge the neon SLAM runner reads.

The neon bundle's SLAM runner
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._process_frame``)
accumulates world-frame LiDAR returns into an in-memory
``_map_chunks`` list every frame the runner's ``_accumulating`` flag
is true.  Once the chunk list crosses ``100`` entries the runner
stitches every chunk into a single ``(N, 4)`` XYZI array, calls the
module-level ``_voxel_dedup(pts)`` helper on it, and installs the
deduped result back as the sole chunk on the map.  ``_voxel_dedup``
reads a single module-level constant --
``_VOXEL_DEDUP_SIZE = 0.05`` -- and dedups by
``floor(pts[:, :3] * (1.0 / _VOXEL_DEDUP_SIZE)).astype(int32)``,
collapsing every point that lands in the same 5 cm grid cell into
one.  The value is authored inline in the neon runner rather than
routed as a caller argument, so every neon SLAM host on the same
runner build dedups at 5 cm.

This module surfaces the observed voxel edge as an agent-facing
lookup pair so a caller planning a SLAM-runner deployment can
compare an intended voxel edge (a coarser dedup for a wide-area
outdoor map, a finer dedup for a millimetre-scale indoor scan)
against the neon-observed value and can carry the module-local
refusal text a driver-side wrapper would surface on a shape
violation.  A shape-graded verb is the useful shape here rather
than a numeric clamp: the neon runner has one authored value on
its build, so the envelope answers ``does this proposed voxel
edge share the shape of a usable neon-style dedup argument`` (a
positive-finite float in the float64 range), not ``does this
voxel edge sit inside a caller-visible range``.  The neon runner
would need a source patch to admit a different value, so this
envelope's ``admits`` verb answers the caller's SHAPE decision
and names the neon-observed value verbatim on the ``envelope``
sub-dict for a caller wanting to reproduce it or compare against
it.

Twin of :mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope`
(the merged strands-labs/robots#3006) and
:mod:`~strands_robots.tools.g1.g1_slam_map_liveness_envelope` (the
merged strands-labs/robots#3011).  The three modules stay
separate because they name three disjoint decisions on the same
neon SLAM runner surface:

* The map-liveness envelope answers a *precondition* on the
  candidate map's point count before a relocalise runs
  (``len(map_pts) >= 100``, a shape refusal on the caller's map
  argument).
* The relocalise envelope answers a *result-side* clamp on the
  live open3d ICP registration (fitness / translation / trace
  bounds, a shape refusal on the ICP output).
* This voxel-dedup envelope answers a *runner-build* observation
  on the dedup grid the ``_map_chunks`` compaction pass reads
  (a snapshot of the runner's authored constant, plus a shape
  refusal on a caller-supplied proposal).

Colocating the three would hand an agent planner a single refusal
payload that mixed the map-cardinality, match-quality, and
dedup-cell-size remedies -- three distinct remedies for three
distinct write paths.  Colocating them would also tie a future
runner-side voxel-edge revision (a runner patch that lowered the
5 cm grid to 1 cm for a millimetre-scale indoor mode) to a
map-liveness or match-quality revision the neon runner does not
couple.

Two things this module is deliberately *not*:

* An execution path.  The neon runner's ``_voxel_dedup`` runs on
  the ``_SlamRunner._process_frame`` thread against an in-memory
  ``_map_chunks`` list; today's
  :class:`~strands_robots.drivers.g1.G1Driver` does not front the
  SLAM runner and so does not open a second dedup path.  A future
  driver method that fronts SLAM accumulation will surface the
  same module-local :data:`_REFUSAL_TEXT` on a shape refusal;
  this module ports the read-only envelope half without also
  introducing a second SLAM path the driver does not yet own,
  refs strands-labs/robots#358.
* An SDK re-import.  The envelope is captured here as a
  module-level constant so
  ``import strands_robots.tools.g1.g1_slam_voxel_dedup_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no ``numpy``,
  ``open3d``, or ``kiss_icp`` submodule at import -- the
  import-hygiene contract every other file in this package
  carries, refs strands-labs/robots#358.  A caller authoring a
  dedup plan before any SLAM extra is installed on their host
  still gets the envelope back verbatim.

Why this module does not quote a driver-side ``rc``.

The G1 driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``);
its FSM rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``).  The
voxel-dedup pass runs on the neon runner's own thread against an
in-memory ``_map_chunks`` list; it never touches ``rt/lowcmd``,
never talks to the locomotion controller, and reaches no SDK RPC
service that ships an rc table for a dedup-argument refusal.
Borrowing ``7404`` on a bad voxel-edge argument would hand an
agent planner a motion-FSM remedy for a dedup-cell-size argument
that has nothing to do with the locomotion FSM.  The refusal shape
this module returns names the shape violation in module-local
text so a planner reads a remedy that matches the surface, and a
future driver-side SLAM accumulation wrapper will surface the
same module-local text -- not a re-borrowed motion code.  This
mirrors the same-surface refusal rule
:mod:`~strands_robots.tools.g1.g1_slam_map_liveness_envelope`
names for the map-liveness precondition, refs
strands-labs/robots#358.

What this module does not decide.

* Whether the neon runner's *kiss-icp mapping voxel*
  (``KISSConfig.mapping.voxel_size = 0.3``) admits a
  caller-proposed value.  That is a distinct authored constant
  on a distinct pass (the kiss-icp local-map registration voxel,
  read *inside* the ICP; this envelope answers the *world-frame
  dedup voxel*, read *after* the ICP result is transformed into
  the map frame).  A future
  ``g1_slam_kiss_icp_mapping_voxel_envelope`` lookup will surface
  that authored value on its own module because the two voxels
  answer two different questions on two different passes.
* Whether a caller-proposed voxel edge is a *good* dedup for a
  given LiDAR range and density.  The Livox Mid-360 the neon
  bundle wires against fires roughly ``50_000`` returns per
  frame within the ``[1.0, 40.0]`` m band (the pending
  ``g1_slam_cloud_range_envelope`` lookup pair names the band on
  its own module); a 5 cm dedup cell against that density leaves
  a map with roughly one point per cell and a working-area map
  size on the order of ``100_000`` points.  A dedup cell finer
  than the sensor's own range resolution (~2 cm on the Mid-360)
  would dedup no points and would waste the compaction pass; a
  dedup cell coarser than the kiss-icp mapping voxel would
  over-dedup and would throw away the structural detail the
  ICP already resolved.  Both are *quality* decisions that
  depend on the caller's sensor and their map-scale goal; this
  envelope answers only the *shape* decision (positive-finite
  float in the float64 range).  A future
  ``g1_slam_voxel_dedup_quality_hint`` verb could surface the
  sensor-vs-mapping-vs-dedup rules of thumb, but that verb belongs
  on a separate module because its refusal shape ("build a
  finer/coarser map") is disjoint from the shape refusal here
  ("that is not a positive-finite float in the float64 range").
* Whether the caller's voxel edge is *reachable* from the neon
  runner without a source patch.  The neon runner reads the
  constant inline; a caller who wants a value other than 5 cm
  today must patch ``_VOXEL_DEDUP_SIZE`` in the runner's own
  module.  A future driver-side SLAM accumulation wrapper that
  parameterised the dedup voxel would land the reachability
  verb; this envelope names only the shape decision on the
  proposal.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.utils import positive_finite_number_error

#: The neon-runner-observed dedup voxel edge, in metres.  The neon
#: bundle authors this as ``_VOXEL_DEDUP_SIZE = 0.05`` at the top
#: of ``cagataycali/neon-the-g1/tools/g1_slam.py`` and reads it
#: verbatim on every ``_voxel_dedup`` call the ``_map_chunks``
#: compaction pass makes.  Named here as a module-level constant
#: so the envelope descriptor names the neon-observed value
#: without a runtime read against the neon module (which would
#: pull ``numpy`` and the SLAM extra on import, breaking the
#: import-hygiene contract).  Quoted verbatim so a widen to the
#: neon runner (a lowered indoor grid, a coarsened outdoor grid)
#: lands in one place on the port side.
_VOXEL_DEDUP_NEON_DEFAULT_M: float = 0.05

#: The module-local refusal text every
#: ``g1_slam_voxel_dedup_admits`` refusal quotes when the
#: caller-supplied voxel edge fails the shape decision.  Named
#: here rather than borrowed from
#: :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` because
#: the SLAM voxel-dedup pass ships no distinct rc -- the neon
#: runner reads its authored constant and never round-trips
#: through a bus that returns one.  The motion-FSM ``7404`` entry
#: (its nearest neighbour) reads
#: ``"Invalid FSM id - need FSM in {500, 501, 801}"`` -- a remedy
#: that points a planner at locomotion FSM transitions to fix a
#: dedup-cell-size argument.  Surfacing the module-local text keeps
#: the refusal payload's remedy on the same surface the write
#: belongs on; a future driver-side SLAM accumulation wrapper will
#: surface this same text rather than re-borrowing a motion code.
_REFUSAL_TEXT: str = (
    "voxel dedup gate refused - the proposed dedup voxel edge is not "
    "a positive-finite float the neon SLAM runner would read as a "
    "dedup cell length. Refs strands-labs/robots#358."
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_slam_voxel_dedup_envelope` so
    :func:`g1_slam_voxel_dedup_admits` names the same field on
    its admitted-path payload and so a widen to the descriptor
    lands in one place.  Every field is a snapshot read; no bus
    is touched.
    """
    return {
        "voxel_dedup_neon_default_m": _VOXEL_DEDUP_NEON_DEFAULT_M,
    }


@tool
def g1_list_slam_voxel_dedup_envelope() -> dict[str, Any]:
    """Return the dedup voxel edge the neon SLAM runner reads.

    Read-only.  No driver instance, no DDS, no SDK, no ``numpy`` /
    ``open3d`` / ``kiss_icp`` submodule import at load time: the
    field is a module-level constant.  Useful before a future
    driver-side wrapper for SLAM accumulation is called, so a
    caller can compare an intended dedup voxel edge against the
    value the neon runner's ``_voxel_dedup`` reads and can carry
    the module-local refusal text a driver-side wrapper would
    surface on a shape violation.

    The envelope carries one field, the neon-observed value in
    metres, because the neon runner authors one dedup voxel edge
    at build time.  A caller wanting a different edge today must
    patch the neon runner's own module; a future driver-side
    accumulation wrapper that parameterised the edge would land
    on this envelope's :func:`g1_slam_voxel_dedup_admits` shape
    grader.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        the neon-runner-observed value
        (``voxel_dedup_neon_default_m``); and a ``refusals`` list
        carrying a single descriptor with the module-local
        :data:`_REFUSAL_TEXT` a future write verb would surface
        on a shape violation.  Every field is a snapshot of an
        observed value or a module-local text; no dynamic decode
        runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"text": _REFUSAL_TEXT},
        ],
    }


@tool
def g1_slam_voxel_dedup_admits(
    voxel_edge_m: float = _VOXEL_DEDUP_NEON_DEFAULT_M,
) -> dict[str, Any]:
    """Decide whether a candidate dedup voxel edge shares the neon-usable shape.

    Read-only.  Grades ``voxel_edge_m`` against the shared
    positive-finite-number domain
    :func:`~strands_robots.utils.positive_finite_number_error`
    names and reports the shape refusal the shared domain
    surfaces.  No driver instance, no DDS, no SDK, no ``numpy`` /
    ``open3d`` / ``kiss_icp`` submodule import: the decision reads
    only the argument itself and the module-level default.

    A value that clears the shared domain is *not* the same as an
    admitted dedup on the neon runner: the neon runner authors
    its own value inline (``_VOXEL_DEDUP_SIZE = 0.05``) and would
    need a source patch to read a different one.  This verb
    answers only the *shape* decision -- is the caller's proposal
    a positive-finite float in the float64 range -- and names the
    neon-observed value verbatim on the ``envelope`` sub-dict for
    a caller wanting to reproduce it or compare against it.  A
    future driver-side accumulation wrapper that parameterised the
    edge would surface the same shape refusal on the same
    module-local text.

    Args:
        voxel_edge_m: The candidate dedup voxel edge in metres,
            i.e. the argument a future driver-side accumulation
            wrapper would forward to ``_voxel_dedup``.  The
            default :data:`_VOXEL_DEDUP_NEON_DEFAULT_M` (the
            observed 5 cm value) admits, so a caller who does not
            pass an explicit argument lands on the neon-observed
            boundary case.  The shared
            :func:`~strands_robots.utils.positive_finite_number_error`
            domain refuses ``bool`` (an ``int`` subclass whose
            ``True`` would otherwise be a silent ``1.0`` metre
            dedup cell -- coarse enough to collapse a working-area
            map to a single point), non-``numbers.Real`` types,
            ``nan``, ``inf``, ``-inf``, values ``<= 0``, and any
            value past the float64 range.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the argument sits inside the shared positive-finite
        domain; a ``refusals`` list of refusal descriptors, each
        carrying the dimension name, the offending value, the
        clamp it violated (the shared domain), the comparison
        (``"shared-domain"``), the shared-domain error text, and
        the module-local :data:`_REFUSAL_TEXT` a driver-side
        wrapper would surface if the dedup were attempted with a
        shape-invalid argument; the same ``envelope`` sub-dict
        :func:`g1_list_slam_voxel_dedup_envelope` returns.  On an
        admitted argument the ``refusals`` list is empty.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    # Shared-domain shape check: the shared
    # positive_finite_number_error refuses bool, non-real,
    # non-finite, value <= 0, and values past the float64 range.
    # The neon runner has no separate numeric clamp on the dedup
    # voxel (it reads its authored constant inline), so the
    # shared domain is the sole shape grade this verb reports.
    domain_err = positive_finite_number_error(voxel_edge_m, "voxel_edge_m", "g1_slam_voxel_dedup_admits")
    if domain_err is not None:
        refusals.append(
            {
                "dimension": "voxel_edge_m",
                "value": voxel_edge_m,
                "bound_key": "voxel_dedup_neon_default_m",
                "bound": _VOXEL_DEDUP_NEON_DEFAULT_M,
                "comparison": "shared-domain",
                "domain_error": domain_err,
                "text": _REFUSAL_TEXT,
            }
        )

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
