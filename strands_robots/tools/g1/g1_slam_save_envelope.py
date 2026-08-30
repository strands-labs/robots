"""Agent-facing lookup for the map-save gate the neon SLAM runner admits a write through.

The neon bundle's SLAM runner
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner.save_map``)
takes a candidate map name and DECIDES two things at once before it lets
the accumulated point cloud reach ``numpy.savez``: the name has to
resolve to a path under ``MAPS_DIR = ~/maps`` after
:meth:`pathlib.Path.resolve`
(``_safe_map_path`` returns ``None`` when the resolved path escapes),
and the runner's accumulated chunks list has to be non-empty
(``if not chunks: return {"ok": False, "error": "no map data accumulated"}``).
Each is a distinct refusal shape: an unsafe name means the caller asked
to write outside the maps directory (a traversal, an absolute path, or a
name that ``pathlib.Path.resolve`` cannot normalise), and an empty chunks
list means the runner has not yet accumulated any map points to save --
either accumulation was never toggled on
(``g1_slam_accumulate(True)``), or the runner was reset after a load
before enough frames landed.  The neon runner refuses outside either
one of them; a caller planning a save wants each answered on its own so
the refusal string names which of the two went wrong.

This module snapshots the observed envelope into module-level
constants and exposes two agent-facing verbs so a caller can decide
the refusal decidably before a future driver-side SLAM save path is
called, rather than pinning the rule inside the write path where the
refusal is invisible to the planner.

Twin of :mod:`~strands_robots.tools.g1.g1_slam_map_liveness_envelope`
(strands-labs/robots#3011) on the same SLAM surface -- that envelope
names the *load*-side floor (the neon runner refuses ICP relocalise on
a map with fewer than 100 points), and this envelope names the *save*-side
floor on the same accumulated chunks list.  The two modules stay separate
because a load reads a persisted map's point count against an ICP
match-quality precondition, and a save reads the runner's live
accumulated chunks against a "have I accumulated anything" precondition:
two different reads with disjoint refusal shapes.  Colocating them
would hand an agent planner a single refusal payload that mixed the
two remedies -- "toggle accumulation on before saving" versus "the
loaded map is too sparse for an ICP relocalise" -- and would tie a
future save-side rule revision (a minimum accumulated frames count,
say) to a load-side ICP-parameter revision the neon bundle does not
couple.

Two things this module is deliberately *not*:

* An execution path.  The neon bundle's ``save_map`` runs a full
  ``numpy.savez`` on the deciding host's disk under
  ``_SlamRunner.save_map``; that write is the SLAM-runner-side of the
  neon bundle's pose pipeline, which today's
  :class:`~strands_robots.drivers.g1.G1Driver` does not front.  A
  future driver method that fronts SLAM save will land the transition
  verb; refs strands-labs/robots#358 for the SDK-facing gate work that
  write belongs on.  This module ports the read-only envelope half
  without also introducing a second SLAM path the driver does not yet
  own.
* An SDK re-import.  The envelope is captured here as module-level
  constants so ``import strands_robots.tools.g1.g1_slam_save_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no ``numpy``,
  ``open3d``, or ``kiss_icp`` submodule at import time -- the
  import-hygiene contract every other file in this package carries,
  refs strands-labs/robots#358.  A caller authoring a save plan
  before any SLAM extra is installed on their host still gets the
  envelope back verbatim.  A revision of the observed rule is a
  driver-side update; when the driver's SLAM save method lands, its
  refusal will surface the same module-local :data:`_REFUSAL_TEXT`
  this module names for a save-gate violation.

Why this module does not quote a driver-side ``rc``.

The G1 driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``); its FSM
rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``).  The SLAM save
decision runs on the neon runner's own thread against an in-memory
chunks list -- it never touches ``rt/lowcmd``, never talks to the
locomotion controller, and reaches no SDK RPC service that ships an rc
table for a save-gate refusal.  Borrowing ``7404`` on a
name-outside-maps-dir refusal would hand an agent planner a motion-FSM
remedy (``"need FSM in {500, 501, 801}"``) for a save that has nothing
to do with the locomotion FSM.  The refusal shape this module returns
names the module-local rule violation in module-local text so a
planner reads a remedy that matches the surface, and a future
driver-side SLAM save wrapper will surface the same module-local
text -- not a re-borrowed motion code.  This mirrors the same-surface
refusal rule
:mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope` names for
the ICP relocalise gate, refs strands-labs/robots#358.

What this module does not decide.

* Whether ``numpy.savez`` succeeds.  A save that clears this envelope
  can still fail at the filesystem (a permission refusal, a disk full
  refusal, a device disappearance); the neon runner surfaces those as
  ``{"ok": False, "error": str(exc)}`` on a distinct surface -- an
  ``OSError`` from the write, not a rule refusal.  A future runner-side
  I/O failure guard is a separate concern with its own refusal shape.
* Whether the accumulated map is *dense enough* for a later load to
  reach the ICP relocalise gate.  A one-point map clears this
  envelope's chunks floor and is a valid save, but a subsequent
  ``g1_slam_load(name, relocalize=True)`` would then trip the twin
  envelope's ``point_count_min = 100`` floor on the load side.  The
  two floors are answered on their own surfaces: this one is "did
  the runner accumulate anything", and #3011's is "does the persisted
  map carry enough points for an ICP relocalise".  A caller planning
  a save-then-relocalise round-trip combines both.
* What ``pathlib.Path.resolve`` would return for a caller's name.  The
  neon runner's ``_safe_map_path`` calls ``.resolve()`` on the joined
  path and then compares against ``MAPS_DIR.resolve()``; the resolve
  step reads the local filesystem's symlink layout, which this module
  does not touch (no filesystem I/O at load time is the hygiene
  contract above).  This module names the *shape* of a rule-violating
  name (a name that is not a string, an empty name, a name carrying a
  path separator or ``..`` component) so the shape refusals fire
  before any resolve is attempted; a future driver-side wrapper's
  live resolve is on top of this shape floor, and its "resolved
  outside MAPS_DIR" refusal is a separate live-runner state read this
  module does not answer.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: The lower clamp on the runner's accumulated chunks list length
#: (inclusive int).  The neon bundle's ``g1_slam.py::_SlamRunner.save_map``
#: reads ``if not chunks: return {"ok": False, "error": "no map data
#: accumulated"}`` against the runner's own ``self._map_chunks`` list.
#: An empty list means the runner has never observed a LiDAR frame under
#: accumulation, so a save would write a zero-point map -- ``numpy.savez``
#: would accept an empty array, but the resulting artifact is unusable
#: (``_SlamRunner.load_map`` reads it back as an empty ``self._map_chunks``,
#: which then trips the twin envelope's ICP-relocalise point-count floor
#: at load time).  Named as an inclusive lower bound (``chunks_count >= 1``
#: admits) because the neon check reads ``not chunks`` which is a strict
#: emptiness test, so a chunks list of exactly one entry is the boundary
#: the runner admits and this envelope quotes verbatim.
_CHUNKS_COUNT_MIN: int = 1

#: The module-local refusal text every ``g1_slam_save_admits`` refusal
#: quotes when the caller's argument sits outside one of the two
#: neon-runner-observed envelopes.  Named here rather than borrowed
#: from :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` because
#: the SLAM save path ships no distinct rc -- the neon runner just
#: returns ``{"ok": False, "error": ...}`` from ``save_map`` on a
#: refused write, which the caller reads as "no artifact written".  The
#: motion-FSM ``7404`` entry (its nearest neighbour) reads ``"Invalid
#: FSM id - need FSM in {500, 501, 801}"`` -- a remedy that points a
#: planner at locomotion FSM transitions to fix a save-shape argument.
#: Surfacing the module-local text keeps the refusal payload's remedy
#: on the same surface the write belongs on, and a future driver-side
#: SLAM save wrapper will surface this same text rather than
#: re-borrowing a motion code.
_REFUSAL_TEXT: str = (
    "save gate refused - one or more of chunks_count/name sits "
    "outside the neon-runner-observed envelope. Refs "
    "strands-labs/robots#358."
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_slam_save_envelope`
    so :func:`g1_slam_save_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place.  Every field is a snapshot read; no bus is touched.
    """
    return {
        "chunks_count_min": _CHUNKS_COUNT_MIN,
    }


@tool
def g1_list_slam_save_envelope() -> dict[str, Any]:
    """Return the save-gate envelope the neon SLAM runner admits a write through.

    Read-only.  No driver instance, no DDS, no SDK, no ``numpy`` /
    ``open3d`` / ``kiss_icp`` submodule import at load time: every
    field is a module-level constant.  Useful before a future
    driver-side wrapper for SLAM save is called, so a caller can
    compare an intended (chunks_count, name) pair against the
    envelope the neon runner's ``save_map`` refuses outside of and
    can carry the module-local refusal text a driver-side wrapper
    would surface on any bound violation.

    Two dimensions ride the same envelope because the neon runner
    reads them together and returns ``{"ok": False, ...}`` on any
    one being outside its range (chunks list empty, name resolving
    outside ``MAPS_DIR`` or non-string).  Colocating the two in one
    envelope lets a caller planning a save ask one verb and receive
    one payload that names both refusal shapes, mirroring how the
    neon runner itself reads them.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        the ``chunks_count_min`` floor the neon runner observed;
        and a ``refusals`` list carrying a single descriptor with
        the module-local :data:`_REFUSAL_TEXT` a future write verb
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


def _reject(
    refusals: list[dict[str, Any]],
    dimension: str,
    value: Any,
    bound_key: str,
    bound: Any,
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
def g1_slam_save_admits(
    chunks_count: int = 1,
    name: str = "map",
) -> dict[str, Any]:
    """Decide whether a (chunks_count, name) pair sits inside the save gate.

    Read-only.  Compares the two arguments against the module-local
    rule set :func:`g1_list_slam_save_envelope` returns and reports
    every rule the pair violates.  No driver instance, no DDS, no
    SDK, no ``numpy`` / ``open3d`` / ``kiss_icp`` submodule import,
    and no filesystem I/O: the decision reads only module-level
    constants and the arguments themselves.

    A pair inside the envelope is *not* the same as an admitted
    save: the neon runner also refuses on a filesystem I/O failure
    (``OSError`` from ``numpy.savez``) and on a name whose resolved
    path escapes ``MAPS_DIR`` after ``pathlib.Path.resolve`` reads
    the local symlink layout -- both are live-runner state reads
    this verb does not answer (that is a future driver-side
    companion verb).  The returned payload names only the module-
    local rule decision.

    All rule violations are graded, so a pair with two violations
    reports both refusals; a caller reading only the first would
    re-plan against the chunks floor and still trip on the name
    shape.  The neon runner's own check short-circuits on the first
    refusal, but a caller planning a save wants every violation
    named so their next attempt fixes the whole pair.

    Args:
        chunks_count: number of point-cloud chunks the runner has
            accumulated.  The default ``1`` (the smallest usable
            accumulation) admits, so a caller who does not pass an
            explicit argument for this dimension lands on the
            admitted side of the chunks floor -- letting them
            probe the name dimension in isolation.  Refused below
            ``chunks_count_min`` (the neon-runner-observed floor;
            below it the runner has not yet accumulated any map
            points to save).  Boolean values are refused explicitly
            because Python's ``bool`` is a subclass of ``int``, so
            a caller passing ``True`` would otherwise silently look
            up ``1`` (a legitimate minimum) and hide the type
            mistake.  Non-int values (float, string, None) are
            refused with a ``non-int`` comparison because the neon
            runner's own ``if not chunks`` compares the list, not a
            fractional or string count, and this envelope names the
            *count* the pair would resolve to.
        name: the filename stem the caller intends to pass to
            ``g1_slam_save(name)``.  The default ``"map"`` (a plain
            alphanumeric stem) admits, so a caller who does not
            pass an explicit argument for this dimension lands on
            the admitted side of the name rule.  Refused when the
            name is not a string, is empty, or carries a path-
            separator character (``/`` or ``\\``) or a ``..``
            component that ``pathlib.Path.resolve`` would use to
            climb out of ``MAPS_DIR``.  Boolean values are refused
            explicitly for the same reason as ``chunks_count``: a
            ``bool`` is a subclass of ``int``, not a ``str``, so
            passing ``True`` here reads as a type mistake.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        every rule is respected; a ``refusals`` list of refusal
        descriptors, each carrying the dimension name, the
        offending value, the rule it violated, the comparison
        ("value < bound", "empty", "non-int", "non-str", or
        "path-shape"), and the module-local :data:`_REFUSAL_TEXT`
        a driver-side wrapper would surface if the save were
        attempted while any rule is violated; the same ``envelope``
        sub-dict :func:`g1_list_slam_save_envelope` returns.  On an
        admitted pair the ``refusals`` list is empty; on a
        rejected pair every violated rule is named so the caller
        sees the whole shape of the refusal at once.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    # chunks_count dimension: bool first (subclass of int), then
    # non-int, then floor.
    if isinstance(chunks_count, bool):
        _reject(
            refusals,
            "chunks_count",
            chunks_count,
            "chunks_count_min",
            _CHUNKS_COUNT_MIN,
            "non-int",
        )
    elif not isinstance(chunks_count, int):
        _reject(
            refusals,
            "chunks_count",
            chunks_count,
            "chunks_count_min",
            _CHUNKS_COUNT_MIN,
            "non-int",
        )
    elif chunks_count < _CHUNKS_COUNT_MIN:
        _reject(
            refusals,
            "chunks_count",
            chunks_count,
            "chunks_count_min",
            _CHUNKS_COUNT_MIN,
            "value < bound",
        )

    # name dimension: bool first (subclass of int, so isinstance(x,
    # str) already excludes it, but naming the refusal explicitly
    # gives it its own comparison string rather than the generic
    # non-str one), then non-str, then empty, then path-shape.
    if isinstance(name, bool):
        _reject(refusals, "name", name, "name_shape", "str", "non-int")
    elif not isinstance(name, str):
        _reject(refusals, "name", name, "name_shape", "str", "non-str")
    elif name == "":
        _reject(refusals, "name", name, "name_shape", "non-empty", "empty")
    elif "/" in name or "\\" in name or ".." in name.split("/") or name.startswith("."):
        # The neon runner's ``_safe_map_path`` refuses on a resolved
        # path that escapes MAPS_DIR; the shape refusals below fire
        # on the caller's raw name before any resolve, so the pair
        # answered here is the *shape* of a rule-violating name
        # rather than the live resolve result.  ``/`` and ``\`` are
        # path separators (a name carrying one is a compound path,
        # which is not a valid map stem), ``..`` is the traversal
        # component ``resolve`` uses to climb out of MAPS_DIR (a
        # bare ``..`` in the split parts is enough to trip it),
        # and a leading ``.`` (``.hidden`` or ``./x``) is either a
        # dotfile (out-of-scope for this bundle) or a self-
        # reference the resolve would collapse.
        _reject(refusals, "name", name, "name_shape", "safe-stem", "path-shape")

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
