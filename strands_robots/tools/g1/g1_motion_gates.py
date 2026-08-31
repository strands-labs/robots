"""Agent-facing lookup for the FSM ids ``G1Driver`` admits writes on.

``G1Driver._check_motion_gates`` refuses every arm-SDK-shaped write while
:attr:`~strands_robots.drivers.g1.G1Driver._fsm_id` is outside
:data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS`, and every
locomotion-shaped write while it is outside
:data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. This module surfaces
those two sets to an agent so a caller can decide the refusal decidably
before ``send_action`` / ``run_policy`` is attempted, rather than triggering
it from the driver at wire time.

The verb here reads the driver's own gate constants -
:data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS` and
:data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` - so a gate widened
or narrowed in the driver's admission table moves both the write path and
this lookup together. No SDK, no DDS, no cache: this module is a pure reader
over module-level tables, so ``import strands_robots.tools.g1.g1_motion_gates``
pulls no ``unitree_sdk2py`` submodule (the same import-hygiene contract every
other file in this package carries, refs strands-labs/robots#358).

What this module does not decide.

* Whether the driver's ``_fsm_id`` currently sits inside either set. That is
  a live read on a running driver and belongs on the driver's
  ``get_status`` envelope (which already carries ``fsm_id``); a companion
  verb that wraps that async call is a separate port from this one, so a
  reference-only lookup can land without also introducing the first
  driver-instance-taking verb in this package.
* Which FSM id names the caller may see. The driver's constants are a set
  of integers, not a name table; the SDK does not ship a canonical id -> name
  mapping and neither does the driver, so this verb returns the integers
  the driver actually gates on rather than a translated label a reader could
  drift from. The write-path refusal quotes those same integers verbatim
  (see the driver's ``_check_motion_gates`` and the ``7404`` entry in
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`), so a caller
  comparing this verb's output to the refusal string sees the same numbers
  on both sides.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import (
    ERR_CODES,
    HANDSHAKE_FSMS,
    WALK_FSMS,
)

#: The two scopes ``G1Driver._check_motion_gates`` accepts. The driver's
#: gate takes a scope name and picks the matching set; this map is the
#: agent-facing side of the same choice. Kept here rather than in the
#: driver because the driver has no use for the mapping (its gate picks
#: one set at admission time, not both), and the mapping only matters
#: when a caller wants to enumerate the driver's contract before writing.
_SCOPE_SETS: dict[str, frozenset[int]] = {
    "arm": HANDSHAKE_FSMS,
    "loco": WALK_FSMS,
}

#: The error-table entry the driver's write path quotes when it refuses on
#: an FSM outside its admitted set. Named here so the returned envelope can
#: carry the exact refusal string the driver would surface, and so a future
#: rewording of that message lands in one place instead of drifting between
#: the driver's log and this lookup.
_FSM_REFUSAL_CODE: int = 7404


@tool
def g1_list_motion_gates(scope: str = "") -> dict[str, Any]:
    """Return the FSM-id sets ``G1Driver._check_motion_gates`` admits on.

    Read-only. Every field is a driver constant; no bus is touched, no
    driver instance is required. Useful before ``send_action`` or
    ``run_policy`` is attempted, so a caller can compare the driver's live
    ``fsm_id`` (from ``G1Driver.get_status``) against the set the gate
    would test membership in.

    Args:
        scope: Optional scope filter. ``"arm"`` returns the FSM ids that
            admit arm-SDK-shaped writes (:data:`HANDSHAKE_FSMS`);
            ``"loco"`` returns the narrower set locomotion-shaped writes
            need (:data:`WALK_FSMS`, which excludes ``500`` because
            sitting accepts arm gestures but not walking). Empty returns
            both scopes so a caller can see the whole gate at once.

    Returns:
        A dict with ``status`` and a ``gates`` list of records, one per
        returned scope. Each record carries ``scope``, an ``fsm_ids`` list
        (sorted ascending, integers only), and the ``refusal_code`` /
        ``refusal_text`` pair the driver's write path would surface when a
        caller writes with the live FSM outside the set. On an unknown
        scope name the returned dict carries ``status="error"`` and a
        ``message`` naming the valid scopes as a resolvable domain.
    """
    if scope and scope not in _SCOPE_SETS:
        valid = sorted(_SCOPE_SETS)
        return {
            "status": "error",
            "message": (f"unknown scope {scope!r}. Valid scopes are {valid}. Refs strands-labs/robots#358."),
        }
    scopes = (scope,) if scope else tuple(sorted(_SCOPE_SETS))
    gates = [
        {
            "scope": name,
            "fsm_ids": sorted(_SCOPE_SETS[name]),
            "refusal_code": _FSM_REFUSAL_CODE,
            "refusal_text": ERR_CODES[_FSM_REFUSAL_CODE],
        }
        for name in scopes
    ]
    return {
        "status": "success",
        "count": len(gates),
        "scope": scope or None,
        "scopes": sorted(_SCOPE_SETS),
        "gates": gates,
    }


@tool
def g1_fsm_admits(fsm_id: int, scope: str = "arm") -> dict[str, Any]:
    """Decide whether a given FSM id is inside ``G1Driver``'s admission set.

    Read-only. Reads the driver's constants (:data:`HANDSHAKE_FSMS` for
    ``arm``, :data:`WALK_FSMS` for ``loco``) and returns the same
    membership answer the driver's write path would compute. A caller with
    a live ``fsm_id`` from ``G1Driver.get_status`` uses this to phrase a
    refusal in its own voice, rather than triggering the driver's refusal
    at wire time.

    Args:
        fsm_id: The FSM id to test. Must be an int; ``bool`` is refused
            (``True`` is ``int(1)`` but a dict-key typo of ``True`` for
            an FSM id is a caller mistake, not a valid gate query).
        scope: Which admission set to test. ``"arm"`` (default) tests
            :data:`HANDSHAKE_FSMS`; ``"loco"`` tests :data:`WALK_FSMS`.

    Returns:
        A dict with ``status``, the requested ``scope``, the tested
        ``fsm_id``, an ``admitted`` boolean naming whether the write path
        would open, the ``fsm_ids`` list the answer was computed against,
        and (when ``admitted`` is ``False``) the ``refusal_code`` and
        ``refusal_text`` the driver would surface. An unknown scope or a
        non-int ``fsm_id`` carries ``status="error"``.
    """
    if scope not in _SCOPE_SETS:
        valid = sorted(_SCOPE_SETS)
        return {
            "status": "error",
            "message": (f"unknown scope {scope!r}. Valid scopes are {valid}. Refs strands-labs/robots#358."),
        }
    if not isinstance(fsm_id, int) or isinstance(fsm_id, bool):
        return {
            "status": "error",
            "message": (
                f"fsm_id must be an int; got {type(fsm_id).__name__} {fsm_id!r}. Refs strands-labs/robots#358."
            ),
        }
    admitted = fsm_id in _SCOPE_SETS[scope]
    result: dict[str, Any] = {
        "status": "success",
        "scope": scope,
        "fsm_id": fsm_id,
        "admitted": admitted,
        "fsm_ids": sorted(_SCOPE_SETS[scope]),
    }
    if not admitted:
        result["refusal_code"] = _FSM_REFUSAL_CODE
        result["refusal_text"] = ERR_CODES[_FSM_REFUSAL_CODE]
    return result
