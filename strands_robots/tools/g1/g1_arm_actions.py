"""Agent-facing lookup for the arm-action ids ``G1ArmActionClient`` accepts.

The Unitree G1 arm-action SDK
(:class:`unitree_sdk2py.g1.arm.g1_arm_action_client.G1ArmActionClient`)
exposes a fixed set of pre-programmed gestures selected by integer id
(``11`` two-hand kiss, ``99`` release-arm, ...); its ``ExecuteAction``
method admits only ids in that set and returns ``rc=7402`` ("Invalid
action id") on every other integer. This module surfaces the id table
to an agent so a caller can decide the refusal decidably before an
execute is attempted, rather than triggering it from the SDK at wire
time.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_arm_action`` verb wrapped
  ``arm.ExecuteAction(id)`` under a single-writer lock; that write is
  the ``rt/armsdk`` topic, which today's
  :class:`~strands_robots.drivers.g1.G1Driver` does not surface (its
  ``send_action`` writes ``rt/lowcmd`` and is a different arm-SDK
  shape - the low-level joint targets, not the pre-programmed
  gestures). A future driver method that opens ``rt/armsdk`` will
  land the execute verb; refs strands-labs/robots#2765 for the arm
  refusal-mode work that gate belongs on. This module ports the
  read-only lookup half without also introducing a second DDS writer
  path the driver does not yet own.
* An SDK re-import. The id table is captured here as a module-level
  constant snapshot of
  :data:`unitree_sdk2py.g1.arm.g1_arm_action_client.action_map` (the
  same 16 entries the SDK ships today); the constant lives here
  rather than being re-imported from the SDK so
  ``import strands_robots.tools.g1.g1_arm_actions`` pulls no
  ``unitree_sdk2py`` submodule - the import-hygiene contract every
  other file in this package carries, refs strands-labs/robots#358.
  An SDK release that widens or narrows the id set is a driver-side
  update; when the driver's arm-execute method lands, its refusal
  will name the ``rc=7402`` error the same
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` entry this
  lookup returns, so both sides quote the same text.

What this module does not decide.

* Whether the FSM currently admits an arm-SDK write. Arm-action
  execution is arm-SDK-shaped: it runs only while
  :attr:`~strands_robots.drivers.g1.G1Driver._fsm_id` is inside
  :data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS`. That
  membership is a live driver read and belongs on
  :mod:`~strands_robots.tools.g1.g1_motion_gates` /
  :mod:`~strands_robots.tools.g1.g1_state`, which already answer it.
* Whether ``rt/armsdk`` is currently held by another writer. The
  SDK's own ``rc=7400`` ("rt/armsdk topic is occupied") reports that
  at execute time; a caller planning an execute cannot decide it
  without opening the topic itself, and this module opens no
  channel.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, HANDSHAKE_FSMS

#: Snapshot of ``unitree_sdk2py.g1.arm.g1_arm_action_client.action_map``
#: as shipped by the Unitree SDK today. The map is small and stable
#: (16 pre-programmed gestures; the SDK has not renumbered an entry
#: since the G1 shipped), and reading it here avoids pulling the SDK
#: at module import - the same rule
#: :mod:`~strands_robots.tools.g1.g1_joints` and
#: :mod:`~strands_robots.tools.g1.g1_motion_gates` carry. A driver
#: method that later fronts ``arm.ExecuteAction`` will validate the
#: id against the SDK's map at wire time; this snapshot is the
#: agent-facing side of the same set.
_ARM_ACTION_MAP: dict[str, int] = {
    "release arm": 99,
    "two-hand kiss": 11,
    "left kiss": 12,
    "right kiss": 13,
    "hands up": 15,
    "clap": 17,
    "high five": 18,
    "hug": 19,
    "heart": 20,
    "right heart": 21,
    "reject": 22,
    "right hand up": 23,
    "x-ray": 24,
    "face wave": 25,
    "high wave": 26,
    "shake hand": 27,
}

#: The id ``ExecuteAction`` uses to drop the arm-action hold and let
#: the driver's ``send_action`` path resume. Called out separately
#: because the neon bundle's ``g1_release_arm`` verb (a single-purpose
#: execute wrapper) resolves to this id, and a future driver method
#: fronting the release will name the same number in its refusal
#: string.
_ARM_RELEASE_ACTION_ID: int = 99

#: The error-table entries the SDK's ``ExecuteAction`` quotes on the
#: three refusal shapes a caller may see. Named here so the verb's
#: returned envelope carries the exact refusal strings a future
#: driver-side wrapper would surface, and so a re-wording of any of
#: them lands in one place instead of drifting between the SDK-side
#: log and this lookup. ``7402`` is the id-not-in-set refusal (the
#: one this lookup is the pre-check for); ``7401`` is the "arm is
#: holding, release first" refusal that fires when a caller executes
#: a second gesture without a ``99`` in between; ``7400`` is the
#: single-writer refusal that fires when two callers execute
#: concurrently.
_INVALID_ACTION_CODE: int = 7402
_HOLDING_CODE: int = 7401
_TOPIC_BUSY_CODE: int = 7400


@tool
def g1_list_arm_actions() -> dict[str, Any]:
    """Return the arm-action ids ``G1ArmActionClient.ExecuteAction`` admits.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for ``arm.ExecuteAction`` is called, so a caller can compare an
    intended action name / id against the set the SDK's execute path
    would test membership in.

    Returns:
        A dict with ``status``, a ``count`` naming the number of
        actions, an ``action_map`` dict (name -> id) covering every
        entry the SDK's own ``action_map`` ships, a sorted
        ``action_ids`` list of the ids alone (useful for a caller
        comparing an integer input), a ``release_action_id`` field
        naming the id ``ExecuteAction`` uses to drop the arm-action
        hold, an ``arm_ready_fsm_ids`` list naming the FSM ids the
        arm-SDK gate admits on (from
        :data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS`,
        surfaced here because arm-action execution is arm-SDK-shaped
        and shares the same gate), and a ``refusals`` list carrying
        the three SDK-side refusal codes and their decoded text
        (``7402`` invalid id, ``7401`` arm is holding, ``7400``
        topic is occupied) that a future execute verb's driver
        wrapper would surface. Every field is a snapshot of an SDK
        or driver constant; no dynamic decode runs here.
    """
    return {
        "status": "success",
        "count": len(_ARM_ACTION_MAP),
        "action_map": dict(_ARM_ACTION_MAP),
        "action_ids": sorted(_ARM_ACTION_MAP.values()),
        "release_action_id": _ARM_RELEASE_ACTION_ID,
        "arm_ready_fsm_ids": sorted(HANDSHAKE_FSMS),
        "refusals": [
            {"code": _INVALID_ACTION_CODE, "text": ERR_CODES[_INVALID_ACTION_CODE]},
            {"code": _HOLDING_CODE, "text": ERR_CODES[_HOLDING_CODE]},
            {"code": _TOPIC_BUSY_CODE, "text": ERR_CODES[_TOPIC_BUSY_CODE]},
        ],
    }


@tool
def g1_arm_action_admits(action: str = "", action_id: int | None = None) -> dict[str, Any]:
    """Decide whether a given arm-action name or id is inside the SDK's admission set.

    Read-only. Reads the module's snapshot of the SDK's ``action_map``
    and returns the same membership answer the SDK's ``ExecuteAction``
    would compute at wire time. A caller with either a name (from the
    neon bundle's docstring) or a raw integer id resolves it against
    the SDK's set before a future execute verb dispatches, rather than
    triggering the SDK's ``rc=7402`` refusal at wire time.

    Exactly one of ``action`` (a name) or ``action_id`` (an int) must
    be supplied. Supplying both, or neither, carries
    ``status="error"``: the ambiguous case is a caller mistake, not a
    lookup this verb should resolve arbitrarily.

    Args:
        action: The arm-action name to test. Case-sensitive to match
            the SDK's own dict keys (the SDK does not lower-case its
            lookup, so a caller writing ``"Two-Hand Kiss"`` gets a
            key-not-found on the wire; this lookup mirrors that).
            Empty string means "no name supplied".
        action_id: The arm-action id to test. Must be an ``int``;
            ``bool`` is refused (``True`` is ``int(1)`` but a dict-key
            typo of ``True`` for an action id is a caller mistake,
            not a valid gate query).

    Returns:
        A dict with ``status`` (``"success"`` on any decidable
        answer, ``"error"`` on the both-supplied / neither-supplied
        ambiguity), a ``query`` sub-dict carrying whichever of
        ``action`` / ``action_id`` was supplied, an ``admitted``
        boolean naming whether the SDK's ``ExecuteAction`` would
        admit the query, and (when ``admitted`` is ``True``) the
        resolved ``action_name`` / ``action_id`` pair a future
        execute verb would forward to the SDK. On a not-admitted
        query the dict also carries ``refusal_code`` /
        ``refusal_text`` naming the ``rc=7402`` refusal the SDK
        would return.
    """
    supplied_name = bool(action)
    supplied_id = action_id is not None
    if supplied_name == supplied_id:
        return {
            "status": "error",
            "message": (
                "supply exactly one of action= (name) or action_id= (int); "
                f"got action={action!r}, action_id={action_id!r}. "
                "Refs strands-labs/robots#358."
            ),
        }
    if supplied_id and isinstance(action_id, bool):
        return {
            "status": "error",
            "message": (f"action_id must be int, got bool ({action_id!r}). Refs strands-labs/robots#358."),
        }
    if supplied_id and not isinstance(action_id, int):
        return {
            "status": "error",
            "message": (
                f"action_id must be int, got {type(action_id).__name__} ({action_id!r}). Refs strands-labs/robots#358."
            ),
        }

    if supplied_name:
        admitted = action in _ARM_ACTION_MAP
        resolved_id = _ARM_ACTION_MAP.get(action)
        query: dict[str, Any] = {"action": action}
    else:
        admitted = action_id in _ARM_ACTION_MAP.values()
        resolved_id = action_id if admitted else None
        query = {"action_id": action_id}

    result: dict[str, Any] = {
        "status": "success",
        "query": query,
        "admitted": admitted,
    }
    if admitted:
        # Reverse-lookup the name when the caller supplied an id.
        # ``action_map`` is small (16 entries) so a linear scan is fine.
        if supplied_name:
            resolved_name = action
        else:
            resolved_name = next(name for name, aid in _ARM_ACTION_MAP.items() if aid == resolved_id)
        result["action_name"] = resolved_name
        result["action_id"] = resolved_id
    else:
        result["refusal_code"] = _INVALID_ACTION_CODE
        result["refusal_text"] = ERR_CODES[_INVALID_ACTION_CODE]
    return result
