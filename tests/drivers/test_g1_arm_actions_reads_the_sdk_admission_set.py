"""The arm-action lookup tools name exactly what ``ExecuteAction`` admits.

The Unitree G1 arm-action SDK
(:class:`unitree_sdk2py.g1.arm.g1_arm_action_client.G1ArmActionClient`)
admits pre-programmed gestures by integer id from a fixed table; the
:mod:`strands_robots.tools.g1.g1_arm_actions` module snapshots that table
into a module-level constant and exposes two agent-facing verbs -
:func:`g1_list_arm_actions` (list the whole set) and
:func:`g1_arm_action_admits` (decide one query) - so a caller can decide
the SDK's ``rc=7402`` refusal decidably before a future execute path is
attempted. The tests here fix that contract without pulling the SDK: the
module is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot rather
than restated in the tests, so a widen or narrow to the constant surfaces
here as a shape change rather than as a diverging table this file would
need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The verbs answer against the
  module-level snapshot, not against a live import of the SDK's
  ``action_map`` (the whole point of the port is that the snapshot lets
  a headless host answer). A driver-side wrapper for
  ``ExecuteAction`` that lands later will re-validate against the SDK's
  live map at wire time; testing the snapshot vs the live map is a
  driver-side test, not a lookup-side one.
* Which FSM ids the arm-SDK gate admits on. The verb surfaces
  :data:`HANDSHAKE_FSMS` verbatim because arm-action execution is
  arm-SDK-shaped and shares the gate; the membership rule for that
  gate is already pinned in
  :mod:`tests.drivers.test_g1_motion_gates_reads_the_driver_contract`.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES, HANDSHAKE_FSMS
from strands_robots.tools.g1.g1_arm_actions import (
    _ARM_ACTION_MAP,
    _ARM_RELEASE_ACTION_ID,
    _HOLDING_CODE,
    _INVALID_ACTION_CODE,
    _TOPIC_BUSY_CODE,
    g1_arm_action_admits,
    g1_list_arm_actions,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on that:
    the wrapper's contract is that it returns the wrapped function's
    return value verbatim. This helper is where a shape drift would
    surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an office
    bring-up. The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the arm-action lookup
    verbs to it too (refs strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_arm_actions")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_arm_actions imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_carries_the_release_id() -> None:
    """The ``release arm`` gesture is id ``99`` and the module names it.

    The SDK's own ``ExecuteAction(99)`` drops the arm-action hold; the
    neon bundle's ``g1_release_arm`` verb resolves to the same id. A
    future driver-side release wrapper will name ``99`` in its
    docstring; this cell pins that the lookup and the release id agree,
    so a rename on either side does not silently break the pair.
    """
    assert _ARM_RELEASE_ACTION_ID == 99
    assert _ARM_ACTION_MAP["release arm"] == _ARM_RELEASE_ACTION_ID


def test_the_snapshot_covers_the_sdk_shipped_set() -> None:
    """The snapshot names every action the SDK's ``action_map`` ships.

    The SDK's own dict has 16 entries (the pre-programmed gestures
    plus release-arm). A drift on either side surfaces here: the
    driver-side arm-execute wrapper (when it lands) will validate the
    same set at wire time and its refusal string will quote
    ``rc=7402`` on any id outside it. The count is pinned rather than
    listed name-by-name so a caller widening the map on the driver
    side updates one number here rather than 16 assertions.
    """
    assert len(_ARM_ACTION_MAP) == 16


def test_g1_list_arm_actions_returns_the_whole_table() -> None:
    """The verb's payload names the map, the ids, and the SDK refusals.

    ``count`` is the size of the module's own snapshot, ``action_map``
    is the snapshot verbatim (a fresh dict, so a caller mutating it
    cannot poison the module's constant), ``action_ids`` is a sorted
    list of the values, and ``refusals`` names the three SDK-side
    refusal codes (``7402`` id-not-in-set, ``7401`` holding, ``7400``
    topic-busy) with the decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_arm_actions)
    assert result["status"] == "success"
    assert result["count"] == len(_ARM_ACTION_MAP)
    assert result["action_map"] == _ARM_ACTION_MAP
    # The verb returns a fresh dict, not the module constant; a mutation
    # on the returned value cannot leak back into the snapshot.
    result["action_map"]["synthetic"] = 999
    assert "synthetic" not in _ARM_ACTION_MAP
    assert result["action_ids"] == sorted(_ARM_ACTION_MAP.values())
    assert result["release_action_id"] == _ARM_RELEASE_ACTION_ID
    assert result["arm_ready_fsm_ids"] == sorted(HANDSHAKE_FSMS)
    codes = {r["code"] for r in result["refusals"]}
    assert codes == {_INVALID_ACTION_CODE, _HOLDING_CODE, _TOPIC_BUSY_CODE}
    for refusal in result["refusals"]:
        assert refusal["text"] == ERR_CODES[refusal["code"]]


def test_g1_arm_action_admits_resolves_a_valid_name() -> None:
    """A name in the snapshot is admitted and the id is returned.

    ``"two-hand kiss"`` is id ``11``; the verb reports ``admitted=True``
    and carries the resolved ``action_name`` / ``action_id`` pair a
    future execute verb would forward to the SDK. No refusal fields
    fire on the admitted path.
    """
    result = _call(g1_arm_action_admits, action="two-hand kiss")
    assert result["status"] == "success"
    assert result["admitted"] is True
    assert result["query"] == {"action": "two-hand kiss"}
    assert result["action_name"] == "two-hand kiss"
    assert result["action_id"] == 11
    assert "refusal_code" not in result


def test_g1_arm_action_admits_resolves_a_valid_id() -> None:
    """An id in the snapshot is admitted and the name is reverse-looked-up.

    ``99`` is ``release arm``; the verb resolves both directions so a
    caller with only an integer receives the name a future execute
    verb's log line would carry.
    """
    result = _call(g1_arm_action_admits, action_id=99)
    assert result["status"] == "success"
    assert result["admitted"] is True
    assert result["query"] == {"action_id": 99}
    assert result["action_name"] == "release arm"
    assert result["action_id"] == 99


def test_g1_arm_action_admits_refuses_an_unknown_name() -> None:
    """A name outside the snapshot carries the SDK's ``rc=7402`` refusal.

    ``"Two-Hand Kiss"`` (title-case) is not in the SDK's own dict;
    the verb reports ``admitted=False`` and quotes the same refusal
    the SDK's ``ExecuteAction`` would return at wire time.
    """
    result = _call(g1_arm_action_admits, action="Two-Hand Kiss")
    assert result["status"] == "success"
    assert result["admitted"] is False
    assert result["query"] == {"action": "Two-Hand Kiss"}
    assert result["refusal_code"] == _INVALID_ACTION_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_ACTION_CODE]
    assert "action_name" not in result


def test_g1_arm_action_admits_refuses_an_unknown_id() -> None:
    """An id outside the snapshot carries the SDK's ``rc=7402`` refusal.

    ``42`` is not in the SDK's shipped set; the verb reports
    ``admitted=False`` and quotes the driver-side error entry the
    SDK's ``ExecuteAction`` would surface as ``rc=7402``.
    """
    result = _call(g1_arm_action_admits, action_id=42)
    assert result["status"] == "success"
    assert result["admitted"] is False
    assert result["query"] == {"action_id": 42}
    assert result["refusal_code"] == _INVALID_ACTION_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_ACTION_CODE]


def test_g1_arm_action_admits_refuses_neither_arg() -> None:
    """Supplying neither name nor id is a caller mistake, not a lookup.

    The verb cannot decide membership against nothing; the returned
    envelope names the two-arg contract and cites the tracking issue
    so a caller reading the refusal sees the resolvable reference.
    """
    result = _call(g1_arm_action_admits)
    assert result["status"] == "error"
    assert "supply exactly one" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_g1_arm_action_admits_refuses_both_args() -> None:
    """Supplying both name and id is also a caller mistake.

    A caller who supplied both might intend either; the verb refuses
    to pick and quotes both values verbatim in the error so the caller
    sees what they sent.
    """
    result = _call(g1_arm_action_admits, action="release arm", action_id=99)
    assert result["status"] == "error"
    assert "supply exactly one" in result["message"]
    assert "release arm" in result["message"]
    assert "99" in result["message"]


def test_g1_arm_action_admits_refuses_bool_id() -> None:
    """``bool`` action_id is refused: ``True`` is a typo, not id ``1``.

    Python's ``bool`` is a subclass of ``int``; passing ``True`` would
    otherwise silently resolve to ``1`` (which is not in the SDK's
    action set, so the outer refusal would surface anyway) - but the
    caller almost certainly meant to pass an int, not a truth value.
    The verb names the type mismatch decidably rather than answering
    a query that a caller writing ``action_id=True`` cannot have
    intended.
    """
    result = _call(g1_arm_action_admits, action_id=True)
    assert result["status"] == "error"
    assert "bool" in result["message"]


def test_g1_arm_action_admits_refuses_non_int_id() -> None:
    """A non-int, non-None ``action_id`` is refused with the type name.

    A caller writing ``action_id="99"`` gets a decidable refusal
    (naming ``str``) rather than a false-negative admission against a
    dict of int values.
    """
    result = _call(g1_arm_action_admits, action_id="99")  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "str" in result["message"]
