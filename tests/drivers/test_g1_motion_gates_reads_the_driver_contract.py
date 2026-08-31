"""The motion-gate tools name exactly what ``G1Driver._check_motion_gates`` admits on.

``_check_motion_gates`` refuses any arm-SDK-shaped write while the driver's
``_fsm_id`` is outside :data:`HANDSHAKE_FSMS`, and any locomotion-shaped
write while it is outside :data:`WALK_FSMS`. Two agent-facing tools -
:func:`g1_list_motion_gates` and :func:`g1_fsm_admits` - exist to surface
those same sets before the write is attempted, so a caller can decide the
refusal decidably rather than triggering it from the driver at wire time.

The membership rules the tools carry are read here off the driver's
constants rather than being restated in the tests, so a driver-side widen
or narrow of a gate (which is what the closed ``strands-labs/robots#2765``
landed for the arm-SDK path with the ``_check_motion_gates`` producer
under ``refs strands-labs/robots#358``) does not require also editing this
file. What the tests do restate is the *shape* of each returned record and
the SDK-load-hygiene contract every file under :mod:`strands_robots.tools.g1`
carries: importing the module must not pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import (
    ERR_CODES,
    HANDSHAKE_FSMS,
    WALK_FSMS,
)
from strands_robots.tools.g1.g1_motion_gates import (
    _FSM_REFUSAL_CODE,
    _SCOPE_SETS,
    g1_fsm_admits,
    g1_list_motion_gates,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function directly
    when called in-process, but a caller cannot rely on that: the wrapper's
    contract is that it returns the wrapped function's return value verbatim.
    This helper is where a shape drift would surface once, rather than at
    every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable with
    the SDK absent; a module that pulled a submodule at import time would
    break every headless CI runner and Thor before an office bring-up. The
    driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only path
    that loads the SDK); this cell holds the motion-gate verbs to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_motion_gates")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_motion_gates imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_scope_map_names_the_driver_admission_sets_verbatim() -> None:
    """The module's scope map is the driver's own admission sets, not a restatement.

    ``arm`` reaches :data:`HANDSHAKE_FSMS`; ``loco`` reaches :data:`WALK_FSMS`.
    A widen or narrow to either set in the driver's own module surfaces here
    as identity, not as a diverging table this test file would need to
    manually update.
    """
    assert _SCOPE_SETS["arm"] is HANDSHAKE_FSMS
    assert _SCOPE_SETS["loco"] is WALK_FSMS


def test_list_returns_every_scope_the_driver_names() -> None:
    """No argument = every scope in the driver's admission table, once each."""
    result = _call(g1_list_motion_gates)
    assert result["status"] == "success"
    assert result["count"] == len(_SCOPE_SETS)
    returned_scopes = [row["scope"] for row in result["gates"]]
    assert sorted(returned_scopes) == sorted(_SCOPE_SETS)
    # The unfiltered call carries no single-scope label on the envelope.
    assert result["scope"] is None
    assert result["scopes"] == sorted(_SCOPE_SETS)


def test_every_gate_row_carries_the_driver_admission_set() -> None:
    """Each returned scope quotes the exact FSM-id set the driver admits on."""
    result = _call(g1_list_motion_gates)
    for row in result["gates"]:
        expected = sorted(_SCOPE_SETS[row["scope"]])
        assert row["fsm_ids"] == expected, (
            f"fsm_ids for scope {row['scope']!r} drifted from the driver's "
            "admission set - the list tool must not restate FSM ids."
        )
        # Every element is a plain int on the wire, not a numpy scalar or
        # a frozenset the driver's set exposes internally.
        assert all(isinstance(fsm, int) and not isinstance(fsm, bool) for fsm in row["fsm_ids"])


def test_every_gate_row_carries_the_write_path_refusal_string() -> None:
    """Each returned scope carries the ``7404`` refusal the driver would surface.

    The driver's write path phrases its FSM-outside-admission refusal against
    the same error-table entry this verb surfaces (``ERR_CODES[7404]``); a
    caller comparing the verb's output against a live refusal log sees the
    same string on both sides.
    """
    result = _call(g1_list_motion_gates)
    for row in result["gates"]:
        assert row["refusal_code"] == _FSM_REFUSAL_CODE
        assert row["refusal_text"] == ERR_CODES[_FSM_REFUSAL_CODE]


def test_scope_filter_returns_that_scope_verbatim() -> None:
    """A scope name selects the FSM-id set the driver's admission table publishes for it."""
    for scope, fsm_ids in _SCOPE_SETS.items():
        result = _call(g1_list_motion_gates, scope=scope)
        assert result["status"] == "success"
        assert result["count"] == 1
        assert result["scope"] == scope
        [row] = result["gates"]
        assert row["scope"] == scope
        assert row["fsm_ids"] == sorted(fsm_ids)


def test_scope_filter_refuses_an_unknown_scope_by_name() -> None:
    """An unknown scope returns an error whose message names the domain.

    The refusal quotes the unknown value, the accepted scope names, and the
    resolvable issue reference the package's contract sits behind - the same
    three anchors every other refusal in this package carries.
    """
    result = _call(g1_list_motion_gates, scope="teleport")
    assert result["status"] == "error"
    assert "teleport" in result["message"]
    assert "arm" in result["message"]  # domain is named
    assert "loco" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_admits_reports_membership_the_gate_would_compute() -> None:
    """Every FSM id the driver admits on returns ``admitted=True``, others return ``False``.

    Sampled from the driver's own admission sets rather than a hard-coded
    list, so a widen of :data:`HANDSHAKE_FSMS` on the driver flips this
    test's answer without also needing to edit this file's expected values.
    """
    for scope, admitted_set in _SCOPE_SETS.items():
        for fsm_id in admitted_set:
            result = _call(g1_fsm_admits, fsm_id=fsm_id, scope=scope)
            assert result["status"] == "success"
            assert result["scope"] == scope
            assert result["fsm_id"] == fsm_id
            assert result["admitted"] is True
            assert result["fsm_ids"] == sorted(admitted_set)
            # An admitted id carries no refusal string - the gate would open.
            assert "refusal_code" not in result
            assert "refusal_text" not in result


def test_admits_reports_a_non_member_as_refused_with_the_driver_message() -> None:
    """An FSM id outside the admission set surfaces the driver's ``7404`` refusal.

    The chosen value ``42`` sits outside both :data:`HANDSHAKE_FSMS` and
    :data:`WALK_FSMS` on the shipped tree, and no plausible driver-side
    widen of either set would include it (the driver's ids are all
    three-digit motion-switcher constants). If a firmware update did add
    ``42`` to either set, this test's replacement would be a value the
    updated driver still refuses - not a delete.
    """
    outside = 42
    assert outside not in HANDSHAKE_FSMS
    assert outside not in WALK_FSMS
    for scope in _SCOPE_SETS:
        result = _call(g1_fsm_admits, fsm_id=outside, scope=scope)
        assert result["status"] == "success"
        assert result["admitted"] is False
        assert result["refusal_code"] == _FSM_REFUSAL_CODE
        assert result["refusal_text"] == ERR_CODES[_FSM_REFUSAL_CODE]


def test_admits_refuses_an_unknown_scope_by_name() -> None:
    """An unknown scope on ``g1_fsm_admits`` returns the same domain-naming error."""
    result = _call(g1_fsm_admits, fsm_id=500, scope="teleport")
    assert result["status"] == "error"
    assert "teleport" in result["message"]
    assert "arm" in result["message"]
    assert "loco" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_admits_refuses_bool_as_fsm_id_despite_being_int_subclass() -> None:
    """``True`` is an ``int(1)``; the tool must not silently accept it as FSM 1.

    A dict-key typo of ``True`` for a numeric FSM id is exactly the class
    of caller mistake this domain refusal exists to name - the gate's
    integer answer computed against ``True`` would compare against a set
    of three-digit ids and always return ``False``, and the caller would
    read that as "the robot is not arm-ready" when in fact they never
    asked a valid question.
    """
    result = _call(g1_fsm_admits, fsm_id=True, scope="arm")  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "bool" in result["message"]


def test_admits_refuses_non_int_fsm_id_and_names_the_type() -> None:
    """A non-int ``fsm_id`` is a type error, refused with the type named."""
    for bad in ("500", None, 500.0, [500]):
        result = _call(g1_fsm_admits, fsm_id=bad, scope="arm")  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert type(bad).__name__ in result["message"]


def test_loco_gate_is_a_subset_of_arm_gate() -> None:
    """The locomotion admission set is narrower than the arm-SDK one.

    This is not the tool's rule - it is the driver's, spelled once in
    :data:`WALK_FSMS` (a subset of :data:`HANDSHAKE_FSMS` because sitting
    at FSM 500 accepts arm gestures but not walking). The pin is here so
    a future edit that broke the subset relation (either by widening
    ``WALK_FSMS`` or by narrowing ``HANDSHAKE_FSMS`` past it) surfaces at
    this cell rather than as a firmware refusal the driver's write path
    would raise on a caller that asked the gate first.
    """
    assert WALK_FSMS <= HANDSHAKE_FSMS
    # The empty set would technically satisfy the subset relation but
    # would mean the driver admits no locomotion writes on any FSM,
    # which is not the shipped contract.
    assert WALK_FSMS


def test_list_scope_names_are_the_scope_the_driver_gate_accepts() -> None:
    """The scope labels this verb accepts are the same names ``_check_motion_gates`` uses.

    The driver's gate takes a scope name and picks one of two admission
    sets. This verb's scope map has to name the same choices - otherwise
    a caller who read ``"arm"`` here and passed it to the driver would
    have to know a translation table this file did not name. The two
    canonical scope names the driver's gate accepts today are ``"arm"``
    and ``"loco"``; the shipped map surfaces both.
    """
    assert set(_SCOPE_SETS) == {"arm", "loco"}
