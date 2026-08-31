"""The error-code lookup tools name exactly what the SDK's handlers surface.

The Unitree G1 SDK returns integer rc codes for every RPC and handler
response (``0`` OK, ``3104`` RPC timeout, ``7302`` invalid FSM id,
``7404`` gate-refused write, ...); the
:mod:`strands_robots.tools.g1._g1_common` module snapshots the
observed name for each of those codes into
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES` and every verb in
this package that surfaces a refusal quotes the same entry verbatim.
The :mod:`strands_robots.tools.g1.g1_error_codes` module ports that
lookup to the ``@tool`` surface so an agent that receives a
``refusal_code`` from any other verb can list the catalogue - or
decode one code by number - through the same tool interface every
other verb here answers on.

The tests here fix that contract without pulling the SDK: the module
is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every decoded text answer is read off
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES` rather than
restated in the tests, so a widen or narrow to the catalogue surfaces
here as a shape change rather than as a diverging table this file
would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The verbs answer against the
  module-level snapshot, not against a live SDK handler; a rc the
  SDK returns that is not in the snapshot surfaces here as
  ``known=False`` decidably, and a driver-side handler that starts
  returning a new rc without a snapshot update surfaces the gap
  through the ``unknown`` marker rather than as a KeyError.
* The exact decoded text for every code. The catalogue lives in
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` and every
  cell here reads the same source; a re-word of one entry lands in
  the constant once and this file picks it up without an assertion
  update.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_error_codes import (
    _UNKNOWN_CODE_TEXT,
    g1_decode_error_code,
    g1_list_error_codes,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on that:
    the wrapper's contract is that it returns the wrapped function's
    return value verbatim. This helper is where a shape drift would
    surface once, rather than at every call site (same idiom as
    ``test_g1_fsm_targets_reads_the_sdk_transition_set`` (removed)).
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an
    office bring-up. The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the error-code lookup
    verbs to it too (refs strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_error_codes")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_error_codes imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_names_ok_and_the_two_arm_side_refusals() -> None:
    """The catalogue names the OK code and the arm-side handler refusals.

    ``0`` (OK), ``7400`` (rt/armsdk topic occupied), ``7401`` (Arm is
    holding), ``7402`` (Invalid action id), ``7404`` (Invalid FSM id -
    the arm's own gate refusal): each of those is a rc a caller can
    receive from an arm-SDK write today, and each names a decoded
    text a verb already surfaces on refusal. This cell pins their
    presence so a widen or drop to the arm-side handler surfaces
    here as a diff, rather than as a silent divergence from the
    verbs' refusal strings.
    """
    for arm_rc in (0, 7400, 7401, 7402, 7404):
        assert arm_rc in ERR_CODES, (
            f"rc={arm_rc} names an arm-side handler refusal the verbs "
            "quote at write time; the catalogue must decode it."
        )


def test_the_snapshot_names_the_loco_side_refusals() -> None:
    """The catalogue names the locomotion-side handler refusals.

    ``7301`` (LocoState not available), ``7302`` (Invalid FSM id -
    the SDK's own transition refusal that
    ``g1_fsm_target_admits`` (removed lookup verb; clamps now live inline)
    surfaces), ``7303`` (Invalid task id - the same shape for the
    ``g1_loco_task_admits`` verb, which is not in this tree yet and so
    is named as a literal rather than cross-referenced): each of those
    is a rc a caller can receive from a loco-side write. This cell
    pins their presence so the catalogue and the lookup verbs quote
    the same numbers, and it does not depend on which of those verbs
    have landed.
    """
    for loco_rc in (7301, 7302, 7303):
        assert loco_rc in ERR_CODES, (
            f"rc={loco_rc} names a loco-side handler refusal the verbs "
            "quote at transition-plan time; the catalogue must decode it."
        )


def test_the_snapshot_names_the_rpc_transport_codes() -> None:
    """The catalogue names the RPC transport codes the SDK can return.

    ``3102`` (RPC_CLIENT_SEND fail), ``3103`` (RPC_CLIENT_API_NOT_REG),
    ``3104`` (RPC_CLIENT_API_TIMEOUT): each is a transport-level rc
    the SDK's ``_Call`` returns before a handler runs, and each is a
    code the driver's own status envelope may surface on a wedged
    RPC. This cell pins them because a caller who receives a
    ``fsm_refusal`` on the driver's status wire compares against
    those numbers to decide whether the RPC layer is wedged (retry-
    worthy) or a handler refused (not retry-worthy).
    """
    for rpc_rc in (3102, 3103, 3104):
        assert rpc_rc in ERR_CODES, (
            f"rc={rpc_rc} is an SDK transport-level code the driver may "
            "surface on a wedged RPC; the catalogue must decode it."
        )


def test_g1_list_error_codes_returns_the_whole_catalogue() -> None:
    """The verb's payload names the catalogue, the codes, and each text.

    ``count`` is the size of :data:`ERR_CODES`, ``error_codes`` is one
    descriptor per code (sorted ascending), ``codes`` is the sorted
    integer list alone. Every descriptor reads its ``text`` from the
    catalogue (not restated in the test body) so a re-word of one
    entry lands once.
    """
    result = _call(g1_list_error_codes)
    assert result["status"] == "success"
    assert result["count"] == len(ERR_CODES)
    assert result["codes"] == sorted(ERR_CODES)
    assert len(result["error_codes"]) == len(ERR_CODES)
    for descriptor in result["error_codes"]:
        code = descriptor["code"]
        assert descriptor["text"] == ERR_CODES[code], (
            f"descriptor for rc={code} carried text "
            f"{descriptor['text']!r} but catalogue holds "
            f"{ERR_CODES[code]!r}. The two must not diverge."
        )


def test_g1_list_error_codes_returns_fresh_containers() -> None:
    """A caller mutating the payload cannot poison the catalogue.

    The verb returns fresh lists and dicts; a mutation on the returned
    ``codes`` list or ``error_codes`` descriptors does not leak back
    into the module's constants. This cell is where a share-a-reference
    regression would surface once, not scattered across every call
    site (same guarantee as the ``fsm_targets`` snapshot in
    ``g1_fsm_targets`` (removed)).
    """
    result = _call(g1_list_error_codes)
    result["codes"].append(9999)
    result["error_codes"][0]["synthetic"] = True
    fresh = _call(g1_list_error_codes)
    assert 9999 not in fresh["codes"]
    assert "synthetic" not in fresh["error_codes"][0]


def test_g1_decode_error_code_resolves_a_known_code() -> None:
    """A code inside the catalogue is decoded to its text.

    ``7302`` is "Invalid FSM id (loco)" - the same rc the SDK's
    ``SetFsmId`` handler returns and that
    ``g1_fsm_target_admits`` (removed lookup verb; clamps now live inline)
    surfaces on a refused query. The two sides quote the same text
    because both read the same catalogue.
    """
    result = _call(g1_decode_error_code, code=7302)
    assert result["status"] == "success"
    assert result["known"] is True
    assert result["query"] == {"code": 7302}
    assert result["text"] == ERR_CODES[7302]


def test_g1_decode_error_code_resolves_the_ok_code() -> None:
    """``rc=0`` is the OK code and decodes to the catalogue's OK text.

    Cell pins the OK path so a caller composing a "did the write
    succeed" branch reads the same text on the success side as on the
    refusal side. If the catalogue ever re-words "OK" to (say) "ok"
    the change lands in one place.
    """
    result = _call(g1_decode_error_code, code=0)
    assert result["known"] is True
    assert result["text"] == "OK"


def test_g1_decode_error_code_flags_an_unknown_code() -> None:
    """A code outside the catalogue is decodably ``unknown``.

    ``9999`` is not a G1 rc. The verb reports ``known=False`` and
    carries the catalogue's ``unknown`` marker under ``text`` so a
    caller composing an error message never has to branch on a
    missing key: the returned envelope always names something. This
    matches the contract
    :func:`~strands_robots.tools.g1._g1_common.decode_code` would
    render for the same number.
    """
    result = _call(g1_decode_error_code, code=9999)
    assert result["status"] == "success"
    assert result["known"] is False
    assert result["query"] == {"code": 9999}
    assert result["text"] == _UNKNOWN_CODE_TEXT


def test_g1_decode_error_code_admits_a_negative_rc_as_unknown() -> None:
    """A negative rc is admitted decidably as ``known=False``.

    The catalogue's own keys are all non-negative, so a negative rc is
    by construction outside it. It is admitted rather than refused
    because the in-tree renderer of a rc is already total over every
    integer: :func:`~strands_robots.tools.g1._g1_common.decode_code`
    answers ``"-1 (unknown)"``, and a lookup verb that refused the
    same value would be narrower than the renderer whose text it
    quotes. The ``-1`` convention itself comes from the neon bundle's
    locomotion wrapper (``cagataycali/neon-the-g1``), which stores it
    for an SDK call that raised instead of returning a rc; no module
    in this tree writes it today, which is exactly why this cell pins
    the decidable answer now rather than after that wrapper is ported
    (refs strands-labs/robots#358). The verb does not refuse the query
    - a caller may need to display it - but it names the code as
    unknown rather than inventing text for it.
    """
    result = _call(g1_decode_error_code, code=-1)
    assert result["status"] == "success"
    assert result["known"] is False
    assert result["text"] == _UNKNOWN_CODE_TEXT


def test_g1_decode_error_code_refuses_a_bool_code() -> None:
    """``bool`` is not an ``int`` this verb should silently accept.

    Python's ``bool`` is a subclass of ``int`` so ``True == 1``, but a
    caller passing ``True`` for a rc is a type mistake, not a valid
    decode query. The verb refuses so a mis-typed argument surfaces
    at the lookup rather than reaching the catalogue's own dict
    coercion (same rule as
    ``g1_fsm_target_admits`` (removed lookup verb; clamps now live inline)).
    """
    result = _call(g1_decode_error_code, code=True)
    assert result["status"] == "error"
    assert "bool" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_g1_decode_error_code_refuses_a_non_int_code() -> None:
    """A non-int, non-bool ``code`` surfaces the type in the refusal.

    ``"7302"`` looks correct to a human reader but is a string; the
    refusal names the type and the value the caller passed, so a
    caller sees which of their many parallel tool calls hit the
    wrong shape (same refusal pattern as
    ``g1_fsm_target_admits`` (removed lookup verb; clamps now live inline)).
    """
    result = _call(g1_decode_error_code, code="7302")  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "str" in result["message"]
    assert "strands-labs/robots#358" in result["message"]
