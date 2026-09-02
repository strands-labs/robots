"""The G1's battery floor is reachable now that the FSM producer is wired.

Predecessor to this file (same path, before this PR) pinned the reverse
contract: on a driver with every other field healthy, ``_check_motion_gates``
refused with ``FSM id unknown`` because :attr:`G1Driver._fsm_id` had no
producer.  The predecessor's own trigger was "if a real writer lands, replace
this test file with one that grades the new reachability directly" -- this
file is that replacement.

Three contracts are stated here as literals so a change to any of them fires
this file, not a distant one:

1. On a fully-healthy driver *whose motion-switcher factory refuses to open*,
   ``_check_motion_gates`` still refuses with ``FSM id unknown`` -- the
   backward-compatible path for a driver constructed without a factory on a
   box that has no ``unitree_sdk2py``.  This is the fallback the predecessor
   test file pinned, held here so it is not accidentally removed the day
   ``_refresh_fsm_id``'s failure semantics change.

2. On a fully-healthy driver with a motion-switcher factory whose
   ``CheckMode()`` returns a healthy handshake FSM (501) and a pack under
   the floor, ``send_action`` refuses with the *battery* message, not the
   FSM one.  That is the new reachability: the battery floor now runs.

3. The FSM check precedes the battery-floor check inside
   :meth:`_check_motion_gates`.  Contract (2) reads whichever gate fires
   first, so the ordering is load-bearing: moving the battery check ahead
   of the FSM check makes (2)'s message-shape assertion trivial and lets a
   silently un-wired FSM open the gate.  Stating the ordering directly means
   a reorder fires a cell whose name says "ordering".

The predecessor's ``_fsm_id has exactly one assignment`` assertion is
retired: the wire that flipped this file is the second and third writer, so
counting one is exactly the invariant this PR removes.  A new invariant
takes its place: :attr:`G1Driver._fsm_id` is written from
:meth:`_refresh_fsm_id` (which reads :func:`read_fsm_id`).  Grading that
plumbing rather than the write count means a refactor that renames the
method but keeps the wire fires nothing here, while a refactor that removes
the read-side call fires a cell whose message names the seam.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import strands_robots.drivers.g1 as g1_module
from strands_robots.drivers.g1 import G1Driver

# ``_CRITICAL_PCT`` is far below the ``_HEALTHY_FLOOR_PCT`` so an ordering
# mistake -- battery checked before FSM -- would flip the refusal text on
# the ``fsm_wired`` case below.  Literal values, not derivations from the
# module's own constants.
_CRITICAL_PCT = 1.0
_HEALTHY_FLOOR_PCT = 15.0
_HEALTHY_PACK_PCT = 92.0

# ``_HEALTHY_MODE_MACHINE`` is what the ``rt/lowstate`` decoder produces on a
# healthy G1 (uint8 layout id echoed by the vendor); ``_check_motion_gates``
# only refuses on ``mode_machine`` when it is ``None``.
_HEALTHY_MODE_MACHINE = 9

# ``501`` is in :data:`HANDSHAKE_FSMS` and :data:`WALK_FSMS`; the gate admits
# it for every scope.  Literal, not derived, so a set-membership change in
# ``_g1_common`` (which would legitimately want a different value here) does
# not silently pass this file.
_HEALTHY_FSM_ID = 501


class _RecordingMotionSwitcherClient:
    """A minimally-real ``MotionSwitcherClient`` stand-in.

    :func:`strands_robots.tools.g1._motion_switcher.read_fsm_id` calls
    ``CheckMode()`` and decodes the return.  Queuing the return here lets
    each cell fabricate the wire it wants graded, without importing the SDK.

    The one method the SDK-side factory calls at open time is ``Init()``; we
    accept a call and record it, but return nothing, matching the SDK's own
    signature.
    """

    def __init__(self, check_mode_return: Any) -> None:
        self._return = check_mode_return
        self.check_mode_calls: int = 0
        self.init_calls: int = 0

    def Init(self) -> None:  # noqa: N802 - SDK spelling
        self.init_calls += 1

    def CheckMode(self) -> Any:  # noqa: N802 - SDK spelling
        self.check_mode_calls += 1
        return self._return


def _pack(pct: float) -> dict[str, float | int]:
    return {
        "pct": pct,
        "current": 0.0,
        "cycle": 0,
        "t": 0.0,
    }


def _gate_ast() -> ast.FunctionDef:
    """Return :meth:`G1Driver._check_motion_gates` from the shipped source."""
    tree = ast.parse(inspect.getsource(g1_module))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "G1Driver":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "_check_motion_gates":
                    return child
    raise AssertionError("G1Driver._check_motion_gates not found")


def test_a_driver_with_no_motion_switcher_factory_still_refuses_with_fsm_unknown() -> None:
    """The backward-compatible path: no factory, no SDK, no gate open.

    A driver constructed without ``motion_switcher_client_factory`` on a box
    with no ``unitree_sdk2py`` has :meth:`_open_motion_switcher_client`
    refuse (the default lazy loader raises :class:`ImportError`), which
    leaves :attr:`_fsm_id` at ``None`` and produces the shipped refusal.
    This is the same message the predecessor test file graded; held here so
    a change to :meth:`_refresh_fsm_id`'s failure semantics fires a cell
    whose name says "no factory" rather than silently making the fallback
    reachable.
    """
    driver = G1Driver(
        tool_name="g1",
        port="1.2.3.4",
        battery_floor_pct=_HEALTHY_FLOOR_PCT,
    )
    driver._connected = True
    driver._mode_machine = _HEALTHY_MODE_MACHINE
    driver._battery = _pack(_HEALTHY_PACK_PCT)
    # No factory, no SDK -> ``_fsm_id`` never gets a producer, refusal fires.
    assert driver._motion_switcher_client_factory is None

    result = driver.send_action({"left_shoulder_pitch": 0.0})

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "FSM id unknown" in text
    assert "motion-switcher" in text
    # And the open-error was recorded on the driver for :meth:`get_status`
    # to surface -- ``get_status`` reports it under
    # ``motion_switcher_open_error``.
    assert driver._motion_switcher_open_error is not None
    # The exact failure text depends on whether ``unitree_sdk2py`` is
    # installed on the box running the tests.  On CI (SDK absent) the
    # importlib call raises ``ModuleNotFoundError`` and the message names
    # the SDK package or the client class.  On a developer box with the SDK
    # installed but no DDS bus reachable, ``Init()`` raises deep inside the
    # C bindings (``AttributeError: 'NoneType' object has no attribute
    # '_ref'``) and the message names neither.  Both are the same defect
    # from the caller's perspective -- the client did not open -- so we
    # grade the invariant the driver actually preserves: an error string
    # was captured on the driver rather than raised through the gate.
    assert isinstance(driver._motion_switcher_open_error, str)
    assert "motion-switcher client could not be opened" in driver._motion_switcher_open_error


def test_battery_floor_reaches_with_a_wired_fsm_and_a_critical_pack() -> None:
    """The reachability contract this file replaces the predecessor with.

    A driver with a motion-switcher factory returning a healthy handshake
    FSM (501) and a pack under the floor refuses for the *battery*, not
    the FSM.  The predecessor pinned "refuses for FSM because the battery
    guard is unreachable"; the wire flipped, so this cell pins the new
    reachability directly.
    """
    client = _RecordingMotionSwitcherClient(
        check_mode_return=(0, {"name": "ai", "form": _HEALTHY_FSM_ID}),
    )
    driver = G1Driver(
        tool_name="g1",
        port="1.2.3.4",
        battery_floor_pct=_HEALTHY_FLOOR_PCT,
        motion_switcher_client_factory=lambda _iface: client,
    )
    driver._connected = True
    driver._mode_machine = _HEALTHY_MODE_MACHINE
    driver._battery = _pack(_CRITICAL_PCT)

    result = driver.send_action({"left_shoulder_pitch": 0.0})

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    # The refusal now names the battery, not the FSM.  Both halves are
    # stated: an ordering mistake that put the battery check first would
    # trip the ``"FSM" not in text`` assertion in the "no factory" cell
    # above, and this one would then be a tautology; stating both here
    # documents the pair.
    assert "battery" in text
    assert f"{_CRITICAL_PCT:.1f}" in text
    assert f"{_HEALTHY_FLOOR_PCT:.1f}" in text
    assert "FSM id unknown" not in text
    # And the wire was actually reached: the recording client saw the call.
    assert client.check_mode_calls >= 1
    # ``_fsm_id`` was written to the healthy value, and the reading is
    # stashed for :meth:`get_status` to surface with the mode label.
    assert driver._fsm_id == _HEALTHY_FSM_ID
    assert driver._last_fsm_reading is not None
    assert driver._last_fsm_reading.mode_name == "ai"


def test_a_wired_fsm_that_reports_released_mode_keeps_the_gate_shut() -> None:
    """``name == ""`` is the "no motion mode selected" reading; refuse honestly.

    :func:`read_fsm_id` returns ``fsm_id=None`` on ``name == ""`` (the SDK's
    high-level-released state), which
    :meth:`G1Driver._refresh_fsm_id` maps back to ``self._fsm_id = None`` so
    the gate refuses.  Grading this here means a factory that reports
    released-mode does not silently open the gate on a stale cache.
    """
    client = _RecordingMotionSwitcherClient(
        check_mode_return=(0, {"name": ""}),
    )
    driver = G1Driver(
        tool_name="g1",
        port="1.2.3.4",
        battery_floor_pct=_HEALTHY_FLOOR_PCT,
        motion_switcher_client_factory=lambda _iface: client,
    )
    driver._connected = True
    driver._mode_machine = _HEALTHY_MODE_MACHINE
    driver._battery = _pack(_HEALTHY_PACK_PCT)
    # Prime a stale cache the SDK's released-mode reading must clear.
    driver._fsm_id = _HEALTHY_FSM_ID

    result = driver.send_action({"left_shoulder_pitch": 0.0})

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "FSM id unknown" in text
    # The reading is stashed with the empty mode label so ``get_status`` can
    # tell the caller "no mode selected" rather than only "FSM unknown".
    assert driver._last_fsm_reading is not None
    assert driver._last_fsm_reading.mode_name == ""
    assert driver._last_fsm_reading.refusal is None
    # And the wire cleared the stale cache.
    assert driver._fsm_id is None


def test_the_fsm_producer_seam_is_refresh_fsm_id_calling_read_fsm_id() -> None:
    """The plumbing invariant that replaces the predecessor's write-count pin.

    The predecessor counted assignments to ``self._fsm_id`` and required
    exactly one (the ``None`` initialiser).  Now that a producer exists,
    the honest replacement is: :meth:`_refresh_fsm_id` exists, and it calls
    :func:`read_fsm_id`.  A refactor that renames either without breaking
    the wire fires nothing here; a refactor that removes the read-side
    call fires this cell with a message that names the seam.
    """
    source = inspect.getsource(g1_module)
    tree = ast.parse(source)

    refresh: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "G1Driver":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "_refresh_fsm_id":
                    refresh = child
                    break
    assert refresh is not None, (
        "G1Driver._refresh_fsm_id is the seam the FSM gate reads through. "
        "Its removal is the change this cell exists to fire on -- restore "
        "the method or replace this test with one that grades the new seam."
    )

    read_calls = [
        node
        for node in ast.walk(refresh)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "read_fsm_id"
    ]
    assert len(read_calls) == 1, (
        f"_refresh_fsm_id calls read_fsm_id() {len(read_calls)} times; expected "
        "exactly one call.  A second call would decode the same reading twice; "
        "removing the call disconnects the FSM producer from the gate."
    )


def test_the_fsm_check_precedes_the_battery_floor_in_the_gate() -> None:
    """The gate tests ``_fsm_id`` before it compares the pack to the floor.

    The reachability contract in
    :func:`test_battery_floor_reaches_with_a_wired_fsm_and_a_critical_pack`
    reads whichever gate fires first, so this ordering is load-bearing.
    Grading the two ``ast.If`` line numbers here means a reordering fails
    a cell whose name says "ordering".
    """
    gate = _gate_ast()
    fsm_checks = [
        node.lineno
        for node in ast.walk(gate)
        if isinstance(node, ast.If) and "self._fsm_id is None" in ast.unparse(node.test)
    ]
    floor_checks = [
        node.lineno
        for node in ast.walk(gate)
        if isinstance(node, ast.If) and "self._battery_floor_pct" in ast.unparse(node.test)
    ]
    assert len(fsm_checks) == 1, f"expected one ``self._fsm_id is None`` test, found {fsm_checks}"
    assert len(floor_checks) == 1, (
        f"expected one comparison against ``self._battery_floor_pct``, found {floor_checks}. "
        "Two of them leave the order the caller meets ambiguous."
    )
    assert fsm_checks[0] < floor_checks[0], (
        f"the battery floor is compared at line {floor_checks[0]}, ahead of the "
        f"``_fsm_id`` check at line {fsm_checks[0]}. Reordering makes the reachability "
        "cell above trip for the wrong reason; state a new reachability test if the "
        "reorder is intended."
    )


def test_refresh_is_called_before_the_fsm_check_in_the_gate() -> None:
    """The FSM refresh happens *before* the ``_fsm_id is None`` check.

    Ordering the refresh after the ``None`` check would leave the first
    gate call refusing every time (cached ``None`` is what it sees), and
    every subsequent call reading a value already stashed by the previous
    call's *failed* refusal.  Grading the ordering here means a subtle
    reorder fires a cell whose name says "refresh order".
    """
    gate = _gate_ast()
    refresh_calls: list[int] = []
    fsm_none_checks: list[int] = []
    for node in ast.walk(gate):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_refresh_fsm_id":
            refresh_calls.append(node.lineno)
        elif isinstance(node, ast.If) and "self._fsm_id is None" in ast.unparse(node.test):
            fsm_none_checks.append(node.lineno)
    assert len(refresh_calls) == 1, (
        f"expected one call to ``self._refresh_fsm_id()`` in the gate, found {refresh_calls}"
    )
    assert len(fsm_none_checks) == 1
    assert refresh_calls[0] < fsm_none_checks[0], (
        f"``_refresh_fsm_id()`` is called at line {refresh_calls[0]}, after the "
        f"``_fsm_id is None`` check at line {fsm_none_checks[0]}. The refresh must run "
        "first so the first gate call reads the live FSM, not the initialiser's None."
    )
