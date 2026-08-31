"""``cleanup`` and ``stop`` read whether the control loop actually halted.

:meth:`~strands_robots.drivers.g1._ControlLoop.stop` signals the loop and joins
its thread within a budget, and *returns whether the thread joined*.  A
caller-supplied policy that outlasts that budget - a remote inference call is
the ordinary case, which :meth:`G1Driver.stop_task`'s own docstring says - leaves
the loop running and the return value ``False``.

``stop_task`` reads it.  The two teardown paths did not, and for ``cleanup`` that
was load-bearing: it closed ``_pubs`` and set it to ``None`` under the live
thread, so :meth:`_ControlLoop._emit_zero_torque` took its silent
``pubs is None`` return and the zero-torque shutdown frame was never published -
by ``cleanup``, nor later when the policy returned.  That is the fall
``cleanup``'s docstring says the path exists to prevent.

The shipped zero-torque test drives a policy that returns immediately, so its
join always succeeds and the discarded value is always ``True``: the fast path
cannot distinguish a checked halt from an unchecked one.  Every cell here that
grades the fix therefore blocks a policy past the join budget, which is why they
cost about two seconds each.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from typing import Any

import pytest

import strands_robots.drivers.g1 as g1_mod
from tests.drivers.test_g1_control_loop import (
    _ControlLoop,
    _RecordingPublisher,
    _StubCRC,
    _StubLowCmd,
)


@pytest.fixture(autouse=True)
def _stub_unitree_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a ``unitree_sdk2py`` stub for the duration of one test.

    The loop's write path and its zero-torque ``finally`` both call
    ``from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_`` inside
    the function body.  On a CI box without the SDK installed that import
    is an :class:`ImportError` the callers explicitly swallow, so the wire
    frames these cells count never leave the loop.  The stub matches the
    one :mod:`tests.drivers.test_g1_control_loop` installs for the same
    reason - imported alongside the reused fake ``_RecordingPublisher`` so
    both files run the same production lane.  ``monkeypatch.setitem``
    restores the previous entries on teardown.
    """
    root = types.ModuleType("unitree_sdk2py")
    idl = types.ModuleType("unitree_sdk2py.idl")
    default = types.ModuleType("unitree_sdk2py.idl.default")
    unitree_hg = types.ModuleType("unitree_sdk2py.idl.unitree_hg")
    unitree_hg_msg = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg")
    dds_ = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    utils = types.ModuleType("unitree_sdk2py.utils")
    crc = types.ModuleType("unitree_sdk2py.utils.crc")

    default.unitree_hg_msg_dds__LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    dds_.LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    crc.CRC = _StubCRC  # type: ignore[attr-defined]

    for name, mod in [
        ("unitree_sdk2py", root),
        ("unitree_sdk2py.idl", idl),
        ("unitree_sdk2py.idl.default", default),
        ("unitree_sdk2py.idl.unitree_hg", unitree_hg),
        ("unitree_sdk2py.idl.unitree_hg.msg", unitree_hg_msg),
        ("unitree_sdk2py.idl.unitree_hg.msg.dds_", dds_),
        ("unitree_sdk2py.utils", utils),
        ("unitree_sdk2py.utils.crc", crc),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


#: The joint every fixture commands.  Its commanded gain is the discriminator:
#: the zero-torque frame is the one that asks for zero stiffness.
_JOINT = "left_knee"

#: Longer than :meth:`_ControlLoop.stop`'s join budget, so a policy parked on
#: this event is guaranteed to outlast it.
_LONGER_THAN_THE_JOIN_BUDGET = 30.0


class _AlwaysReadyMotionSwitcher:
    """A ``MotionSwitcherClient`` stand-in that always reports FSM 500.

    ``read_fsm_id`` calls ``CheckMode()`` and decodes ``(status, {"name",
    "form"})``; 500 is in ``HANDSHAKE_FSMS``, which is the FSM these cells'
    driver is primed with.  Returns instantly, so it models a healthy wire and
    not the slow one graded in
    ``test_g1_fsm_refresh_is_off_the_control_loop_thread``.
    """

    def CheckMode(self) -> Any:  # noqa: N802 - SDK spelling
        return (0, {"name": "ai", "form": 500})


def _connected_driver(publisher: _RecordingPublisher) -> Any:
    """A real ``G1Driver`` wired to ``publisher``, in the state a rollout runs in.

    Constructed without DDS: the lifecycle methods and the task-admission lock
    are what these cells grade, and every attribute the loop reads is a real
    value so a typo fails on ``AttributeError`` rather than reading a stand-in.
    """
    driver = g1_mod.G1Driver(
        port="127.0.0.1",
        network_interface="lo",
        # A rollout runs on a driver whose FSM producer is wired: the loop's
        # refresher thread re-reads through this client, which is what keeps
        # the cached FSM inside the per-step gate's staleness bound.  Without
        # a factory the refresher has nothing to read and a long rollout would
        # exit on a stale cache - a real refusal, but not the one these cells
        # grade.
        motion_switcher_client_factory=lambda _iface: _AlwaysReadyMotionSwitcher(),
    )
    driver._pubs = publisher  # type: ignore[assignment]
    driver._subs = None
    driver._connected = True
    driver._mode_machine = 9
    driver._fsm_id = 500
    # ``_fsm_id`` and ``_fsm_read_at`` are one fact in production - the only
    # writer of the id stamps the time in the same branch - so a fixture that
    # assigns the id assigns the stamp too.  Set here as well as left to the
    # refresher because the loop is started directly (no ``run_policy``
    # admission), and step 1 can beat the refresher's first read.
    driver._fsm_read_at = time.monotonic()
    driver._battery = {"pct": 80.0}
    driver._imu = {"rpy": [0.0, 0.0, 0.0]}
    return driver


def _zero_torque_frames(publisher: _RecordingPublisher) -> int:
    """Count frames on ``publisher`` that command zero stiffness on ``_JOINT``."""
    slot = g1_mod._G1_JOINT_INDEX[_JOINT]
    return sum(1 for call in publisher.calls if call[2].motor_cmd[slot].kp == 0.0)


def _await_exit(loop: _ControlLoop, timeout: float = 10.0) -> None:
    """Poll until ``loop`` has left its thread, so a frame count is settled."""
    deadline = time.monotonic() + timeout
    while loop.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not loop.is_running, "loop did not exit within timeout"


class _Rollout:
    """A started loop whose policy parks until :meth:`release` is called.

    ``blocking=False`` gives the fast path the shipped suite already drives, so
    one helper serves both the regression cells and their controls.
    """

    def __init__(self, blocking: bool) -> None:
        self.publisher = _RecordingPublisher()
        self.driver = _connected_driver(self.publisher)
        self._released = threading.Event()

        def policy(_observation: Any) -> dict[str, float]:
            if blocking:
                self._released.wait(_LONGER_THAN_THE_JOIN_BUDGET)
            return {_JOINT: 0.0}

        self.loop = _ControlLoop(driver=self.driver, policy=policy, duration=60.0, n_steps=None)
        self.driver._loop = self.loop
        self.loop.start()
        # Let the loop reach the policy call before the teardown under test.
        time.sleep(0.05)

    def release(self) -> None:
        """Let a parked policy return, then wait for the loop to exit."""
        self._released.set()
        _await_exit(self.loop)


@pytest.fixture
def unjoined() -> Any:
    """A rollout whose policy outlasts the join budget.  Always released."""
    rollout = _Rollout(blocking=True)
    try:
        yield rollout
    finally:
        rollout.release()


@pytest.fixture
def joins() -> Any:
    """A rollout whose policy returns immediately, so the join succeeds."""
    rollout = _Rollout(blocking=False)
    try:
        yield rollout
    finally:
        rollout.release()


class TestThePremise:
    """The facts the regression cells rest on, stated independently."""

    def test_stop_reports_whether_the_thread_joined(self, unjoined: Any) -> None:
        assert unjoined.loop.stop("stop_task") is False
        assert unjoined.loop.is_running

    def test_a_returning_policy_joins_inside_the_budget(self, joins: Any) -> None:
        assert joins.loop.stop("stop_task") is True

    def test_the_zero_torque_frame_is_dropped_when_the_publisher_is_gone(self) -> None:
        """``_emit_zero_torque`` returns silently, which is why losing it is quiet."""
        publisher = _RecordingPublisher()
        driver = _connected_driver(publisher)
        loop = _ControlLoop(driver=driver, policy=lambda _o: {_JOINT: 0.0}, duration=1.0, n_steps=None)
        driver._pubs = None
        # Its ``-> None`` return is the type checker's to state; what a cell
        # can add is that the frame is dropped rather than raised.
        loop._emit_zero_torque()
        assert publisher.calls == []


class TestCleanupKeepsThePublisherForAnUnjoinedLoop:
    """The regression: the shutdown frame survives a policy that overruns."""

    def test_the_zero_torque_frame_reaches_the_wire(self, unjoined: Any) -> None:
        unjoined.driver.cleanup()
        unjoined.release()
        assert _zero_torque_frames(unjoined.publisher) == 1

    def test_the_publisher_is_not_closed_under_the_live_loop(self, unjoined: Any) -> None:
        unjoined.driver.cleanup()
        assert unjoined.loop.is_running
        assert unjoined.publisher.closed is False

    def test_the_publisher_reference_is_kept(self, unjoined: Any) -> None:
        unjoined.driver.cleanup()
        assert unjoined.driver._pubs is unjoined.publisher

    def test_the_overrun_is_reported_at_error(self, unjoined: Any, caplog: Any) -> None:
        with caplog.at_level(logging.ERROR, logger=g1_mod.__name__):
            unjoined.driver.cleanup()
        messages = [record.getMessage() for record in caplog.records]
        assert any("did not join" in message for message in messages), messages

    def test_the_report_names_calling_cleanup_again(self, unjoined: Any, caplog: Any) -> None:
        with caplog.at_level(logging.ERROR, logger=g1_mod.__name__):
            unjoined.driver.cleanup()
        assert any("cleanup() again" in record.getMessage() for record in caplog.records)


class TestStopReportsAnUnjoinedLoop:
    """``stop`` has no envelope, so an overrun is logged rather than returned."""

    def test_the_overrun_is_reported_at_error(self, unjoined: Any, caplog: Any) -> None:
        import asyncio

        with caplog.at_level(logging.ERROR, logger=g1_mod.__name__):
            asyncio.run(unjoined.driver.stop())
        messages = [record.getMessage() for record in caplog.records]
        assert any("did not join" in message for message in messages), messages


class TestNothingLeaksEitherWay:
    """Holds on both trees, and is the reason keeping the publisher is safe.

    Deferring the release is only defensible if the resources are still
    released, so these say what must stay true whichever branch ``cleanup``
    takes.  They are guards rather than regression cells: each one passes on
    the pre-fix code too, where the first ``cleanup`` had already closed the
    publisher.
    """

    def test_a_second_cleanup_releases_the_publisher(self, unjoined: Any) -> None:
        unjoined.driver.cleanup()
        unjoined.release()
        unjoined.driver.cleanup()
        assert unjoined.driver._pubs is None
        assert unjoined.publisher.closed is True

    def test_the_subscribers_close_either_way(self, unjoined: Any) -> None:
        """The loop never reads them, so holding them back buys nothing."""
        subscribers = _RecordingPublisher()
        unjoined.driver._subs = subscribers  # type: ignore[assignment]
        unjoined.driver.cleanup()
        assert subscribers.closed is True
        assert unjoined.driver._subs is None

    def test_stop_releases_no_resources_at_all(self, unjoined: Any) -> None:
        """``stop`` halts the loop; ``cleanup`` is what releases the wire."""
        import asyncio

        asyncio.run(unjoined.driver.stop())
        assert unjoined.driver._pubs is unjoined.publisher
        assert unjoined.publisher.closed is False


class TestTheJoinedPathIsUnchanged:
    """Over-reach controls: reading the outcome must cost the fast path nothing."""

    def test_the_publisher_is_released(self, joins: Any) -> None:
        joins.driver.cleanup()
        assert joins.driver._pubs is None
        assert joins.publisher.closed is True

    def test_the_zero_torque_frame_still_goes_out(self, joins: Any) -> None:
        joins.driver.cleanup()
        assert _zero_torque_frames(joins.publisher) == 1

    def test_nothing_is_reported_at_error(self, joins: Any, caplog: Any) -> None:
        with caplog.at_level(logging.ERROR, logger=g1_mod.__name__):
            joins.driver.cleanup()
        assert [record.getMessage() for record in caplog.records] == []

    def test_cleanup_with_no_loop_still_releases_everything(self) -> None:
        publisher = _RecordingPublisher()
        driver = _connected_driver(publisher)
        driver.cleanup()
        assert driver._pubs is None
        assert publisher.closed is True
        assert driver._connected is False

    def test_cleanup_is_still_idempotent(self) -> None:
        driver = _connected_driver(_RecordingPublisher())
        driver.cleanup()
        driver.cleanup()
        assert driver._pubs is None


class TestEveryTeardownPathReadsTheOutcome:
    """Derived, so a fourth path cannot land discarding the join result.

    Read structurally rather than behaviourally: a teardown added later would
    have no cell here, and the point is that it is held to the rule on arrival.
    """

    @staticmethod
    def _halt_call_sites() -> dict[str, bool]:
        """Map enclosing function name -> whether it reads ``loop.stop``'s value."""
        import ast

        source = g1_mod.__loader__.get_source(g1_mod.__name__)  # type: ignore[union-attr]
        assert source is not None
        tree = ast.parse(source)
        discarded: dict[str, bool] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for statement in ast.walk(node):
                if not isinstance(statement, ast.Expr):
                    continue
                call = statement.value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "stop"
                    and "loop" in ast.unparse(call.func.value)
                ):
                    discarded[node.name] = True
        return discarded

    def test_the_scan_finds_the_halt_calls_at_all(self) -> None:
        """Non-vacuity: the pattern this rule bans must be expressible."""
        import ast

        planted = ast.parse("def teardown(self):\n    loop.stop('x')\n")
        found = [
            node.name
            for node in ast.walk(planted)
            if isinstance(node, ast.FunctionDef)
            for statement in ast.walk(node)
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "stop"
        ]
        assert found == ["teardown"]

    def test_no_function_discards_the_halt_outcome(self) -> None:
        assert self._halt_call_sites() == {}
