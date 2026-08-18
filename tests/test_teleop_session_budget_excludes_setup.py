"""``teleoperate(duration=...)`` bounds teleoperation, not setup.

The control loop judges every frame with two helpers from
``strands_robots.mesh.security``, imported lazily because the mesh package
reaches ``strands_robots.simulation`` and the mixin must not depend on it. That
import used to be resolved inside ``_teleop_loop``, i.e. **after**
``teleoperate`` stamped ``_teleop_start_mono``, which is the base the deadline is
built from. Importing the mesh package costs ~2s on a cold process (it pulls
~840 modules), so the first session in a process spent that time between the
clock starting and the first poll:

* a ``duration`` under the import cost ended the loop on its first deadline
  check, having polled the leader **zero** times, and reported
  ``status="success"`` at "0 frames, 0.0Hz" - the outcome
  ``positive_finite_number_error(duration, ...)`` refuses ``duration=0`` for, and
  the outcome ``_teleop_stats``' own status derivation exists to refuse;
* any longer session was silently shortened by the same amount.

Both are pinned here by charging a fixed, deterministic cost to resolving that
one module: the module is dropped from ``sys.modules`` and a ``sys.meta_path``
finder sleeps while it is looked up. That reproduces the shape on any machine
without depending on the real import being slow, and it fails on either side of
the fix for the same reason a cold process did.

The two boundaries that must not move are pinned as well: a warm session is
unchanged, and a session refused for an unusable ``duration`` still resolves
nothing - the knob guards run before setup, so a refusal must not pay for it.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import time
from typing import Any

import pytest

from strands_robots import teleop_mixin
from tests.test_teleop import FakeHost, FakeTeleop

#: The module whose resolution used to be charged to the session budget.
_MESH_SLEW_MODULE = "strands_robots.mesh.security"

#: Cost charged to resolving it. Comfortably above the session budgets below so
#: a pre-fix session cannot reach its first poll, and small enough to keep the
#: suite quick.
_RESOLVE_COST_S = 0.8

#: Session budget under test. Well below ``_RESOLVE_COST_S``, which is the whole
#: point: this is the duration a cold process used to spend entirely on setup.
_SESSION_S = 0.3


class _SleepingFinder:
    """Meta-path finder that charges a fixed cost to resolve one module.

    Returns ``None`` so the real finders still resolve it - the cost is added,
    nothing about the import's outcome changes.
    """

    def __init__(self, target: str, delay: float) -> None:
        self.target = target
        self.delay = delay
        self.calls = 0

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:  # noqa: ARG002
        if fullname == self.target:
            self.calls += 1
            time.sleep(self.delay)
        return None


def _charge_a_cold_resolve(monkeypatch: pytest.MonkeyPatch) -> _SleepingFinder:
    """Make resolving the mesh slew module cost ``_RESOLVE_COST_S``.

    Only the leaf module is dropped: its parent package stays in
    ``sys.modules``, so this does not re-import the mesh package (and cannot
    hand a later test a second copy of it).
    """
    finder = _SleepingFinder(_MESH_SLEW_MODULE, _RESOLVE_COST_S)
    monkeypatch.delitem(sys.modules, _MESH_SLEW_MODULE, raising=False)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    return finder


def _host_with_leader() -> tuple[FakeHost, FakeTeleop]:
    host = FakeHost()
    dev = FakeTeleop({"a.pos": 1.0})
    host.attach_teleop(dev, name="lead")
    return host, dev


class TestTheBudgetMeasuresTeleoperationNotSetup:
    """A session's ``duration`` must be time the leader is actually polled."""

    def test_a_cold_resolve_does_not_consume_the_whole_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        host, dev = _host_with_leader()
        finder = _charge_a_cold_resolve(monkeypatch)

        res = host.teleoperate(hz=50.0, duration=_SESSION_S, block=True)

        assert finder.calls >= 1, (
            f"premise: {_MESH_SLEW_MODULE} must be resolved cold for this to measure "
            f"anything; it was already in sys.modules, so no cost was charged"
        )
        assert res["status"] == "success"
        assert dev.get_action_calls > 0, (
            f"a {_SESSION_S}s session polled the leader {dev.get_action_calls} times: the "
            f"{_RESOLVE_COST_S}s resolve of {_MESH_SLEW_MODULE} was charged to the budget, so the "
            f"deadline had already passed by the loop's first check. Resolve it before the "
            f"session clock starts."
        )
        assert host.sent, "no frame reached send_action"

    def test_the_reported_elapsed_excludes_the_resolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        host, _dev = _host_with_leader()
        finder = _charge_a_cold_resolve(monkeypatch)

        res = host.teleoperate(hz=50.0, duration=_SESSION_S, block=True)

        assert finder.calls >= 1, "premise: the resolve must be cold"
        elapsed = res["content"][1]["json"]["elapsed_s"]
        assert elapsed < _RESOLVE_COST_S, (
            f"reported elapsed {elapsed:.3f}s for a {_SESSION_S}s session, which is at least the "
            f"{_RESOLVE_COST_S}s resolve cost: the session clock was started before setup "
            f"finished, so the report describes setup plus teleoperation"
        )

    def test_a_background_session_reports_started_only_after_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``block=False`` claims the session started; setup must be done by then.

        Otherwise the call returns "Teleoperation started", and for the next
        couple of seconds ``get_teleoperate_status()`` reports a running session
        at 0 frames and 0Hz because the loop thread is still importing.
        """
        host, _dev = _host_with_leader()
        finder = _charge_a_cold_resolve(monkeypatch)

        start = time.perf_counter()
        res = host.teleoperate(hz=50.0, duration=None, block=False)
        call_s = time.perf_counter() - start
        try:
            assert res["status"] == "success"
            assert finder.calls >= 1, "premise: the resolve must be cold"
            assert call_s >= _RESOLVE_COST_S / 2, (
                f"teleoperate(block=False) returned in {call_s:.3f}s while resolving "
                f"{_MESH_SLEW_MODULE} costs {_RESOLVE_COST_S}s: the cost was deferred to the loop "
                f"thread, so the session was reported started before it could poll anything"
            )
        finally:
            host.stop_teleoperate()


class TestTheResolveIsWiredAheadOfTheClock:
    """Pinned over the source: reordering these two statements restores the bug.

    The ordering is what makes the fix work, and it is invisible in review - both
    statements read fine either way round.
    """

    @staticmethod
    def _teleoperate_tree() -> ast.Module:
        source = textwrap.dedent(inspect.getsource(teleop_mixin.TeleopMixin.teleoperate))
        return ast.parse(source)

    def test_the_module_is_resolved_before_the_session_clock_is_stamped(self) -> None:
        tree = self._teleoperate_tree()

        resolves = [line for line, mod in _import_module_calls(tree) if mod == _MESH_SLEW_MODULE]
        stamps = _self_assignment_lines(tree, "_teleop_start_mono")

        assert stamps, "premise: teleoperate must stamp _teleop_start_mono"
        assert resolves, (
            f"teleoperate never resolves {_MESH_SLEW_MODULE}, so the loop's lazy import charges "
            f"its cost to the session budget stamped at line {min(stamps)}"
        )
        assert min(resolves) < min(stamps), (
            f"{_MESH_SLEW_MODULE} is resolved at line {min(resolves)}, after the session clock is "
            f"stamped at line {min(stamps)}: the deadline is that stamp plus duration, so the "
            f"resolve is inside the window it bounds"
        )

    def test_the_pre_resolved_module_is_the_one_the_loop_imports(self) -> None:
        """One module name, not two that can drift apart.

        Pre-resolving a *different* module would leave the loop paying the cost
        while looking correct.
        """
        loop_source = textwrap.dedent(inspect.getsource(teleop_mixin.TeleopMixin._teleop_loop))
        loop_imports = {
            node.module
            for node in ast.walk(ast.parse(loop_source))
            if isinstance(node, ast.ImportFrom) and node.module and not node.level
        }
        resolved = {mod for _line, mod in _import_module_calls(self._teleoperate_tree())}

        assert _MESH_SLEW_MODULE in loop_imports, (
            f"premise: the loop must still import {_MESH_SLEW_MODULE} lazily (module scope would "
            f"invert the layering); found {sorted(loop_imports)}"
        )
        assert _MESH_SLEW_MODULE in resolved, (
            f"teleoperate pre-resolves {sorted(resolved)}, which does not include the "
            f"{_MESH_SLEW_MODULE} the loop imports"
        )


class TestTheBoundariesDoNotMove:
    """Two things the fix must leave exactly as they were."""

    def test_a_warm_session_is_unchanged(self) -> None:
        """The common case - anything after the first session in a process."""
        import strands_robots.mesh.security  # noqa: F401  (warm it deliberately)

        host, dev = _host_with_leader()
        res = host.teleoperate(hz=50.0, duration=_SESSION_S, block=True)

        assert res["status"] == "success"
        assert dev.get_action_calls > 0
        elapsed = res["content"][1]["json"]["elapsed_s"]
        assert elapsed == pytest.approx(_SESSION_S, abs=0.25), (
            f"a warm session reported {elapsed:.3f}s for a {_SESSION_S}s budget"
        )

    def test_a_refused_duration_still_resolves_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The knob guards run before setup, so a refusal must not pay for it.

        ``teleoperate`` validates ``hz``/``duration`` before it connects a
        device; the resolve belongs with setup, on the far side of that guard.
        """
        host, dev = _host_with_leader()
        finder = _charge_a_cold_resolve(monkeypatch)

        res = host.teleoperate(hz=50.0, duration=0.0, block=True)

        assert res["status"] == "error"
        assert "duration must be > 0" in res["content"][0]["text"]
        assert dev.connect_calls == 0, "a refused session connected a device"
        assert finder.calls == 0, (
            f"a refused session resolved {_MESH_SLEW_MODULE}: the resolve was hoisted above the "
            f"rate/duration guards, so an unusable knob now costs an import before it is refused"
        )


# --- helpers ----------------------------------------------------------------


def _import_module_calls(tree: ast.Module) -> list[tuple[int, str]]:
    """``(line, module)`` for every ``importlib.import_module("x")`` in ``tree``."""
    calls: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "import_module"):
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            calls.append((node.lineno, first.value))
    return calls


def _self_assignment_lines(tree: ast.Module, attr: str) -> list[int]:
    """Lines assigning ``self.<attr>``."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == attr for t in node.targets)
    ]
