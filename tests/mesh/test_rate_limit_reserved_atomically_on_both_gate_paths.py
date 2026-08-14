"""The per-action rate limit must be reserved atomically on *both* gate paths.

``robot_mesh`` bounds LLM-driven actuation with a per-action sliding window
(``emergency_stop`` at 3/60s, ``broadcast`` at 10/60s, ...). A slot is
inspected early by :func:`~strands_robots.tools.robot_mesh._rate_limit_check`,
which deliberately does **not** consume one so a declined human-in-the-loop
approval cannot lock the agent out of a genuine emergency. Everything between
that check and the point a slot is actually taken is therefore a window in
which a concurrent invocation can claim the last slot, and the reservation has
to notice.

Two gate paths reach that point:

* the **approved** path, after ``tool_context.interrupt`` returns an
  affirmative response, and
* the **ungated** path, for an action the operator has taken out of
  ``STRANDS_MESH_HITL_ACTIONS``.

Both reserve through
:func:`~strands_robots.tools.robot_mesh._rate_limit_check_and_record`, which
re-checks and appends under a single ``_RATE_LOCK`` acquisition. The ungated
path matters most: once the interrupt gate is narrowed the rate limit is the
only bound left on LLM-driven actuation, so a call that races past it reaches
the mesh with no operator and no limit in the way.

``tests/mesh/test_robot_mesh_security.py`` pins the helpers directly. This
module pins the *tool*: that each gate path refuses the raced call, reports and
audits it, dispatches nothing, and leaves the window holding no more than the
configured maximum.
"""

from __future__ import annotations

import ast
import inspect
import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots.tools.robot_mesh as rmt

ACTION = "emergency_stop"
LIMIT = 3

#: The two gate paths, as (label, ``STRANDS_MESH_HITL_ACTIONS`` value).
GATE_PATHS = [
    pytest.param(ACTION, id="approved"),
    pytest.param("none", id="ungated"),
]


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Iterator[None]:
    """Isolate the sliding window, the audit log and the interrupt config."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    monkeypatch.setitem(rmt._RATE_LIMITS, ACTION, (LIMIT, 60.0))
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()
    yield
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()


def _tool() -> Any:
    """The undecorated tool function (the ``@tool`` wrapper needs an agent)."""
    return getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh


def _ctx(response: str = "y") -> MagicMock:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


def _mesh() -> MagicMock:
    m = MagicMock(name="Mesh")
    m.emergency_stop.return_value = [{"status": "ok"}]
    return m


def _fill(n: int) -> None:
    """Reserve *n* slots so the window has ``LIMIT - n`` left."""
    for _ in range(n):
        assert rmt._rate_limit_check_and_record(ACTION) is None


def _call(hitl: str, ctx: MagicMock, mesh: MagicMock, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", hitl)
    rmt._reset_interrupt_actions_cache()
    with patch.object(rmt, "_resolve_mesh", return_value=mesh):
        result: dict[str, Any] = _tool()(action=ACTION, tool_context=ctx)
    return result


def _steal_slot_in_the_window(monkeypatch: pytest.MonkeyPatch, slots: int = 1) -> None:
    """Take *slots* between the pre-gate check and this call's reservation.

    ``_numeric_option_error`` is the first thing ``robot_mesh`` calls after the
    pre-gate rate-limit check, so consuming there models a concurrent
    invocation that claimed the last slot while this one was still working -
    deterministically, with no threads.
    """
    real = rmt._numeric_option_error

    def gated(*args: Any, **kwargs: Any) -> Any:
        for _ in range(slots):
            rmt._rate_limit_check_and_record(ACTION)
        return real(*args, **kwargs)

    monkeypatch.setattr(rmt, "_numeric_option_error", gated)


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


class TestEachGatePathRefusesACallThatRacedPastTheCheck:
    """The window between the check and the reservation is not a hole."""

    @pytest.mark.parametrize("hitl", GATE_PATHS)
    def test_a_raced_call_is_refused_and_dispatches_nothing(self, hitl: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _fill(LIMIT - 1)  # one slot left when the pre-gate check runs
        _steal_slot_in_the_window(monkeypatch)  # ... and it is gone by the reservation
        mesh = _mesh()

        result = _call(hitl, _ctx(), mesh, monkeypatch)

        assert result["status"] == "error", result
        assert "rate limit exceeded" in _text(result)
        mesh.emergency_stop.assert_not_called()

    @pytest.mark.parametrize("hitl", GATE_PATHS)
    def test_a_raced_call_leaves_the_window_at_the_configured_maximum(
        self, hitl: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fill(LIMIT - 1)
        _steal_slot_in_the_window(monkeypatch)

        _call(hitl, _ctx(), _mesh(), monkeypatch)

        # The stolen slot filled the window; the refused call added nothing.
        assert len(rmt._RATE_HISTORY[ACTION]) == LIMIT

    @pytest.mark.parametrize("hitl", GATE_PATHS)
    def test_two_concurrent_calls_yield_exactly_one_dispatch(self, hitl: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real scenario: both pass the check, only one may reserve."""
        _fill(LIMIT - 1)
        monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", hitl)
        rmt._reset_interrupt_actions_cache()

        arrived = threading.Barrier(3, timeout=30)
        real = rmt._numeric_option_error

        def gated(*args: Any, **kwargs: Any) -> Any:
            arrived.wait()  # hold both workers between check and reservation
            return real(*args, **kwargs)

        mesh = _mesh()
        statuses: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            r = _tool()(action=ACTION, tool_context=_ctx())
            with lock:
                statuses.append(r["status"])

        # Patch on this thread only: ``unittest.mock`` is not thread-safe, so a
        # per-worker context manager would race on the same attribute.
        with (
            patch.object(rmt, "_resolve_mesh", return_value=mesh),
            patch.object(rmt, "_numeric_option_error", gated),
        ):
            threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
            for t in threads:
                t.start()
            arrived.wait()
            for t in threads:
                t.join(timeout=30)
                assert not t.is_alive(), "worker did not finish"

        assert sorted(statuses) == ["error", "success"], statuses
        assert mesh.emergency_stop.call_count == 1
        assert len(rmt._RATE_HISTORY[ACTION]) == LIMIT


class TestTheRefusalTellsTheCallerWhatHappened:
    @pytest.mark.parametrize("hitl", GATE_PATHS)
    def test_the_refusal_names_the_action_the_limit_and_a_retry_hint(
        self, hitl: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fill(LIMIT - 1)
        _steal_slot_in_the_window(monkeypatch)

        result = _call(hitl, _ctx(), _mesh(), monkeypatch)

        assert result["status"] == "error", result
        text = _text(result)
        assert ACTION in text
        assert f"max {LIMIT} calls" in text
        assert "raced past" in text
        assert "Try again in" in text

    def test_the_refusal_does_not_claim_an_operator_approval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ungated path has no approval, so the reason must not name one."""
        _fill(LIMIT - 1)
        _steal_slot_in_the_window(monkeypatch)

        result = _call("none", _ctx(), _mesh(), monkeypatch)

        # Guard against passing on a *success* text, which names no approval
        # either: the wording only matters once the call is actually refused.
        assert result["status"] == "error", result
        assert "raced past" in _text(result)
        assert "approval" not in _text(result).lower()

    @pytest.mark.parametrize("hitl", GATE_PATHS)
    def test_the_raced_call_is_audited_as_a_failure(self, hitl: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _fill(LIMIT - 1)
        _steal_slot_in_the_window(monkeypatch)
        rows: list[tuple[Any, ...]] = []
        monkeypatch.setattr(
            rmt,
            "_audit_tool_action",
            lambda action, target, success, detail: rows.append((action, target, success, detail)),
        )

        _call(hitl, _ctx(), _mesh(), monkeypatch)

        raced = [r for r in rows if "rate_limit_race" in str(r[3])]
        assert len(raced) == 1, rows
        assert raced[0][0] == ACTION
        assert raced[0][2] is False


class TestReservingIsTheOnlyWayASlotIsConsumed:
    """Structural: no gate path may append to the window without re-checking."""

    @staticmethod
    def _reservations(source: str) -> list[ast.Assign]:
        """Assignments in ``robot_mesh`` whose value is the atomic reservation."""
        fn = next(
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "robot_mesh"
        )
        return [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_rate_limit_check_and_record"
        ]

    @staticmethod
    def _module_source() -> str:
        return inspect.getsource(rmt)

    def test_both_gate_paths_reserve_through_the_atomic_helper(self) -> None:
        assert len(self._reservations(self._module_source())) == 2

    def test_no_helper_appends_a_slot_without_re_checking(self) -> None:
        """A record-only primitive is exactly the racy pattern removed here."""
        source = self._module_source()
        appenders = [
            fn.name
            for fn in ast.walk(ast.parse(source))
            if isinstance(fn, ast.FunctionDef)
            and fn.name.startswith("_rate_limit")
            and fn.name != "_rate_limit_check_and_record"
            and "bucket.append(" in (ast.get_source_segment(source, fn) or "")
        ]
        assert appenders == [], f"these append without re-checking: {appenders}"

    def test_every_reservation_returns_its_verdict_to_the_caller(self) -> None:
        """A discarded verdict is a reservation that refuses nothing."""
        source = self._module_source()
        for assign in self._reservations(source):
            target = assign.targets[0]
            assert isinstance(target, ast.Name)
            segment = ast.get_source_segment(source, assign) or ""
            assert target.id in segment
        # Each reservation is followed by a guard that returns an error.
        text = source
        assert text.count("if rl_race_err is not None:") == 2
        assert text.count("return _err(rl_race_err)") == 2

    def test_the_scan_detects_a_record_only_helper(self) -> None:
        """Meta: the appender scan is not vacuous."""
        planted = self._module_source() + (
            "\n\ndef _rate_limit_record_planted(action):\n"
            "    bucket = _RATE_HISTORY.setdefault(action, collections.deque())\n"
            "    bucket.append(0.0)\n"
        )
        appenders = [
            fn.name
            for fn in ast.walk(ast.parse(planted))
            if isinstance(fn, ast.FunctionDef)
            and fn.name.startswith("_rate_limit")
            and fn.name != "_rate_limit_check_and_record"
            and "bucket.append(" in (ast.get_source_segment(planted, fn) or "")
        ]
        assert "_rate_limit_record_planted" in appenders


class TestTheSurroundingSafetyPropertiesStillHold:
    """Over-reach controls: the reservation must not refuse ordinary calls."""

    @pytest.mark.parametrize("hitl", GATE_PATHS)
    def test_a_call_with_a_slot_free_still_dispatches(self, hitl: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _fill(LIMIT - 1)  # one slot left, and nobody steals it
        mesh = _mesh()

        result = _call(hitl, _ctx(), mesh, monkeypatch)

        assert result["status"] == "success", result
        mesh.emergency_stop.assert_called_once()
        assert len(rmt._RATE_HISTORY[ACTION]) == LIMIT

    def test_the_ungated_path_really_skips_the_interrupt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-vacuity: the ungated cases above are not silently gated."""
        ctx = _ctx()

        result = _call("none", ctx, _mesh(), monkeypatch)

        assert result["status"] == "success", result
        ctx.interrupt.assert_not_called()

    def test_a_declined_approval_still_consumes_no_slot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reserving after approval must not resurrect the decline charge."""
        result = _call(ACTION, _ctx("n"), _mesh(), monkeypatch)

        assert result["status"] == "error"
        assert "declined" in _text(result).lower()
        assert len(rmt._RATE_HISTORY.get(ACTION, ())) == 0

    def test_a_full_window_is_refused_before_the_operator_is_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pre-gate check still spends no operator attention."""
        _fill(LIMIT)
        ctx = _ctx()

        result = _call(ACTION, ctx, _mesh(), monkeypatch)

        assert result["status"] == "error"
        assert "rate limit exceeded" in _text(result)
        assert "raced past" not in _text(result)
        ctx.interrupt.assert_not_called()
