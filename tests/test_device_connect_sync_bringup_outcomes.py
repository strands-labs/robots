"""What ``init_device_connect_sync`` tells its caller about a bring-up.

The wrapper starts :func:`strands_robots.device_connect.init_device_connect` on a
daemon thread and bounds the wait, so the bring-up has exactly three outcomes:
it completes, it fails, or the budget expires. All three have to be
distinguishable by the caller -- the return type is a ``DeviceRuntime``, not an
optional one, and ``strands_robots.robot``'s foreground runner wraps the call in
``except Exception`` precisely so a failed bring-up is reported rather than
printing an "is online" line for a runtime that never came up.

The two failure outcomes cross a thread boundary, which is why they are pinned
here: the recorded exception is re-raised on the caller's thread, and an expired
budget raises rather than handing back the ``None`` the holder still contains.

Nothing here needs a broker, a Docker stack or the Kit runtime: the awaited half
is substituted, and the wrapper's budget is a module attribute so the expiry
arm resolves in milliseconds instead of the shipped 30 seconds.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import pathlib
import types
from typing import Any

import pytest

# Import the integration module lazily inside each helper. Sibling modules in
# this directory substitute ``device_connect_edge`` submodules with mocks at
# import time, so binding it at collection time would fix whichever version
# happened to be installed first for every test in the session.


def _dc() -> Any:
    """The Device Connect integration module."""
    import strands_robots.device_connect as module

    return module


def _robot() -> Any:
    """A stand-in for the object the wrapper adapts; only its name is read."""
    return types.SimpleNamespace(tool_name_str="arm")


async def _completes(*_a: Any, **_k: Any) -> Any:
    """A bring-up that returns a runtime."""
    return types.SimpleNamespace(name="runtime")


async def _fails(*_a: Any, **_k: Any) -> Any:
    """A bring-up that fails the way an unreachable broker does."""
    raise RuntimeError("no broker at tcp://127.0.0.1:7447")


async def _never_finishes(*_a: Any, **_k: Any) -> Any:
    """A bring-up still running when the wrapper's budget expires.

    The thread it runs on is a daemon, which is the shipped design: the runtime
    is meant to outlive the call, so there is nothing for the test to join.
    """
    await asyncio.sleep(300)


def _sync(monkeypatch: pytest.MonkeyPatch, bring_up: Any, *, budget: float = 0.05) -> Any:
    """Call the wrapper with ``bring_up`` substituted and a short budget."""
    module = _dc()
    monkeypatch.setattr(module, "init_device_connect", bring_up)
    monkeypatch.setattr(module, "_INIT_TIMEOUT_S", budget)
    return module.init_device_connect_sync(_robot(), peer_id="arm-1")


class TestEveryBringUpOutcomeReachesTheCaller:
    """The three outcomes, and the two that cross a thread boundary."""

    def test_a_completed_bring_up_returns_the_runtime_it_built(self, monkeypatch):
        """The control: a bring-up that finishes hands back its runtime."""
        runtime = _sync(monkeypatch, _completes)

        assert runtime is not None
        assert runtime.name == "runtime"

    def test_a_completed_bring_up_wires_the_loop_and_thread_it_runs_on(self, monkeypatch):
        """The returned runtime carries the loop and thread serving it, so the
        caller can reach the machinery the wrapper created on its behalf."""
        runtime = _sync(monkeypatch, _completes)

        assert runtime._loop.is_running()
        assert runtime._thread.daemon is True

    def test_a_failed_bring_up_re_raises_the_exception_it_recorded(self, monkeypatch):
        """A failure on the background thread is carried to the caller's thread
        as the same exception object, so its cause survives the crossing."""
        marker = RuntimeError("etcd refused the registration")

        async def _raise_marker(*_a: Any, **_k: Any) -> Any:
            raise marker

        with pytest.raises(RuntimeError) as excinfo:
            _sync(monkeypatch, _raise_marker)

        assert excinfo.value is marker

    def test_an_expired_budget_raises_rather_than_returning_none(self, monkeypatch):
        """A bring-up still running when the budget expires is a failed
        bring-up: the holder is empty, and returning it would hand the caller
        ``None`` where its own annotation promises a runtime."""
        with pytest.raises(TimeoutError):
            _sync(monkeypatch, _never_finishes)

    def test_an_expired_budget_names_the_budget_and_where_to_look(self, monkeypatch):
        """The refusal has to say how long was waited and what to check, since
        the caller cannot see the thread the bring-up is still running on."""
        with pytest.raises(TimeoutError) as excinfo:
            _sync(monkeypatch, _never_finishes, budget=0.05)

        message = str(excinfo.value)
        assert "init_device_connect_sync" in message
        assert "0.05s" in message
        assert "still running" in message
        assert "broker" in message

    def test_a_failure_inside_the_budget_reports_its_cause_not_the_budget(self, monkeypatch):
        """Guard order: the recorded exception is checked first, so a bring-up
        that failed quickly is never reported as a slow one."""
        with pytest.raises(RuntimeError) as excinfo:
            _sync(monkeypatch, _fails, budget=30.0)

        assert "no broker" in str(excinfo.value)
        assert "did not come up" not in str(excinfo.value)


class TestTheForegroundRunnerReportsAFailedBringUp:
    """``Robot(...).run()`` wraps the wrapper in ``except Exception`` so an
    operator is told when the runtime did not come up. That only works if the
    wrapper raises: a returned ``None`` is indistinguishable from success."""

    def _foreground(self, monkeypatch, bring_up: Any, *, budget: float = 0.05) -> list[str]:
        import strands_robots.robot as robot_module

        module = _dc()
        monkeypatch.setattr(module, "init_device_connect", bring_up)
        monkeypatch.setattr(module, "_INIT_TIMEOUT_S", budget)

        instance = types.SimpleNamespace(_peer_id="arm-1", _peer_type="robot", mesh=None, tool_name_str="arm")
        # The runner sleeps forever and then exits the process; end both.
        monkeypatch.setattr("time.sleep", lambda _s: (_ for _ in ()).throw(KeyboardInterrupt()))
        monkeypatch.setattr(robot_module.os, "_exit", lambda _code: None)

        records: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(f"{record.levelname} {record.getMessage()}")

        handler = _Capture()
        logger = logging.getLogger("strands_robots.robot")
        logger.addHandler(handler)
        try:
            robot_module._run_device_connect_foreground(instance)
        finally:
            logger.removeHandler(handler)
        return records

    def test_an_expired_budget_is_reported_to_the_operator(self, monkeypatch):
        """Without a raise this outcome logged nothing at all, leaving the
        runner's "is online" line as the only thing said about a runtime that
        never came up."""
        records = self._foreground(monkeypatch, _never_finishes)

        failures = [r for r in records if "Device Connect init failed" in r]
        assert failures, records
        assert "did not come up" in failures[0]

    def test_a_failed_bring_up_is_still_reported_to_the_operator(self, monkeypatch):
        """The unchanged half: an exception already reached this warning."""
        records = self._foreground(monkeypatch, _fails, budget=30.0)

        failures = [r for r in records if "Device Connect init failed" in r]
        assert failures, records
        assert "no broker" in failures[0]


class TestTheBudgetIsReadFromTheModuleRatherThanAnInlineLiteral:
    """Premises the tests above rest on, pinned so they cannot quietly stop
    holding: the budget is a module attribute (so an expiry is reachable in a
    unit test at all) and the return type is not optional (so handing back the
    empty holder was a defect rather than a documented outcome)."""

    def test_the_budget_is_a_positive_finite_number(self):
        module = _dc()

        budget = module._INIT_TIMEOUT_S
        assert isinstance(budget, float)
        assert 0.0 < budget < float("inf")

    def test_the_wait_is_bounded_by_the_module_budget(self):
        """The wait must read the module attribute, not an inline number: a
        re-hardcoded budget would make the expiry arm untestable again."""
        module = _dc()
        source = pathlib.Path(module.__file__).read_text()
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "init_device_connect_sync"
        )

        waits = [
            call
            for call in ast.walk(function)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "wait"
        ]
        assert len(waits) == 1, [ast.unparse(w) for w in waits]
        timeouts = [kw.value for kw in waits[0].keywords if kw.arg == "timeout"]
        assert len(timeouts) == 1, ast.unparse(waits[0])
        assert isinstance(timeouts[0], ast.Name), ast.unparse(timeouts[0])
        assert timeouts[0].id == "_INIT_TIMEOUT_S"

    def test_the_wait_result_decides_whether_the_bring_up_finished(self):
        """The bound wait's own answer is the only thing separating a completed
        bring-up from an expired budget, so it must be read rather than dropped."""
        module = _dc()
        source = pathlib.Path(module.__file__).read_text()
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "init_device_connect_sync"
        )

        bound = [
            node.targets[0].id
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "wait"
        ]
        assert len(bound) == 1, bound
        tested = {ast.unparse(node.test) for node in ast.walk(function) if isinstance(node, ast.If)}
        assert f"not {bound[0]}" in tested, tested

    def test_the_wrapper_promises_a_runtime_rather_than_an_optional_one(self):
        module = _dc()

        assert module.init_device_connect_sync.__annotations__["return"] == "DeviceRuntime"
