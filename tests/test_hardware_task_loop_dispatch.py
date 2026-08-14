"""Behavior tests for how a hardware rollout is dispatched onto an event loop.

``Robot._run_control_loop`` has to work from two kinds of caller: a synchronous
one (``run_policy``, ``start_task``'s executor worker) and an asynchronous one.
``Robot.stream(action="execute")`` is an ``async def`` that calls
``_execute_task_sync``, so the "already on a running loop" branch is that
surface's *live* path rather than a hypothetical one.

The branch was chosen by wrapping ``except RuntimeError`` around both the loop
probe *and* the nested dispatch. Only the probe's own ``RuntimeError`` means "no
loop is running"; a ``RuntimeError`` raised by the nested dispatch (a thread the
pool cannot start, an executor already shut down) landed in the same handler,
whose ``asyncio.run`` is invalid by construction on exactly that branch. The
caller was told ``asyncio.run() cannot be called from a running event loop``
instead of the cause, and the ``task_runner`` coroutine the handler built was
left un-awaited.

These tests pin that:

    - an async caller really does take the nested branch, and a sync caller
      really does not, so neither assertion below is vacuous;
    - a dispatch failure propagates *its own* cause, with the asyncio internal
      absent from the message;
    - that failure leaks no un-awaited coroutine;
    - the rollout is driven exactly once on either branch, and not at all when
      the dispatch fails, so a failed dispatch is never retried on a thread that
      cannot serve it;
    - the ``except RuntimeError`` guards the probe alone.

No serial/USB hardware is touched: the driver is an in-memory fake, the connect
path is a stub, and the policy is the built-in ``mock`` provider.
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import gc
import inspect
import threading
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from tests.test_hardware_control_loop_rate_guard import _FakeArm

# duration=0.2 at 50 Hz drives ten applied actions -- long enough that "the
# rollout ran" is unambiguous, short enough to stay a unit test.
_DURATION = 0.2
_EXPECTED_ACTIONS = 10


@pytest.fixture
def hw() -> Iterator[Any]:
    """A ``Robot`` wired to an in-memory arm and a stubbed connect path."""
    robot = HwRobot.__new__(HwRobot)
    robot.tool_name_str = "test_arm"
    robot.action_horizon = 1
    robot.data_config = None
    robot.control_frequency = 50.0
    robot.action_sleep_time = 1.0 / 50.0
    robot._task_state = RobotTaskState()
    robot._executor = ThreadPoolExecutor(max_workers=1)
    robot._shutdown_event = threading.Event()
    robot._stop_requested = threading.Event()
    robot._task_admission = threading.Lock()
    robot._task_claimed = False
    robot.mesh = None
    robot.peer_id = None
    robot.robot = _FakeArm()

    async def _connected() -> tuple[bool, str]:
        return (True, "")

    async def _ready() -> bool:
        return True

    def _init_policy(policy: Any) -> Any:
        return _ready()

    def _no_telemetry(observation: dict[str, Any], *, skip_images: bool = False) -> None:
        return None

    robot._connect_robot = _connected  # type: ignore[method-assign]
    robot._initialize_policy = _init_policy  # type: ignore[method-assign]
    robot._publish_ros_telemetry = _no_telemetry  # type: ignore[method-assign]
    try:
        yield robot
    finally:
        robot._shutdown_event.set()
        robot._task_state.status = TaskStatus.STOPPED
        robot._executor.shutdown(wait=False)


def _drive(robot: Any) -> dict[str, Any]:
    """Run one rollout through the real dispatch, as ``stream`` would."""
    return robot._run_control_loop("pick", 5555, "localhost", "mock", _DURATION)


def _pool_spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record every *nested-dispatch* pool construction.

    Only bare ``ThreadPoolExecutor()`` calls are counted. The rollout itself
    reaches the arm through ``asyncio.to_thread``, which builds the loop's own
    default executor with ``thread_name_prefix=...``; counting that too would
    make "the nested branch was taken" true on both branches.
    """
    built: list[int] = []
    real = concurrent.futures.ThreadPoolExecutor

    class Recording(real):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if not args and not kwargs:
                built.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", Recording)
    return built


def _refusing_pool(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    """Make the nested dispatch's pool construction fail with ``message``."""

    class Refusing:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(message)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", Refusing)


class TestWhichBranchEachCallerTakes:
    """Neither branch assertion below is vacuous."""

    def test_an_async_caller_takes_the_nested_branch(self, hw: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        built = _pool_spy(monkeypatch)

        async def caller() -> dict[str, Any]:
            return _drive(hw)

        result = asyncio.run(caller())
        assert built, "an async caller must reach the nested dispatch"
        assert result["status"] == "success"

    def test_a_sync_caller_does_not(self, hw: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        built = _pool_spy(monkeypatch)
        result = _drive(hw)
        assert built == [], "a sync caller must run the rollout on this thread"
        assert result["status"] == "success"

    def test_stream_reaches_the_dispatch_from_a_running_loop(self) -> None:
        """``stream`` is the async surface that makes the nested branch live."""
        assert inspect.isasyncgenfunction(HwRobot.stream)
        assert "_execute_task_sync" in (inspect.getsource(HwRobot.stream) or "")


class TestADispatchFailureReportsItsOwnCause:
    """The handler must not answer a failure it cannot serve."""

    def test_the_cause_survives(self, hw: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        _refusing_pool(monkeypatch, "can't start new thread")

        async def caller() -> dict[str, Any]:
            return _drive(hw)

        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(caller())
        message = str(excinfo.value)
        assert "can't start new thread" in message
        assert "cannot be called from a running event loop" not in message

    def test_the_arm_is_not_commanded(self, hw: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed dispatch is not retried on a thread that cannot serve it."""
        _refusing_pool(monkeypatch, "can't start new thread")

        async def caller() -> dict[str, Any]:
            return _drive(hw)

        with pytest.raises(RuntimeError):
            asyncio.run(caller())
        assert hw.robot.sent_actions == []

    def test_no_coroutine_is_left_un_awaited(self, hw: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        _refusing_pool(monkeypatch, "can't start new thread")

        async def caller() -> dict[str, Any]:
            return _drive(hw)

        raised = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                asyncio.run(caller())
            except RuntimeError:
                # Deliberately keep no reference to the exception: its traceback
                # holds the frame that holds the coroutine, which would postpone
                # the finalizer (and its warning) past this recording block.
                raised = True
            gc.collect()
        assert raised, "the dispatch failure must still propagate"
        leaked = [w for w in caught if "was never awaited" in str(w.message)]
        assert leaked == [], f"the failed dispatch leaked {len(leaked)} un-awaited coroutine(s)"


class TestTheRolloutIsDrivenExactlyOnce:
    """Neither branch may drive the rollout twice."""

    def test_on_the_sync_branch(self, hw: Any) -> None:
        result = _drive(hw)
        assert result["status"] == "success"
        assert len(hw.robot.sent_actions) == _EXPECTED_ACTIONS

    def test_on_the_nested_branch(self, hw: Any) -> None:
        async def caller() -> dict[str, Any]:
            return _drive(hw)

        result = asyncio.run(caller())
        assert result["status"] == "success"
        assert len(hw.robot.sent_actions) == _EXPECTED_ACTIONS


class TestTheSyncBranchStillPropagates:
    """A failure of the sync branch keeps its own cause too."""

    def test_a_runtime_error_from_the_run_is_not_swallowed(self, hw: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        def refusing_run(coro: Any, **kwargs: Any) -> Any:
            coro.close()
            raise RuntimeError("Event loop is closed")

        monkeypatch.setattr(asyncio, "run", refusing_run)
        with pytest.raises(RuntimeError, match="Event loop is closed"):
            _drive(hw)
        assert hw.robot.sent_actions == []


class TestTheGuardWrapsTheProbeAlone:
    """Structural: only the loop probe may sit inside the RuntimeError guard."""

    def test_the_guarded_body_is_the_probe(self) -> None:
        fn = ast.parse(inspect.getsource(HwRobot._run_control_loop).lstrip()).body[0]
        assert isinstance(fn, ast.FunctionDef)
        guards = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Try)
            and any(
                isinstance(handler.type, ast.Name) and handler.type.id == "RuntimeError" for handler in node.handlers
            )
        ]
        assert len(guards) == 1, f"expected one RuntimeError guard, found {len(guards)}"
        guard = guards[0]
        assert len(guard.body) == 1, "the guard must cover the probe alone"
        probe = ast.unparse(guard.body[0])
        assert "get_running_loop" in probe
        assert "asyncio.run" not in probe

    def test_the_handler_does_not_run_the_rollout(self) -> None:
        fn = ast.parse(inspect.getsource(HwRobot._run_control_loop).lstrip()).body[0]
        assert isinstance(fn, ast.FunctionDef)
        guard = next(
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Try)
            and any(
                isinstance(handler.type, ast.Name) and handler.type.id == "RuntimeError" for handler in node.handlers
            )
        )
        handler = "\n".join(ast.unparse(stmt) for stmt in guard.handlers[0].body)
        assert "task_runner" not in handler
        assert "asyncio.run" not in handler
