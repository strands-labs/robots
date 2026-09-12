"""``Robot.stream`` lets ``execute``/``start`` reach a provider that builds without a port.

The tool's own schema offers ``policy_provider`` values ``mock`` and
``lerobot_local``, whose registry entries neither require nor read a
``policy_port``. The ``execute``/``start`` guard nonetheless refused any call
without one, and the dispatcher then refused any call *with* one for those
providers (``_policy_port_error``: "declares no policy_port"), so no tool call
could run them on a real arm. The guard now judges ``instruction`` alone and
leaves the port to the provider-aware dispatcher.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

import pytest
from strands.types.tools import ToolUse

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from tests._daemon_executor import DaemonThreadExecutor


def _make_robot() -> HwRobot:
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "test_arm"
    hw.action_horizon = 8
    hw.data_config = None
    hw.control_frequency = 30.0
    hw.action_sleep_time = 1.0 / 30.0
    hw._task_state = RobotTaskState()
    hw._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="test_arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = object()
    return hw


@pytest.fixture
def dispatched(monkeypatch: pytest.MonkeyPatch) -> tuple[HwRobot, list[tuple[str, Any]]]:
    hw = _make_robot()
    calls: list[tuple[str, Any]] = []

    def _execute(instruction: str, port: Any, host: str, provider: str, duration: float) -> dict[str, Any]:
        calls.append(("execute", port))
        return {"status": "success", "content": [{"text": "done"}]}

    def _start(instruction: str, port: Any, host: str, provider: str, duration: float) -> dict[str, Any]:
        calls.append(("start", port))
        return {"status": "success", "content": [{"text": "started"}]}

    hw._execute_task_sync = _execute  # type: ignore[assignment]
    hw.start_task = _start  # type: ignore[assignment]
    monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
    return hw, calls


def _stream(hw: HwRobot, **inp: Any) -> dict[str, Any]:
    tool_use = cast(ToolUse, {"toolUseId": "tu", "input": inp})

    async def _run() -> list:
        return [ev async for ev in hw.stream(tool_use, {})]

    return _run_events(_run)[-1].tool_result


def _run_events(coro_fn):  # type: ignore[no-untyped-def]
    return asyncio.run(coro_fn())


@pytest.mark.parametrize("action", ["execute", "start"])
@pytest.mark.parametrize("provider", ["mock", "lerobot_local"])
def test_a_port_less_provider_reaches_the_dispatcher_without_a_port(dispatched, action: str, provider: str) -> None:
    hw, calls = dispatched
    result = _stream(hw, action=action, instruction="pick", policy_provider=provider)

    assert result["status"] == "success", result
    assert calls == [(action, None)]


@pytest.mark.parametrize("action", ["execute", "start"])
def test_a_missing_instruction_is_still_refused(dispatched, action: str) -> None:
    hw, calls = dispatched
    result = _stream(hw, action=action, policy_provider="mock")

    assert result["status"] == "error"
    assert "instruction" in result["content"][0]["text"]
    assert calls == []


@pytest.mark.parametrize("action", ["execute", "start"])
def test_the_real_dispatcher_still_refuses_groot_without_a_port(action: str) -> None:
    """The judgement moved, it did not vanish: ``groot`` requires a port and says so."""
    hw = _make_robot()
    method = hw._execute_task_sync if action == "execute" else hw.start_task
    result = method("pick", policy_port=None, policy_provider="groot", duration=0.05)

    assert result["status"] == "error"
    assert "policy_port is required" in result["content"][0]["text"]
