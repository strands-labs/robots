"""Behavior tests for the accepted ``duration`` domain of a hardware task.

``strands_robots.hardware_robot.Robot`` bounds every rollout by comparing
elapsed time against ``duration`` (``time.monotonic() - start_mono <
duration``). On hardware that comparison is ANDed with the optional ``n_steps``
cap rather than superseded by it, so ``duration`` is the effective horizon of
every task. These tests pin that a budget the loop cannot honor is refused at
the public entry point instead of being spent on the arm:

    - ``0`` / negative / ``nan`` makes the loop condition false on its first
      evaluation, so the task used to report ``status="success"`` for a rollout
      that never queried the policy and never commanded a servo;
    - ``inf`` never makes it false, so the loop commanded the bus indefinitely
      and the blocking entry point never returned;
    - a non-numeric budget reached the comparison intact and surfaced a bare
      ``TypeError`` naming a comparison internal rather than the parameter;
    - the refusal happens before the arm is commanded and, for ``start_task``,
      before the work is submitted, so a budget that cannot be honored is never
      reported back as a started task;
    - ``_execute_task_sync`` refuses too: it is the chokepoint the agent-tool
      ``execute`` action and the mesh ``execute`` dispatch reach directly, so a
      peer-supplied budget is bounded by the same rule;
    - every budget that IS accepted still runs, including a fractional and a
      NumPy-scalar one;
    - the accepted domain matches the simulation's rollout budget
      (``SimEngine._validate_duration``), so the same value cannot be refused
      for a digital twin and accepted for the arm it mirrors.

No serial/USB hardware is touched: the driver is an in-memory fake and the
policy is a structural stub.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from strands_robots.simulation.base import SimEngine
from tests.test_hardware_control_loop_rate_guard import _FakeArm

# Budgets the loop cannot honor. ``0`` / negative / ``nan`` make the loop
# condition false immediately (a task that commands nothing); ``inf`` never
# makes it false (a task that never ends); the rest are not numbers the
# elapsed-time comparison can be made against at all.
UNUSABLE_DURATIONS: list[Any] = [
    0,
    0.0,
    -1.0,
    -30,
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    "5",
    None,
    [1.0],
]

# Budgets that are honorable: any positive finite real, fractional or NumPy.
USABLE_DURATIONS: list[Any] = [0.25, 2, 30.0, np.float32(0.3), np.int64(1)]


class _Policy:
    """Structural stand-in for a policy: the members the loop reads."""

    supports_rtc = False
    execution_horizon = 1

    def set_control_frequency(self, hz: float) -> None:
        return None

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        return None

    async def get_actions(self, observation: Any, instruction: str) -> list[dict[str, Any]]:
        return [{"j0.pos": 0.1}]


@pytest.fixture
def hw() -> Any:
    """A ``Robot`` wired to an in-memory arm, with the connect path stubbed.

    Yields the instance. ``policy_inits`` records every policy initialization
    so a test can assert a refused budget never got that far, and
    ``robot.sent_actions`` records every command that reached the arm.
    """
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
    robot.policy_inits = []  # type: ignore[attr-defined]

    async def _connected() -> tuple[bool, str]:
        return (True, "")

    async def _ready() -> bool:
        return True

    def _init_policy(policy: Any) -> Any:
        robot.policy_inits.append(policy)  # type: ignore[attr-defined]
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


def _text(result: dict[str, Any]) -> str:
    """The text of a tool-shaped result."""
    return " ".join(block["text"] for block in result["content"] if "text" in block)


class TestUnusableBudgetRefused:
    """Every public entry point refuses a budget the loop cannot honor."""

    @pytest.mark.parametrize("duration", UNUSABLE_DURATIONS)
    def test_run_policy_refuses(self, hw: Any, duration: Any):
        """``run_policy`` errors naming the parameter, not a comparison."""
        result = hw.run_policy(policy_object=_Policy(), instruction="probe", duration=duration)

        assert result["status"] == "error"
        assert "duration" in _text(result)
        assert "run_policy" in _text(result)

    @pytest.mark.parametrize("duration", UNUSABLE_DURATIONS)
    def test_start_task_refuses(self, hw: Any, duration: Any):
        """``start_task`` errors instead of reporting a started task."""
        result = hw.start_task("probe", policy_port=9000, duration=duration)

        assert result["status"] == "error"
        assert "duration" in _text(result)
        assert "Task started" not in _text(result)

    @pytest.mark.parametrize("duration", UNUSABLE_DURATIONS)
    def test_the_shared_chokepoint_refuses(self, hw: Any, duration: Any):
        """``_execute_task_sync`` refuses on its own.

        The agent-tool ``execute`` action and the mesh ``execute`` dispatch call
        it directly rather than through ``run_policy``, so a peer-supplied
        budget must be bounded here too.
        """
        result = hw._execute_task_sync("probe", policy_port=9000, duration=duration)

        assert result["status"] == "error"
        assert "duration" in _text(result)


class TestRefusalPrecedesTheArm:
    """A refused budget is never spent on hardware."""

    @pytest.mark.parametrize("duration", [0, -1.0, float("nan"), float("inf"), "5"])
    def test_no_action_reaches_the_arm(self, hw: Any, duration: Any):
        """No command is written and no policy is initialized."""
        hw.run_policy(policy_object=_Policy(), instruction="probe", duration=duration)

        assert hw.robot.sent_actions == []
        assert hw.policy_inits == []

    @pytest.mark.parametrize("duration", [0, -1.0, float("nan"), float("inf")])
    def test_start_task_does_not_submit_the_work(self, hw: Any, duration: Any):
        """The refusal precedes ``executor.submit``.

        A budget checked on the background thread would still have reported
        "Task started" to the caller, so the guard has to run before the
        submit - which is observable as no future ever being created.
        """
        hw.start_task("probe", policy_port=9000, duration=duration)

        assert hw._task_state.task_future is None
        assert hw._task_state.status == TaskStatus.IDLE


class TestAnEndlessBudgetIsRefused:
    """``inf`` is a runaway, not a long task."""

    def test_run_policy_returns_instead_of_commanding_forever(self, hw: Any):
        """The blocking call returns; the loop is never entered.

        ``inf`` never makes ``time.monotonic() - start_mono < duration`` false, so
        the loop kept commanding the servo bus with no bound and the caller
        never got control back. Driven from a worker thread so the assertion
        fails as a timeout rather than hanging the suite.
        """
        outcome: list[dict[str, Any]] = []

        def drive() -> None:
            outcome.append(hw.run_policy(policy_object=_Policy(), instruction="probe", duration=float("inf")))

        worker = threading.Thread(target=drive, daemon=True)
        worker.start()
        worker.join(timeout=5.0)

        assert not worker.is_alive(), "run_policy(duration=inf) never returned"
        assert outcome[0]["status"] == "error"
        assert hw.robot.sent_actions == []


class TestUsableBudgetHonored:
    """Every budget the loop can honor still runs."""

    @pytest.mark.parametrize("duration", USABLE_DURATIONS)
    def test_the_rollout_runs_and_commands_the_arm(self, hw: Any, duration: Any):
        """A positive finite budget reaches the loop and moves the arm."""
        result = hw.run_policy(policy_object=_Policy(), instruction="probe", duration=duration, n_steps=3)

        assert result["status"] == "success"
        assert len(hw.robot.sent_actions) == 3


class TestBudgetBoundsTheRolloutEvenWithAStepCap:
    """On hardware ``duration`` is effective alongside ``n_steps``.

    The simulation recomputes ``duration`` from ``n_steps`` and validates it
    only when no step count was given. The hardware loop ANDs the two
    conditions instead, so an unusable ``duration`` silences a rollout that
    asked for a specific number of steps - and must be refused even then.
    """

    @pytest.mark.parametrize("duration", [0, -1.0, float("nan")])
    def test_a_step_cap_does_not_excuse_the_budget(self, hw: Any, duration: Any):
        """A step cap does not make an unusable budget acceptable."""
        result = hw.run_policy(policy_object=_Policy(), instruction="probe", duration=duration, n_steps=10)

        assert result["status"] == "error"
        assert "duration" in _text(result)
        assert hw.robot.sent_actions == []

    def test_the_step_cap_still_wins_when_both_are_usable(self, hw: Any):
        """A usable budget generous enough for the cap applies the cap."""
        result = hw.run_policy(policy_object=_Policy(), instruction="probe", duration=30.0, n_steps=3)

        assert result["status"] == "success"
        assert len(hw.robot.sent_actions) == 3


class TestDomainMatchesSimulation:
    """The arm and its digital twin accept the same rollout budgets.

    ``SimEngine._validate_duration`` and the hardware guard both delegate to
    ``positive_finite_number_error``; this pins that they cannot diverge, so a
    budget rehearsed in sim is honored on the arm and vice versa.
    """

    @pytest.mark.parametrize("duration", UNUSABLE_DURATIONS + USABLE_DURATIONS)
    def test_both_layers_agree(self, hw: Any, duration: Any):
        """Refused by one layer if and only if refused by the other."""
        sim_refuses = SimEngine._validate_duration(duration, "run_policy") is not None
        hw_refuses = hw._duration_error(duration, "run_policy") is not None

        assert hw_refuses == sim_refuses, f"verdicts differ for duration={duration!r}"

    def test_the_message_is_the_shared_one(self, hw: Any):
        """Both layers name the parameter identically."""
        sim_error = SimEngine._validate_duration(0, "run_policy")
        hw_error = hw._duration_error(0, "run_policy")

        assert sim_error is not None and hw_error is not None
        assert _text(hw_error) == _text(sim_error)


def test_the_guard_does_not_need_an_event_loop(hw: Any):
    """The refusal is synchronous: no event loop is started to produce it."""
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    assert hw._duration_error(0, "run_policy") is not None
