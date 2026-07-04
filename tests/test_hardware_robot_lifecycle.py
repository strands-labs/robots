"""Behavior tests for the hardware Robot control loop and teleop hot path.

These cover ``strands_robots.hardware_robot.Robot`` without any serial/USB
hardware: a lightweight in-memory fake stands in for the lerobot robot, and
the policy is a deterministic stub. The focus is the *hot path* an autonomous
operator drives -- task lifecycle (start/status/stop), the async execution
loop (connect -> policy -> observe -> act), the tool ``stream`` dispatch, and
the mesh teleop publish/receive lifecycle.

A ``Robot`` is built via ``__new__`` + manual attribute wiring (the same
pattern the existing ``test_robot_factory`` helper tests use) so construction
never touches ``_initialize_robot``/lerobot hardware drivers.
"""

from __future__ import annotations

import asyncio
import pkgutil
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from strands_robots.policies.base import Policy
from tests.tool_result_contract import tool_json


class _FakeLeRobot:
    """Minimal stand-in for a connected lerobot Robot.

    Records sent actions and serves a fixed observation. ``is_connected``
    flips to True on ``connect()`` so the connect path can be exercised
    without serial traffic.
    """

    def __init__(self, *, connected: bool = False, calibrated: bool = True) -> None:
        self.name = "fake_arm"
        self.robot_type = "fake_arm"
        self._connected = connected
        self.is_calibrated = calibrated
        self.sent_actions: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"cameras": {}})()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = False) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> dict[str, Any]:
        return {"j0.pos": 0.0, "j1.pos": 0.0}

    def send_action(self, action: dict[str, Any]) -> None:
        self.sent_actions.append(action)


class _StubPolicy:
    """Deterministic policy: returns a fixed action chunk each step.

    Records the Real-Time Chunking contract calls (``set_control_frequency`` /
    ``set_rtc_observed_delay``) the control loop now issues, mirroring the
    no-op base ``Policy`` hooks real providers inherit.
    """

    def __init__(self) -> None:
        self.state_keys: list[str] | None = None
        self.control_frequency_calls: list[float] = []
        self.observed_delays: list[int | None] = []

    def set_robot_state_keys(self, keys: list[str]) -> None:
        self.state_keys = keys

    def set_control_frequency(self, hz: float) -> None:
        self.control_frequency_calls.append(hz)

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        self.observed_delays.append(steps)

    async def get_actions(self, observation: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
        return [{"j0.pos": 0.1}, {"j1.pos": 0.2}]


def _make_robot(fake: _FakeLeRobot | None = None, control_frequency: float = 1000.0) -> HwRobot:
    """Construct a Robot bypassing hardware init."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "test_arm"
    hw.action_horizon = 8
    hw.data_config = None
    hw.control_frequency = control_frequency
    hw.action_sleep_time = 1.0 / control_frequency
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_arm_executor")
    hw._shutdown_event = threading.Event()
    hw.mesh = None
    hw.peer_id = None
    hw.robot = fake if fake is not None else _FakeLeRobot()
    return hw


# ---------------------------------------------------------------------------
# Task lifecycle: start / status / stop state machine
# ---------------------------------------------------------------------------


class TestTaskStatusReporting:
    def test_idle_status_text(self):
        hw = _make_robot()
        result = hw.get_task_status()
        assert result["status"] == "success"
        assert "IDLE" in result["content"][0]["text"]
        hw.cleanup()

    def test_running_status_reports_duration_and_steps(self, monkeypatch):
        hw = _make_robot()
        hw._task_state.status = TaskStatus.RUNNING
        hw._task_state.instruction = "pick cube"
        hw._task_state.start_time = 100.0
        hw._task_state.step_count = 7
        monkeypatch.setattr("strands_robots.hardware_robot.time.time", lambda: 103.5)

        text = hw.get_task_status()["content"][0]["text"]
        assert "RUNNING" in text
        assert "pick cube" in text
        assert "Duration: 3.5s" in text
        assert "Steps: 7" in text
        hw.cleanup()

    def test_error_status_includes_error_message(self):
        hw = _make_robot()
        hw._task_state.status = TaskStatus.ERROR
        hw._task_state.error_message = "policy unreachable"
        text = hw.get_task_status()["content"][0]["text"]
        assert "ERROR" in text
        assert "policy unreachable" in text
        hw.cleanup()


class TestStopTask:
    def test_stop_when_idle_is_noop_success(self):
        hw = _make_robot()
        result = hw.stop_task()
        assert result["status"] == "success"
        assert "No task running" in result["content"][0]["text"]
        hw.cleanup()

    def test_stop_running_sets_stopped_and_cancels_future(self):
        hw = _make_robot()
        hw._task_state.status = TaskStatus.RUNNING
        hw._task_state.instruction = "wave"

        cancelled = {"called": False}

        class _Fut:
            def cancel(self) -> bool:
                cancelled["called"] = True
                return True

        hw._task_state.task_future = _Fut()
        result = hw.stop_task()
        assert result["status"] == "success"
        assert hw._task_state.status == TaskStatus.STOPPED
        assert cancelled["called"] is True
        hw.cleanup()


class TestStartTask:
    def test_start_when_already_running_returns_error(self):
        hw = _make_robot()
        hw._task_state.status = TaskStatus.RUNNING
        hw._task_state.instruction = "busy"
        result = hw.start_task("new task", policy_port=5555)
        assert result["status"] == "error"
        assert "already running" in result["content"][0]["text"].lower()
        hw.cleanup()

    def test_start_submits_and_completes(self):
        fake = _FakeLeRobot()
        hw = _make_robot(fake)

        captured = {}

        def _fake_sync(instruction, port, host, provider, duration):
            captured["instruction"] = instruction
            hw._task_state.status = TaskStatus.COMPLETED
            return {"status": "success", "content": [{"text": "done"}]}

        hw._execute_task_sync = _fake_sync  # type: ignore[assignment]
        result = hw.start_task("grab", policy_port=5555, duration=1.0)
        assert result["status"] == "success"
        assert "grab" in result["content"][0]["text"]
        # Wait for the background future to run.
        hw._task_state.task_future.result(timeout=5)
        assert captured["instruction"] == "grab"
        hw.cleanup()


# ---------------------------------------------------------------------------
# Async execution loop: connect -> policy -> observe -> act
# ---------------------------------------------------------------------------


class TestConnectRobot:
    def test_connect_success_flips_connected(self):
        fake = _FakeLeRobot(connected=False)
        hw = _make_robot(fake)
        ok, err = asyncio.run(hw._connect_robot())
        assert ok is True
        assert err == ""
        assert fake.is_connected is True
        hw.cleanup()

    def test_connect_already_connected_short_circuits(self):
        fake = _FakeLeRobot(connected=True)
        hw = _make_robot(fake)
        ok, err = asyncio.run(hw._connect_robot())
        assert ok is True
        assert err == ""
        hw.cleanup()

    def test_connect_uncalibrated_returns_error(self):
        fake = _FakeLeRobot(connected=False, calibrated=False)
        hw = _make_robot(fake)
        ok, err = asyncio.run(hw._connect_robot())
        assert ok is False
        assert "not calibrated" in err
        hw.cleanup()

    def test_connect_failure_is_reported(self):
        fake = _FakeLeRobot(connected=False)

        def _boom(calibrate: bool = False) -> None:
            raise OSError("port /dev/null busy")

        fake.connect = _boom  # type: ignore[method-assign]
        hw = _make_robot(fake)
        ok, err = asyncio.run(hw._connect_robot())
        assert ok is False
        assert "connection failed" in err.lower()
        hw.cleanup()


class TestInitializePolicy:
    def test_filters_camera_keys_from_state_keys(self):
        fake = _FakeLeRobot()
        fake.config.cameras = {"wrist": {}}

        def _obs() -> dict[str, Any]:
            return {"j0.pos": 0.0, "wrist": b"img"}

        fake.get_observation = _obs  # type: ignore[method-assign]
        hw = _make_robot(fake)
        policy = _StubPolicy()
        ok = asyncio.run(hw._initialize_policy(policy))
        assert ok is True
        assert policy.state_keys == ["j0.pos"]
        hw.cleanup()

    def test_returns_false_when_observation_raises(self):
        fake = _FakeLeRobot()

        def _boom() -> dict[str, Any]:
            raise RuntimeError("no obs")

        fake.get_observation = _boom  # type: ignore[method-assign]
        hw = _make_robot(fake)
        ok = asyncio.run(hw._initialize_policy(_StubPolicy()))
        assert ok is False
        hw.cleanup()


class TestGetPolicy:
    def test_missing_port_raises(self):
        hw = _make_robot()
        with pytest.raises(ValueError, match="policy_port is required"):
            asyncio.run(hw._get_policy(policy_port=None))
        hw.cleanup()

    def test_builds_policy_via_create_policy(self, monkeypatch):
        hw = _make_robot()
        hw.data_config = "so101"
        captured = {}

        def _fake_create(provider, **cfg):
            captured["provider"] = provider
            captured["cfg"] = cfg
            return _StubPolicy()

        monkeypatch.setattr("strands_robots.policies.create_policy", _fake_create)
        policy = asyncio.run(hw._get_policy(policy_port=5555, policy_host="h", policy_provider="mock"))
        assert isinstance(policy, _StubPolicy)
        assert captured["provider"] == "mock"
        assert captured["cfg"]["port"] == 5555
        assert captured["cfg"]["host"] == "h"
        assert captured["cfg"]["data_config"] == "so101"
        hw.cleanup()


@pytest.mark.timeout(30)
class TestExecuteTaskAsync:
    def test_full_loop_sends_actions_and_completes(self, monkeypatch):
        """One observe->act iteration runs, then the loop terminates.

        The control loop is bounded by ``while time.time() - start < duration``.
        Driving that with a hand-counted finite iterator is fragile: it assumes
        the code reads ``time.time()`` an exact number of times before the loop
        (connect/policy-init logging, the Python/lerobot build, or any future
        refactor that adds a clock read would desync the iterator, leave the
        captured ``start`` pinned to the terminal value, and spin the loop
        forever). Instead use a stateful clock that advances ONLY when the
        policy emits its chunk: ``start`` is captured at t=0 regardless of how
        many intervening reads occur, and the post-chunk jump trips the guard
        deterministically after exactly one iteration.
        """
        fake = _FakeLeRobot(connected=False)
        hw = _make_robot(fake, control_frequency=1000.0)

        clock = {"now": 0.0}
        monkeypatch.setattr(
            "strands_robots.hardware_robot.time.time",
            lambda: clock["now"],
        )

        class _ClockAdvancingPolicy(_StubPolicy):
            async def get_actions(self, observation, instruction):
                actions = await super().get_actions(observation, instruction)
                # Jump past the duration so the next guard check exits the loop.
                clock["now"] += 100.0
                return actions

        async def _fake_get_policy(*a, **k):
            return _ClockAdvancingPolicy()

        hw._get_policy = _fake_get_policy  # type: ignore[assignment]

        asyncio.run(hw._execute_task_async("pick", policy_port=5555, duration=0.05))
        assert hw._task_state.status == TaskStatus.COMPLETED
        assert len(fake.sent_actions) == 2  # two actions in the policy chunk
        assert hw._task_state.step_count == 2
        hw.cleanup()

    def test_connect_failure_sets_error_state(self):
        fake = _FakeLeRobot(connected=False, calibrated=False)
        hw = _make_robot(fake)
        asyncio.run(hw._execute_task_async("pick", policy_port=5555, duration=0.01))
        assert hw._task_state.status == TaskStatus.ERROR
        assert "not calibrated" in hw._task_state.error_message
        hw.cleanup()

    def test_policy_init_failure_sets_error(self, monkeypatch):
        fake = _FakeLeRobot(connected=True)
        hw = _make_robot(fake)

        async def _fake_get_policy(*a, **k):
            return _StubPolicy()

        async def _fail_init(policy):
            return False

        hw._get_policy = _fake_get_policy  # type: ignore[assignment]
        hw._initialize_policy = _fail_init  # type: ignore[assignment]
        asyncio.run(hw._execute_task_async("pick", policy_port=5555, duration=0.01))
        assert hw._task_state.status == TaskStatus.ERROR
        assert "Failed to initialize policy" in hw._task_state.error_message
        hw.cleanup()


@pytest.mark.timeout(30)
class TestRunPolicyObject:
    """``run_policy(policy_object=...)`` - sim-parity rollout for pre-built policies.

    The hardware counterpart of ``Simulation.run_policy(policy_object=...)``:
    the caller's own object must be initialized and driven through the same
    control loop ``start_task`` uses, ``n_steps`` must bound the applied
    actions deterministically (no wall-clock dependence), and the server-policy
    construction path (``_get_policy``) must never run.
    """

    def test_drives_prebuilt_object_and_reports_json(self):
        fake = _FakeLeRobot(connected=False)
        hw = _make_robot(fake)
        policy = _StubPolicy()

        async def _boom(*a, **k):
            raise AssertionError("_get_policy must not be called when policy_object is given")

        hw._get_policy = _boom  # type: ignore[assignment]
        result = hw.run_policy(policy_object=policy, instruction="pick", n_steps=4)

        assert result["status"] == "success"
        payload = tool_json(result)
        assert payload["status"] == "completed"
        assert payload["steps"] == 4
        assert payload["policy"] == "_StubPolicy"
        assert len(fake.sent_actions) == 4
        # The caller's OWN object was initialized and driven (identity, not a
        # rebuilt copy): state keys + the RTC control rate landed on it.
        assert policy.state_keys == ["j0.pos", "j1.pos"]
        assert policy.control_frequency_calls == [1000.0]
        assert all(d == 0 for d in policy.observed_delays)
        hw.cleanup()

    def test_n_steps_stops_mid_chunk(self):
        fake = _FakeLeRobot(connected=False)
        hw = _make_robot(fake)

        class _FiveChunk(_StubPolicy):
            async def get_actions(self, observation, instruction):
                return [{"j0.pos": 0.1 * i} for i in range(5)]

        result = hw.run_policy(policy_object=_FiveChunk(), instruction="pick", n_steps=3)
        assert result["status"] == "success"
        assert tool_json(result)["steps"] == 3
        assert len(fake.sent_actions) == 3  # stopped inside the 5-action chunk
        hw.cleanup()

    def test_requires_policy_object(self):
        hw = _make_robot()
        result = hw.run_policy(policy_object=None)  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert "policy_object is required" in result["content"][0]["text"]
        hw.cleanup()

    def test_rejects_concurrent_task(self):
        hw = _make_robot()
        hw._task_state.status = TaskStatus.RUNNING
        hw._task_state.instruction = "busy"
        result = hw.run_policy(policy_object=_StubPolicy(), instruction="pick", n_steps=1)
        assert result["status"] == "error"
        assert "already running" in result["content"][0]["text"].lower()
        hw._task_state.status = TaskStatus.IDLE
        hw.cleanup()

    def test_connect_failure_reports_error_payload(self):
        fake = _FakeLeRobot(connected=False, calibrated=False)
        hw = _make_robot(fake)
        result = hw.run_policy(policy_object=_StubPolicy(), instruction="pick", n_steps=1)
        assert result["status"] == "error"
        payload = tool_json(result)
        assert payload["status"] == "error"
        assert "not calibrated" in payload["error"]
        assert fake.sent_actions == []
        hw.cleanup()


class _RtcChunkPolicy(Policy):
    """RTC-capable policy: emits a 5-action chunk but owns a 2-step re-query.

    ``supports_rtc`` + ``execution_horizon == 2`` mean a correct consumer
    (``resolve_chunk_length``) executes exactly 2 actions before re-querying,
    regardless of the caller's ``action_horizon`` (the RTC policy owns the
    interval). It records the control rate and observed-delay the control loop
    supplies via the base ``Policy`` RTC hooks.
    """

    supports_rtc = True
    actions_per_step = 5  # trained chunk length emitted by get_actions

    def __init__(self) -> None:
        self.state_keys: list[str] | None = None
        self.control_frequency_calls: list[float] = []
        self.observed_delays: list[int | None] = []

    @property
    def provider_name(self) -> str:
        return "rtc-test"

    @property
    def requires_images(self) -> bool:
        return False

    @property
    def execution_horizon(self) -> int:
        # RTC re-query interval, deliberately shorter than the 5-action chunk.
        return 2

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.state_keys = robot_state_keys

    def set_control_frequency(self, hz: float) -> None:
        self.control_frequency_calls.append(hz)

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        self.observed_delays.append(steps)

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return [{f"j{i}.pos": 0.1 * i} for i in range(self.actions_per_step)]


@pytest.mark.timeout(30)
class TestExecuteTaskAsyncRtcContract:
    """Hardware control loop honours the same RTC contract as the sim runner.

    Mirrors ``tests/simulation/test_policy_runner_async_rtc.py`` for the real-
    robot path: the loop must set the control frequency once before the rollout,
    supply a counted (zero) observed-delay per inference, and size each chunk by
    the policy's RTC ``execution_horizon`` rather than the raw ``action_horizon``
    slice. Pre-fix this consumed the full 5-action chunk (``[:action_horizon]``
    with ``action_horizon == 8``); the fix consumes exactly ``execution_horizon``
    (2) per query.
    """

    def _run_one_iteration(self, monkeypatch, policy: Policy, *, control_frequency: float = 100.0):
        fake = _FakeLeRobot(connected=False)
        hw = _make_robot(fake, control_frequency=control_frequency)

        clock = {"now": 0.0}
        monkeypatch.setattr(
            "strands_robots.hardware_robot.time.time",
            lambda: clock["now"],
        )

        # Advance the clock past the duration once the chunk is fetched so the
        # loop runs exactly one observe->infer->apply iteration.
        orig_get_actions = policy.get_actions

        async def _advancing_get_actions(observation_dict, instruction, **kwargs):
            actions = await orig_get_actions(observation_dict, instruction, **kwargs)
            clock["now"] += 100.0
            return actions

        policy.get_actions = _advancing_get_actions  # type: ignore[method-assign]

        async def _fake_get_policy(*a, **k):
            return policy

        hw._get_policy = _fake_get_policy  # type: ignore[assignment]
        asyncio.run(hw._execute_task_async("pick", policy_port=5555, duration=0.05))
        return hw, fake

    def test_sets_control_frequency_once_before_loop(self, monkeypatch):
        policy = _RtcChunkPolicy()
        hw, _ = self._run_one_iteration(monkeypatch, policy, control_frequency=100.0)
        assert hw._task_state.status == TaskStatus.COMPLETED
        # Set exactly once, before the rollout, with the loop's control rate.
        assert policy.control_frequency_calls == [100.0]
        hw.cleanup()

    def test_supplies_zero_observed_delay_per_inference(self, monkeypatch):
        policy = _RtcChunkPolicy()
        hw, _ = self._run_one_iteration(monkeypatch, policy)
        # Synchronous loop: exactly 0 control steps elapse during inference.
        assert policy.observed_delays, "policy was never queried"
        assert all(d == 0 for d in policy.observed_delays), policy.observed_delays
        hw.cleanup()

    def test_consumes_rtc_execution_horizon_not_action_horizon(self, monkeypatch):
        policy = _RtcChunkPolicy()
        hw, fake = self._run_one_iteration(monkeypatch, policy)
        # action_horizon defaults to 8 and the chunk is 5 actions, so the old
        # robot_actions[:self.action_horizon] slice would have applied all 5.
        # The RTC contract caps consumption at execution_horizon (2).
        assert len(fake.sent_actions) == 2
        assert hw._task_state.step_count == 2
        hw.cleanup()


# ---------------------------------------------------------------------------
# Tool stream dispatch
# ---------------------------------------------------------------------------


def _drain(agen) -> list:
    async def _run() -> list:
        return [ev async for ev in agen]

    return asyncio.run(_run())


class TestStreamDispatch:
    def test_execute_requires_instruction_and_port(self):
        hw = _make_robot()
        events = _drain(hw.stream({"toolUseId": "t1", "input": {"action": "execute"}}, {}))
        result = events[-1].tool_result
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"]
        hw.cleanup()

    def test_status_action_dispatches(self):
        hw = _make_robot()
        events = _drain(hw.stream({"toolUseId": "t2", "input": {"action": "status"}}, {}))
        result = events[-1].tool_result
        assert result["status"] == "success"
        assert "IDLE" in result["content"][0]["text"]
        hw.cleanup()

    def test_stop_action_dispatches(self):
        hw = _make_robot()
        events = _drain(hw.stream({"toolUseId": "t3", "input": {"action": "stop"}}, {}))
        result = events[-1].tool_result
        assert result["status"] == "success"
        hw.cleanup()

    def test_unknown_action_is_error(self):
        hw = _make_robot()
        events = _drain(hw.stream({"toolUseId": "t4", "input": {"action": "fly"}}, {}))
        result = events[-1].tool_result
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]
        hw.cleanup()

    def test_start_action_dispatches_to_start_task(self):
        hw = _make_robot()
        captured = {}

        def _fake_start(instruction, port, host, provider, duration):
            captured["instruction"] = instruction
            return {"status": "success", "content": [{"text": "started"}]}

        hw.start_task = _fake_start  # type: ignore[assignment]
        events = _drain(
            hw.stream(
                {"toolUseId": "t5", "input": {"action": "start", "instruction": "go", "policy_port": 5555}},
                {},
            )
        )
        result = events[-1].tool_result
        assert result["status"] == "success"
        assert captured["instruction"] == "go"
        hw.cleanup()


# ---------------------------------------------------------------------------
# Status / spec surface
# ---------------------------------------------------------------------------


class TestStatusSurface:
    def test_get_status_reports_connection_and_task(self):
        fake = _FakeLeRobot(connected=True)
        hw = _make_robot(fake)
        status = asyncio.run(hw.get_status())
        assert status["robot_name"] == "test_arm"
        assert status["is_connected"] is True
        assert status["task_status"] == "idle"
        hw.cleanup()

    def test_get_status_is_fail_soft_when_the_device_read_raises(self):
        """A device read that raises must degrade to a structured error dict.

        ``get_status`` is the operator's health probe; a raising serial/USB
        backend must never propagate out of it and crash the caller. Instead
        it reports ``task_status="error"`` with the failure text and a safe
        ``is_connected=False``, so a supervising agent can react rather than
        take an unhandled exception.
        """

        class _RaisingDevice:
            name = "raising_arm"

            @property
            def is_connected(self) -> bool:
                raise RuntimeError("serial bus fault")

        hw = _make_robot()
        hw.robot = _RaisingDevice()
        status = asyncio.run(hw.get_status())
        assert status["task_status"] == "error"
        assert status["is_connected"] is False
        assert status["robot_name"] == "test_arm"
        assert "serial bus fault" in status["error"]
        hw.cleanup()

    def test_get_status_surfaces_task_error_message(self):
        """A recorded task error is surfaced under ``task_error`` in the healthy
        status dict, so an operator sees *why* the last run failed without a
        separate call."""
        fake = _FakeLeRobot(connected=True)
        hw = _make_robot(fake)
        hw._task_state.status = TaskStatus.ERROR
        hw._task_state.error_message = "gripper stalled"
        status = asyncio.run(hw.get_status())
        assert status["task_status"] == "error"
        assert status["task_error"] == "gripper stalled"
        hw.cleanup()

    def test_tool_spec_advertises_actions(self):
        hw = _make_robot()
        spec = hw.tool_spec
        assert spec["name"] == "test_arm"
        enum = spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        assert set(enum) == {"execute", "start", "status", "stop"}
        assert hw.tool_type == "robot"
        assert hw.tool_name == "test_arm"
        hw.cleanup()


# ---------------------------------------------------------------------------
# Robot construction: input validation
# ---------------------------------------------------------------------------


class TestInitializeRobotValidation:
    def test_unsupported_robot_argument_is_rejected(self):
        """``_initialize_robot`` accepts a lerobot Robot, a RobotConfig, or a
        robot-type string; anything else is an operator mistake and must raise
        rather than be silently coerced or ignored.
        """
        hw = _make_robot()
        with pytest.raises(ValueError, match="Unsupported robot type"):
            hw._initialize_robot(123, None)  # type: ignore[arg-type]
        hw.cleanup()


# ---------------------------------------------------------------------------
# Mesh teleop publish / receive lifecycle
# ---------------------------------------------------------------------------


class _FakeMesh:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive


class _FakePublisher:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.topic = "strands/peer/input/leader"
        self.started = False
        self.stopped = False
        self.stats = {"frames": 42, "hz_actual": 49.5}

    def start(self) -> None:
        self.started = True

    def stop(self) -> dict[str, Any]:
        self.stopped = True
        return {"device": "leader", "frames": 42, "hz_actual": 49.5}


class _FakeReceiver:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.topic = "strands/src/input/leader"
        self.started = False
        self.stopped = False
        self.stats = {"frames_received": 10, "hz_actual": 50.0}

    def start(self) -> None:
        self.started = True

    def stop(self) -> dict[str, Any]:
        self.stopped = True
        return {"source": "src", "frames_received": 10, "hz_actual": 50.0}


class TestTeleopPublish:
    def test_publish_requires_active_mesh(self):
        hw = _make_robot()
        hw.mesh = None
        result = hw.start_teleop_publish(teleoperator=object())
        assert result["status"] == "error"
        assert "Mesh not active" in result["content"][0]["text"]
        hw.cleanup()

    def test_publish_starts_publisher_and_reports_peer(self, monkeypatch):
        import strands_robots.mesh as mesh_mod

        monkeypatch.setattr(mesh_mod, "InputPublisher", _FakePublisher)
        hw = _make_robot()
        hw.mesh = _FakeMesh(alive=True)
        hw.peer_id = "peer-1"
        result = hw.start_teleop_publish(teleoperator=object(), device_name="leader", method="arm", hz=50.0)
        assert result["status"] == "success"
        assert "peer-1" in result["content"][0]["text"]
        assert hw._input_publishers["leader"].started is True
        hw.cleanup()

    def test_publish_twice_stops_prior_publisher(self, monkeypatch):
        import strands_robots.mesh as mesh_mod

        monkeypatch.setattr(mesh_mod, "InputPublisher", _FakePublisher)
        hw = _make_robot()
        hw.mesh = _FakeMesh(alive=True)
        hw.peer_id = "peer-1"
        hw.start_teleop_publish(teleoperator=object(), device_name="leader")
        first = hw._input_publishers["leader"]
        hw.start_teleop_publish(teleoperator=object(), device_name="leader")
        assert first.stopped is True
        assert hw._input_publishers["leader"] is not first
        hw.cleanup()


class TestTeleopReceive:
    def test_receive_requires_active_mesh(self):
        hw = _make_robot()
        hw.mesh = _FakeMesh(alive=False)
        result = hw.start_teleop_receive(source_peer_id="src")
        assert result["status"] == "error"
        assert "Mesh not active" in result["content"][0]["text"]
        hw.cleanup()

    def test_receive_starts_receiver(self, monkeypatch):
        import strands_robots.mesh as mesh_mod

        monkeypatch.setattr(mesh_mod, "InputReceiver", _FakeReceiver)
        hw = _make_robot()
        hw.mesh = _FakeMesh(alive=True)
        result = hw.start_teleop_receive(source_peer_id="src", device_name="leader")
        assert result["status"] == "success"
        key = "src/leader"
        assert hw._input_receivers[key].started is True
        hw.cleanup()


class TestTeleopStopAndStatus:
    def _wire(self, hw):
        hw._input_publishers = {"leader": _FakePublisher()}
        hw._input_receivers = {"src/leader": _FakeReceiver()}

    def test_stop_all_clears_publishers_and_receivers(self):
        hw = _make_robot()
        self._wire(hw)
        result = hw.stop_teleop()
        assert result["status"] == "success"
        assert hw._input_publishers == {}
        assert hw._input_receivers == {}
        hw.cleanup()

    def test_stop_named_device_only(self):
        hw = _make_robot()
        hw._input_publishers = {"leader": _FakePublisher(), "gamepad": _FakePublisher()}
        hw._input_receivers = {}
        hw.stop_teleop(device_name="leader")
        assert "leader" not in hw._input_publishers
        assert "gamepad" in hw._input_publishers
        hw.cleanup()

    def test_stop_with_no_sessions(self):
        hw = _make_robot()
        result = hw.stop_teleop()
        assert result["status"] == "success"
        assert "No active teleop" in result["content"][0]["text"]
        hw.cleanup()

    def test_get_teleop_status_counts_sessions(self):
        hw = _make_robot()
        self._wire(hw)
        result = hw.get_teleop_status()
        assert result["status"] == "success"
        assert len(tool_json(result)["publishers"]) == 1
        assert len(tool_json(result)["receivers"]) == 1
        assert "Publishers: 1 active" in result["content"][0]["text"]
        assert "Receivers: 1 active" in result["content"][0]["text"]
        hw.cleanup()


# ---------------------------------------------------------------------------
# Cleanup / stop
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_stops_running_task_and_mesh(self):
        hw = _make_robot()
        hw._task_state.status = TaskStatus.RUNNING

        class _Mesh:
            def __init__(self) -> None:
                self.stopped = False

            def stop(self) -> None:
                self.stopped = True

        mesh = _Mesh()
        hw.mesh = mesh
        hw.cleanup()
        assert mesh.stopped is True
        assert hw._shutdown_event.is_set()

    def test_cleanup_survives_mesh_stop_error(self, caplog):
        hw = _make_robot()

        class _BadMesh:
            def stop(self) -> None:
                raise RuntimeError("mesh boom")

        hw.mesh = _BadMesh()
        # Should not raise.
        hw.cleanup()

    def test_stop_disconnects_robot(self):
        fake = _FakeLeRobot(connected=True)
        hw = _make_robot(fake)
        asyncio.run(hw.stop())
        assert fake.is_connected is False


class TestExecuteTaskSync:
    def test_sync_runner_no_running_loop_completes(self, monkeypatch):
        """_execute_task_sync drives the async loop via asyncio.run when no
        event loop is running, and reports success on completion."""
        fake = _FakeLeRobot(connected=True)
        hw = _make_robot(fake)

        async def _ok_async(*a, **k):
            hw._task_state.status = TaskStatus.COMPLETED
            hw._task_state.duration = 1.2
            hw._task_state.step_count = 5

        hw._execute_task_async = _ok_async  # type: ignore[assignment]
        result = hw._execute_task_sync("pick", policy_port=5555, policy_provider="mock", duration=1.0)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "completed" in text
        assert "mock" in text
        hw.cleanup()

    def test_sync_runner_reports_error_status(self):
        fake = _FakeLeRobot(connected=True)
        hw = _make_robot(fake)

        async def _err_async(*a, **k):
            hw._task_state.status = TaskStatus.ERROR
            hw._task_state.error_message = "boom"

        hw._execute_task_async = _err_async  # type: ignore[assignment]
        result = hw._execute_task_sync("pick", policy_port=5555)
        assert result["status"] == "error"
        assert "boom" in result["content"][0]["text"]
        hw.cleanup()

    def test_sync_runner_within_running_loop_offloads_to_thread(self):
        """When called from inside a live event loop, _execute_task_sync must
        offload to a worker thread instead of calling asyncio.run() on the
        running loop (which raises "cannot be called from a running event
        loop"). An async tool context drives this path, so it must complete
        and report success rather than crash."""
        fake = _FakeLeRobot(connected=True)
        hw = _make_robot(fake)

        loop_ids: list[int] = []

        async def _ok_async(*a, **k):
            # Records the loop the task actually ran on so we can assert it is
            # a fresh worker-thread loop, not the caller's running loop.
            loop_ids.append(id(asyncio.get_running_loop()))
            hw._task_state.status = TaskStatus.COMPLETED
            hw._task_state.duration = 0.5
            hw._task_state.step_count = 3

        hw._execute_task_async = _ok_async  # type: ignore[assignment]

        async def _driver() -> dict:
            caller_loop_id = id(asyncio.get_running_loop())
            # Blocking call from within a running loop; exercises the
            # ThreadPoolExecutor branch. Must not raise.
            result = hw._execute_task_sync("pick", policy_port=5555, policy_provider="mock")
            return {"result": result, "caller_loop_id": caller_loop_id}

        out = asyncio.run(_driver())
        assert out["result"]["status"] == "success"
        assert "completed" in out["result"]["content"][0]["text"]
        # The task ran on a distinct loop created in the worker thread, proving
        # the running loop was not reused by asyncio.run().
        assert loop_ids and loop_ids[0] != out["caller_loop_id"]
        hw.cleanup()


class TestStreamExecuteHappyPath:
    def test_execute_action_runs_task_and_returns_result(self):
        hw = _make_robot()
        captured = {}

        def _fake_sync(instruction, port, host, provider, duration):
            captured["instruction"] = instruction
            captured["port"] = port
            return {"status": "success", "content": [{"text": "task done"}]}

        hw._execute_task_sync = _fake_sync  # type: ignore[assignment]
        events = _drain(
            hw.stream(
                {"toolUseId": "e1", "input": {"action": "execute", "instruction": "lift", "policy_port": 5555}},
                {},
            )
        )
        result = events[-1].tool_result
        assert result["status"] == "success"
        assert captured["instruction"] == "lift"
        assert captured["port"] == 5555
        hw.cleanup()

    def test_start_action_requires_instruction_and_port(self):
        hw = _make_robot()
        events = _drain(hw.stream({"toolUseId": "e2", "input": {"action": "start", "instruction": "go"}}, {}))
        result = events[-1].tool_result
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"]
        hw.cleanup()


class TestEnsureLerobotRobotsRegistered:
    """Behavior tests for ``_ensure_lerobot_robots_registered``.

    This helper populates lerobot's ``RobotConfig`` choice registry by
    walking every ``lerobot.robots`` subpackage (so robot types whose driver
    lives in a shared module -- e.g. ``so101_follower`` in ``so_follower`` --
    are discovered) and then invoking lerobot's third-party plugin loader.
    The walk is driven entirely by ``pkgutil``/``importlib``/``sys.modules``,
    so every branch is exercised with injected fakes and no real lerobot
    driver imports, USB probes, or hardware.

    The function is ``@functools.cache``d; ``cache_clear()`` runs before each
    case so the walk actually re-executes.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        import strands_robots.hardware_robot as hw

        hw._ensure_lerobot_robots_registered.cache_clear()
        yield
        hw._ensure_lerobot_robots_registered.cache_clear()

    @staticmethod
    def _install_fake_lerobot_robots(monkeypatch, subpackages):
        """Wire a fake ``lerobot.robots`` whose ``pkgutil`` walk yields
        ``subpackages`` -- a list of ``(name, ispkg)`` tuples."""
        import strands_robots.hardware_robot as hw

        fake_robots = types.ModuleType("lerobot.robots")
        fake_robots.__path__ = ["/fake/lerobot/robots"]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "lerobot.robots", fake_robots)

        modinfo = [pkgutil.ModuleInfo(None, name, ispkg) for name, ispkg in subpackages]
        monkeypatch.setattr(hw.pkgutil, "iter_modules", lambda path: iter(modinfo))
        return fake_robots

    def test_lerobot_robots_absent_and_lerobot_absent_is_debug(self, monkeypatch, caplog):
        """lerobot wholly missing -> debug-level, no warning, returns cleanly."""
        import strands_robots.hardware_robot as hw

        monkeypatch.setitem(sys.modules, "lerobot.robots", None)
        monkeypatch.setitem(sys.modules, "lerobot", None)
        with caplog.at_level("WARNING"):
            hw._ensure_lerobot_robots_registered()
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_partial_install_warns(self, monkeypatch, caplog):
        """lerobot present but lerobot.robots unimportable -> a warning fires."""
        import strands_robots.hardware_robot as hw

        # ``import lerobot.robots`` fails, but the ``import lerobot`` probe
        # succeeds -> the partial-install warning branch.
        monkeypatch.setitem(sys.modules, "lerobot.robots", None)
        monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
        with caplog.at_level("WARNING"):
            hw._ensure_lerobot_robots_registered()
        assert any("partial install" in r.message for r in caplog.records)

    def test_walks_subpackages_and_skips_failing_driver(self, monkeypatch):
        """Each importable subpackage is imported; a driver whose import
        raises ImportError/OSError is skipped without crashing the walk."""
        import strands_robots.hardware_robot as hw

        self._install_fake_lerobot_robots(
            monkeypatch,
            [("so_follower", True), ("unitree", True), ("_helpers", False)],
        )
        imported: list[str] = []

        def fake_import(name):
            imported.append(name)
            if name.endswith("unitree"):
                raise OSError("unitree_sdk2py USB probe failed")
            return types.ModuleType(name)

        monkeypatch.setattr(hw.importlib, "import_module", fake_import)
        # Make the third-party plugin loader a no-op success.
        plugins = types.ModuleType("lerobot.utils.import_utils")
        plugins.register_third_party_plugins = lambda: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "lerobot.utils.import_utils", plugins)

        hw._ensure_lerobot_robots_registered()

        # The package subpackages were imported; the non-package module was
        # skipped by the ``is_pkg`` guard.
        assert "lerobot.robots.so_follower" in imported
        assert "lerobot.robots.unitree" in imported
        assert "lerobot.robots._helpers" not in imported

    def test_third_party_loader_missing_is_debug(self, monkeypatch, caplog):
        """Older lerobot without ``register_third_party_plugins`` -> debug,
        built-ins still registered, no warning."""
        import strands_robots.hardware_robot as hw

        self._install_fake_lerobot_robots(monkeypatch, [("so_follower", True)])
        monkeypatch.setattr(hw.importlib, "import_module", lambda name: types.ModuleType(name))
        # A module that lacks the attribute -> the ``from ... import`` raises
        # ImportError -> the "loader unavailable" debug branch.
        empty = types.ModuleType("lerobot.utils.import_utils")
        monkeypatch.setitem(sys.modules, "lerobot.utils.import_utils", empty)

        with caplog.at_level("WARNING"):
            hw._ensure_lerobot_robots_registered()
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_third_party_loader_failure_warns(self, monkeypatch, caplog):
        """A broken third-party plugin loader degrades to a warning, not a
        crash -- hardware init must survive plugin registration failures."""
        import strands_robots.hardware_robot as hw

        self._install_fake_lerobot_robots(monkeypatch, [("so_follower", True)])
        monkeypatch.setattr(hw.importlib, "import_module", lambda name: types.ModuleType(name))

        def boom():
            raise OSError("plugin entry-point probe failed")

        plugins = types.ModuleType("lerobot.utils.import_utils")
        plugins.register_third_party_plugins = boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "lerobot.utils.import_utils", plugins)

        with caplog.at_level("WARNING"):
            hw._ensure_lerobot_robots_registered()
        assert any("third-party plugin registration failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# lazy-connect rollback: a failed connect must not leave the port half-open
# ---------------------------------------------------------------------------


class _HalfOpenConnectLeRobot(_FakeLeRobot):
    """connect() opens the port, then fails the handshake (mirrors lerobot's
    MotorsBus.connect: openPort -> _handshake raises), leaving is_connected
    True. Exposes itself as ``bus`` like the SO follower robots do."""

    def __init__(self) -> None:
        super().__init__(connected=False)
        self.connect_attempts = 0
        self.bus = self
        self.bus_disconnect_calls: list[bool] = []

    def connect(self, calibrate: bool = False) -> None:  # noqa: ARG002 - lerobot signature
        self.connect_attempts += 1
        self._connected = True  # port opened...
        raise ConnectionError("motor handshake failed")  # ...handshake failed

    def disconnect(self, disable_torque: bool = True) -> None:
        self.bus_disconnect_calls.append(disable_torque)
        self._connected = False


def test_send_action_rolls_back_half_open_lazy_connect() -> None:
    """A failed lazy connect must not leave the port half-open: every
    send_action retries the connect and reports error (an unpowered follower
    would otherwise report success forever via fire-and-forget writes)."""
    fake = _HalfOpenConnectLeRobot()
    hw = _make_robot(fake)

    r1 = hw.send_action({"j0.pos": 0.1})
    r2 = hw.send_action({"j0.pos": 0.2})

    assert r1["status"] == "error"
    assert r2["status"] == "error"
    # rollback closed the port, so the second call retried the connect
    assert fake.connect_attempts == 2
    # port closed WITHOUT a torque write (it would raise on a dead bus)
    assert fake.bus_disconnect_calls == [False, False]
    # nothing was ever written to the dead bus
    assert fake.sent_actions == []
    hw.cleanup()


def test_connect_robot_rolls_back_half_open_connect() -> None:
    """The explicit connect path must roll back too: without it, a failed
    connect leaves is_connected True and the NEXT _connect_robot short-circuits
    on "already connected" -- reporting success against a dead bus."""
    fake = _HalfOpenConnectLeRobot()
    hw = _make_robot(fake)

    ok1, err1 = asyncio.run(hw._connect_robot())
    ok2, err2 = asyncio.run(hw._connect_robot())

    assert ok1 is False and "handshake" in err1
    assert ok2 is False and "handshake" in err2  # retried, not "already connected"
    assert fake.connect_attempts == 2
    assert fake.bus_disconnect_calls == [False, False]
    hw.cleanup()
