# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A rollout can neither start nor finish successfully on a shut-down robot.

``_shutdown_event`` is one of the control loop's exit conditions::

    while (
        time.monotonic() - start_mono < duration
        and (n_steps is None or self._task_state.step_count < n_steps)
        and self._task_state.status == TaskStatus.RUNNING
        and not self._stop_requested.is_set()
        and not self._shutdown_event.is_set()   # <- set by cleanup()
    ):

but it was neither an admission check nor a terminal-status discriminator, so
once ``cleanup()`` had set it a rollout ran the whole bring-up, fell out of the
loop on the condition's first evaluation, and reported itself ``completed``.

Two producers reach that state, and they need different fixes - which is why
both are pinned here:

**A task started after ``cleanup()``.** Bring-up is not side-effect-free, so
this is refused at the entry points rather than merely relabelled. Measured on
a two-device arm, calling each entry point once after ``cleanup()``:

    entry point                    verdict on main       connect()  left open
    run_policy                     success                   1        True
    execute (agent tool / mesh)     success                   1        True
    start_task                     raises RuntimeError        0        False

- ``_connect_robot()`` re-opens the motors bus and warms every camera. Because
  ``cleanup()`` does not disconnect the robot and the executor is already shut
  down, those devices stay open for the life of the process - a second
  ``cleanup()`` does not close them either.
- ``Policy.reset()`` is called, clearing the per-episode state of a policy
  object the caller may still be driving (the documented ``policy_object=``
  reuse pattern).
- The policy is never queried and the arm is never commanded, yet two of the
  three entry points returned ``status="success"`` with ``steps: 0`` - a result
  indistinguishable from a rollout that really drove the arm.
- ``start_task`` was the odd one out: it raised
  ``RuntimeError("cannot schedule new futures after shutdown")`` from the
  executor submit, naming a ``concurrent.futures`` internal rather than the
  robot, against this module's contract that a handler returns an error dict.

**A ``cleanup()`` landing during bring-up.** No entry-point guard can cover
this one: ``cleanup()`` sets ``_shutdown_event`` and only then calls
``stop_task()``, gated on ``status == RUNNING``. A task still in ``CONNECTING``
therefore gets no stop latch, finishes bring-up, and exits the loop on its
first evaluation::

    status when cleanup() ran : connecting
    _stop_requested set       : False        <- stop_task() never called
    verdict on main           : success
    task status / steps       : completed / 0

so the terminal block has to treat ``_shutdown_event`` the way it already
treats the stop latch.

No serial port is opened and no arm is commanded: an in-memory double stands in
for the driver, and the bring-up window is opened by the test rather than slept
through, so every assertion here is deterministic.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from strands_robots.policies.base import Policy

#: Upper bound on any wait, so a broken contract fails instead of hanging.
DEADLINE = 10.0


class _Camera:
    """Camera double recording whether bring-up left it open."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._connected = False
        self.connect_calls = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, warmup: bool = True) -> None:
        self.connect_calls += 1
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False


class Bus:
    """Motors-bus double recording every device effect a rollout has.

    ``connect()`` blocks on ``connect_gate`` when the test arms it, which is how
    the ``cleanup()``-during-bring-up window is opened deterministically instead
    of being slept through.
    """

    def __init__(self) -> None:
        self.name = self.robot_type = "arm"
        self.is_calibrated = True
        self.cameras = {"wrist": _Camera("wrist")}
        self.config = type("Cfg", (), {"cameras": {}})()
        self._connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.commands: list[dict[str, Any]] = []
        self.connect_entered = threading.Event()
        self.connect_gate: threading.Event | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = False) -> None:
        self.connect_calls += 1
        self.connect_entered.set()
        if self.connect_gate is not None and not self.connect_gate.wait(DEADLINE):  # pragma: no cover - deadline
            raise TimeoutError("bring-up gate never opened")
        for cam in self.cameras.values():
            cam.connect()
        self._connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        for cam in self.cameras.values():
            cam.disconnect()
        self._connected = False

    def get_observation(self) -> dict[str, Any]:
        return {"gripper.pos": 0.0}

    def send_action(self, action: dict[str, Any]) -> None:
        self.commands.append(action)


class CountingPolicy(Policy):
    """Records the per-episode reset and query calls bring-up makes."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.get_actions_calls = 0

    @property
    def provider_name(self) -> str:
        return "counting"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        pass

    def reset(self, seed: int | None = None) -> None:
        self.reset_calls += 1

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.get_actions_calls += 1
        return [{"gripper.pos": 0.11}] * 4


def make_robot(bus: Bus) -> HwRobot:
    """Build a Robot bypassing hardware init (the pattern used across tests/)."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "arm"
    hw.action_horizon = 4
    hw.data_config = None
    hw.control_frequency = 500.0
    hw.action_sleep_time = 1.0 / 500.0
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = bus
    return hw


@pytest.fixture
def bus() -> Bus:
    return Bus()


@pytest.fixture
def hw(bus: Bus) -> Any:
    robot = make_robot(bus)
    yield robot
    if bus.connect_gate is not None:
        bus.connect_gate.set()
    robot.cleanup()


def _text(result: dict[str, Any]) -> str:
    """The text of a tool-shaped result."""
    return " ".join(block["text"] for block in result["content"] if "text" in block)


def _shut_down(hw: HwRobot) -> None:
    """Put the robot in the post-``cleanup()`` state the guard refuses."""
    hw.cleanup()
    assert hw._shutdown_event.is_set()


class TestEveryEntryPointRefusesAfterShutdown:
    """All three admission paths refuse in the same tool shape."""

    def test_run_policy_refuses(self, hw: HwRobot, bus: Bus):
        """``run_policy`` errors naming the robot, not a completed rollout."""
        _shut_down(hw)

        result = hw.run_policy(policy_object=CountingPolicy(), instruction="after shutdown", n_steps=6)

        assert result["status"] == "error"
        assert "run_policy" in _text(result)
        assert "shut down" in _text(result)

    def test_execute_action_refuses(self, hw: HwRobot, bus: Bus):
        """The agent-tool ``execute`` / mesh dispatch chokepoint refuses too."""
        _shut_down(hw)

        result = hw._execute_task_sync("after shutdown", policy_object=CountingPolicy(), n_steps=6)

        assert result["status"] == "error"
        assert "execute_task" in _text(result)
        assert "shut down" in _text(result)

    def test_start_task_refuses_instead_of_raising(self, hw: HwRobot, bus: Bus):
        """``start_task`` reports the robot, not a ``concurrent.futures`` internal.

        Pre-fix this raised ``RuntimeError("cannot schedule new futures after
        shutdown")`` out of the executor submit.
        """
        _shut_down(hw)

        result = hw.start_task("after shutdown")

        assert result["status"] == "error"
        assert "start_task" in _text(result)
        assert "shut down" in _text(result)
        assert "futures" not in _text(result)

    def test_the_refusal_does_not_leave_the_claim_held(self, hw: HwRobot, bus: Bus):
        """A refused rollout must not lock the bus out of future ones."""
        _shut_down(hw)

        hw.run_policy(policy_object=CountingPolicy(), instruction="after shutdown", n_steps=6)

        assert hw._task_claimed is False


class TestRefusalHappensBeforeAnyDeviceEffect:
    """The refusal is what makes the leak and the state clobber unreachable."""

    @pytest.mark.parametrize("entry", ["run_policy", "execute_task", "start_task"])
    def test_no_entry_point_re_opens_the_hardware(self, hw: HwRobot, bus: Bus, entry: str):
        """``cleanup()`` does not disconnect, so a re-open would never be closed."""
        _shut_down(hw)

        policy = CountingPolicy()
        if entry == "run_policy":
            hw.run_policy(policy_object=policy, instruction="after shutdown", n_steps=6)
        elif entry == "execute_task":
            hw._execute_task_sync("after shutdown", policy_object=policy, n_steps=6)
        else:
            hw.start_task("after shutdown")

        assert bus.connect_calls == 0
        assert bus.is_connected is False
        assert bus.cameras["wrist"].connect_calls == 0
        assert bus.cameras["wrist"].is_connected is False

    def test_a_reusable_policy_keeps_its_episode_state(self, hw: HwRobot, bus: Bus):
        """``Policy.reset()`` must not fire for a rollout that cannot run.

        A caller may drive one policy object through several tasks, so clearing
        its action-chunk cache / sampler RNG on a refused call would corrupt a
        rollout running elsewhere.
        """
        _shut_down(hw)
        policy = CountingPolicy()

        hw.run_policy(policy_object=policy, instruction="after shutdown", n_steps=6)

        assert policy.reset_calls == 0
        assert policy.get_actions_calls == 0

    def test_the_arm_is_never_commanded(self, hw: HwRobot, bus: Bus):
        _shut_down(hw)

        hw.run_policy(policy_object=CountingPolicy(), instruction="after shutdown", n_steps=6)

        assert bus.commands == []


class TestNoFalseTerminalStatus:
    """A rollout that never ran is never reported as a completed one."""

    def test_the_task_state_is_not_marked_completed(self, hw: HwRobot, bus: Bus):
        """Pre-fix this was ``COMPLETED`` with 0 steps."""
        _shut_down(hw)

        hw.run_policy(policy_object=CountingPolicy(), instruction="after shutdown", n_steps=6)

        assert hw._task_state.status is not TaskStatus.COMPLETED
        assert hw._task_state.step_count == 0

    def test_the_json_payload_does_not_claim_a_rollout(self, hw: HwRobot, bus: Bus):
        """``run_policy``'s refusal carries no ``completed`` status payload."""
        _shut_down(hw)

        result = hw.run_policy(policy_object=CountingPolicy(), instruction="after shutdown", n_steps=6)

        payloads = [block["json"] for block in result["content"] if "json" in block]
        assert all(block.get("status") != "completed" for block in payloads)

    def test_a_shutdown_during_bring_up_reports_stopped_not_completed(self, hw: HwRobot, bus: Bus):
        """The path no entry-point guard can cover.

        ``cleanup()`` sets ``_shutdown_event`` then calls ``stop_task()`` only
        for ``status == RUNNING``, so a task still in ``CONNECTING`` gets no stop
        latch and used to fall out of the loop reporting ``completed`` / 0 steps.
        """
        bus.connect_gate = threading.Event()
        out: dict[str, Any] = {}
        thread = threading.Thread(
            target=lambda: out.__setitem__(
                "result",
                hw.run_policy(policy_object=CountingPolicy(), instruction="interrupted", n_steps=6),
            ),
            daemon=True,
        )
        thread.start()
        assert bus.connect_entered.wait(DEADLINE), "rollout never reached connect()"
        assert hw._task_state.status is TaskStatus.CONNECTING

        # The shutdown lands mid-bring-up, so no stop latch is set for it.
        hw.cleanup()
        assert hw._stop_requested.is_set() is False
        bus.connect_gate.set()
        thread.join(DEADLINE)
        assert not thread.is_alive(), "rollout never finished"

        assert out["result"]["status"] == "error"
        assert hw._task_state.status is TaskStatus.STOPPED
        assert hw._task_state.step_count == 0
        assert bus.commands == []


class TestAHealthyRolloutIsUnaffected:
    """The guard refuses exactly the shut-down case and nothing else."""

    def test_a_rollout_before_cleanup_still_completes(self, hw: HwRobot, bus: Bus):
        policy = CountingPolicy()

        result = hw.run_policy(policy_object=policy, instruction="healthy", n_steps=6)

        assert result["status"] == "success"
        assert hw._task_state.status is TaskStatus.COMPLETED
        assert hw._task_state.step_count == 6
        assert len(bus.commands) == 6
        assert policy.reset_calls == 1
        assert bus.connect_calls == 1

    def test_start_task_before_cleanup_still_submits(self, hw: HwRobot, bus: Bus):
        """A well-formed start still submits; the shutdown guard is the only refusal.

        The port is explicit because ``start_task`` now judges it before the
        submit: an absent ``policy_port`` is refused on its own terms (no policy
        can be built from it), which would mask the property under test here.
        """
        result = hw.start_task("healthy", policy_port=5555)

        assert result["status"] == "success"
        assert "Task started" in _text(result)
