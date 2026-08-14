"""Behavior tests for the accepted ``policy_port`` domain of a hardware task.

``policy_port`` is the one caller-supplied value that decides whether a policy
exists at all: ``Robot._get_policy`` hands it to ``create_policy``, whose
provider constructor dials it. Its siblings in the same signature are refused
before the motors bus is claimed - ``duration`` and ``n_steps`` both are, and
the call sites say why ("a peer-supplied budget is refused before the arm is
commanded rather than after"). The port was read only inside
``_execute_task_async``, *after* ``_connect_robot`` had energized the arm.

These tests pin that a port no policy can be built from is refused where the
budget already is:

    - the refusal precedes ``_connect_robot``, so the bring-up window that
      method's own comment describes as "a motors-bus handshake plus per-camera
      warmup - seconds on a real arm" is not spent on a call that cannot start;
    - it precedes the bus claim, so a rejected port does not take the arm away
      from a rollout that could still run;
    - ``start_task`` reports it instead of ``status="success"`` / "Task started"
      followed by a failure on the executor thread, where nobody is left to tell;
    - a supplied-but-unusable ``0`` / ``False`` is named as invalid rather than
      reported as ``"policy_port is required"``, which said a port the caller
      passed was missing;
    - a pre-built ``policy_object`` makes the port inert on
      ``_execute_task_sync``, so it is not validated there - refusing a value the
      call never reads would be a false rejection;
    - every port that IS usable still connects and reaches the policy build;
    - the accepted domain is the shared ``tcp_port_error`` one the policy
      providers themselves apply, so the same port cannot be accepted by the
      arm's task entry points and refused by the provider they hand it to.

No serial/USB hardware is touched: the driver is an in-memory fake, the connect
path is a recording stub, and the policy is a structural stub.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import threading
from typing import Any

import numpy as np
import pytest

import strands_robots.hardware_robot as hardware_robot
from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from strands_robots.utils import tcp_port_error
from tests._daemon_executor import DaemonThreadExecutor
from tests.test_hardware_control_loop_rate_guard import _FakeArm

# Ports no policy can be built from. ``0`` asks the kernel for an ephemeral port
# rather than naming one to dial; the out-of-range values have nothing to
# connect to; ``True``/``False`` are ``int`` subclasses that would act as
# silent ports 1 and 0; the rest are not port numbers at all. ``None`` is
# excluded here and covered on its own: it is the "not supplied" spelling, so
# its message differs.
UNUSABLE_PORTS: list[Any] = [
    0,
    -1,
    65536,
    99999,
    float("nan"),
    float("inf"),
    True,
    False,
    2.7,
    "5555",
    [5555],
    np.int64(5555),
]

# Ports that name a socket a provider can dial.
USABLE_PORTS: list[Any] = [1, 5555, 65535]


class _Policy:
    """Structural stand-in for a policy: the members the control loop reads."""

    supports_rtc = False
    execution_horizon = 1

    def set_control_frequency(self, hz: float) -> None:
        return None

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        return None

    def reset(self, seed: int | None = None) -> None:
        return None

    async def get_actions(self, observation: Any, instruction: str) -> list[dict[str, Any]]:
        return [{"j0.pos": 0.1}]


@pytest.fixture
def hw() -> Any:
    """A ``Robot`` wired to an in-memory arm, with a *recording* connect path.

    ``robot.connects`` counts every ``_connect_robot`` call, which is what makes
    "the refusal precedes the bring-up window" an observation rather than a
    claim. ``robot.robot.sent_actions`` records every command that reached the
    arm.
    """
    robot = HwRobot.__new__(HwRobot)
    robot.tool_name_str = "test_arm"
    robot.action_horizon = 1
    robot.data_config = None
    robot.control_frequency = 50.0
    robot.action_sleep_time = 1.0 / 50.0
    robot._task_state = RobotTaskState()
    robot._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="test_arm_executor")
    robot._shutdown_event = threading.Event()
    robot._stop_requested = threading.Event()
    robot._task_admission = threading.Lock()
    robot._task_claimed = False
    robot.mesh = None
    robot.peer_id = None
    robot.robot = _FakeArm()
    robot.connects = []  # type: ignore[attr-defined]

    async def _connected() -> tuple[bool, str]:
        robot.connects.append(True)  # type: ignore[attr-defined]
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


def _text(result: dict[str, Any]) -> str:
    """The text of a tool-shaped result."""
    return " ".join(block["text"] for block in result["content"] if "text" in block)


class TestUnusablePortRefusedBeforeTheArmIsTouched:
    """The port is judged where the budget already is: before connect, before the claim."""

    @pytest.mark.parametrize("port", UNUSABLE_PORTS, ids=repr)
    def test_execute_task_sync_refuses_without_connecting(self, hw: Any, port: Any) -> None:
        result = hw._execute_task_sync("pick", policy_port=port, duration=0.05)

        assert result["status"] == "error"
        assert "policy_port" in _text(result)
        assert hw.connects == [], "the refused port still ran the bring-up window"
        assert hw._task_claimed is False, "the refused port took the motors bus"
        assert hw.robot.sent_actions == []

    @pytest.mark.parametrize("port", UNUSABLE_PORTS, ids=repr)
    def test_start_task_refuses_instead_of_reporting_a_started_task(self, hw: Any, port: Any) -> None:
        result = hw.start_task("pick", policy_port=port, duration=0.05)

        assert result["status"] == "error"
        assert "policy_port" in _text(result)
        assert "Task started" not in _text(result)
        assert hw._task_state.task_future is None, "work was submitted for a port that cannot build a policy"
        assert hw.connects == []
        assert hw._task_claimed is False

    @pytest.mark.parametrize("entry", ["execute_task", "start_task"])
    def test_a_missing_port_is_reported_as_required(self, hw: Any, entry: str) -> None:
        """``None`` is the "not supplied" spelling, so it says required - early."""
        if entry == "execute_task":
            result = hw._execute_task_sync("pick", policy_port=None, duration=0.05)
        else:
            result = hw.start_task("pick", policy_port=None, duration=0.05)

        assert result["status"] == "error"
        assert "policy_port is required" in _text(result)
        assert entry in _text(result)
        assert hw.connects == []


class TestTheMessageNamesTheValueTheCallerSupplied:
    """A supplied port is not reported as a missing one."""

    @pytest.mark.parametrize("port", [0, False], ids=["0", "False"])
    def test_a_falsy_supplied_port_is_named_not_called_missing(self, hw: Any, port: Any) -> None:
        """Pre-fix these read as absent: falsy, so ``_get_policy`` said "required"."""
        result = hw.start_task("pick", policy_port=port, duration=0.05)

        text = _text(result)
        assert f"invalid policy_port: {port!r}" in text
        assert "is required" not in text

    def test_an_out_of_range_port_names_the_range(self, hw: Any) -> None:
        result = hw.start_task("pick", policy_port=99999, duration=0.05)

        assert "invalid policy_port: 99999 (expected 1-65535)" in _text(result)


class TestABudgetIsStillJudgedFirst:
    """Adding a port check does not reorder the checks that were already there."""

    def test_an_unusable_budget_is_reported_before_the_port(self, hw: Any) -> None:
        result = hw._execute_task_sync("pick", policy_port=99999, duration=0)

        assert "duration must be > 0" in _text(result)
        assert "policy_port" not in _text(result)
        assert hw.connects == []


class TestAPreBuiltPolicyMakesThePortInert:
    """``_execute_task_sync`` does not read the port when a policy is supplied."""

    def test_a_bad_port_is_ignored_when_a_policy_object_is_given(self, hw: Any) -> None:
        result = hw._execute_task_sync("pick", policy_port=99999, policy_object=_Policy(), n_steps=3)

        assert "policy_port" not in _text(result)
        assert hw.connects == [True], "a rollout that never reads the port still has to connect"
        assert len(hw.robot.sent_actions) == 3

    def test_a_missing_port_is_ignored_when_a_policy_object_is_given(self, hw: Any) -> None:
        result = hw._execute_task_sync("pick", policy_object=_Policy(), n_steps=2)

        assert "policy_port" not in _text(result)
        assert len(hw.robot.sent_actions) == 2


class TestAUsablePortStillRuns:
    """The guard refuses exactly the unusable ports and nothing else."""

    @pytest.mark.parametrize("port", USABLE_PORTS, ids=repr)
    def test_a_usable_port_reaches_the_policy_build(self, hw: Any, port: Any) -> None:
        """The port passes the guard, the arm connects, and the rollout runs."""
        result = hw._execute_task_sync("pick", policy_port=port, policy_provider="mock", duration=0.2)

        assert "policy_port" not in _text(result)
        assert hw.connects == [True], "a usable port was refused before connect"
        assert hw.robot.sent_actions, "a usable port did not reach the control loop"

    @pytest.mark.parametrize("port", USABLE_PORTS, ids=repr)
    def test_a_usable_port_still_submits(self, hw: Any, port: Any) -> None:
        result = hw.start_task("pick", policy_port=port, policy_provider="mock", duration=0.2)

        assert result["status"] == "success"
        assert "Task started" in _text(result)
        future = hw._task_state.task_future
        assert future is not None
        future.result(timeout=30)
        assert hw.connects == [True]


class TestTheDomainMatchesTheProviderThatDialsIt:
    """One rule: what the entry point accepts is what the provider accepts."""

    @pytest.mark.parametrize("port", [*UNUSABLE_PORTS, *USABLE_PORTS], ids=repr)
    def test_entry_point_and_shared_domain_agree(self, hw: Any, port: Any) -> None:
        shared_refuses = tcp_port_error(port, "policy_port", "start_task") is not None
        result = hw.start_task("pick", policy_port=port, policy_provider="mock", duration=0.05)
        entry_refuses = "policy_port" in _text(result)

        assert entry_refuses is shared_refuses, f"verdicts differ for policy_port={port!r}"

    def test_the_message_is_the_shared_one_verbatim(self, hw: Any) -> None:
        result = hw.start_task("pick", policy_port=70000, policy_provider="mock", duration=0.05)

        assert _text(result) == tcp_port_error(70000, "policy_port", "start_task")


def _policy_port_surfaces(source: str) -> dict[str, tuple[bool, bool]]:
    """Map every method declaring ``policy_port`` to ``(checks, forwards)``.

    ``checks`` is a call to the guard; ``forwards`` is passing the parameter on
    to another call, which is how the internal relay methods satisfy the rule.
    """
    tree = ast.parse(source)
    found: dict[str, tuple[bool, bool]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = node.args
        names = {a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)}
        if "policy_port" not in names:
            continue
        checks = False
        forwards = False
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if attr == "_policy_port_error":
                checks = True
                continue
            passed = [a for a in call.args if isinstance(a, ast.Name)]
            passed += [k.value for k in call.keywords if isinstance(k.value, ast.Name)]
            if any(a.id == "policy_port" for a in passed):
                forwards = True
        found[node.name] = (checks, forwards)
    return found


def _shipped_surfaces() -> dict[str, tuple[bool, bool]]:
    """:func:`_policy_port_surfaces` applied to the shipped module."""
    return _policy_port_surfaces(pathlib.Path(inspect.getfile(hardware_robot)).read_text(encoding="utf-8"))


class TestEveryPortTakingSurfaceIsAccountedFor:
    """A new task entry point cannot ship without deciding what it does with the port."""

    # The two public/chokepoint entries that must judge the port themselves.
    ENTRY_POINTS = frozenset({"_execute_task_sync", "start_task"})
    # Relays that hand it on unchanged, plus the private builder that is the
    # floor for a direct call.
    RELAYS = frozenset({"_drive_claimed_task", "_run_control_loop", "_execute_task_async"})
    FLOOR = frozenset({"_get_policy"})
    # The rule itself, which takes the value in order to judge it.
    OWNER = frozenset({"_policy_port_error"})

    def test_the_surface_set_is_the_expected_one(self) -> None:
        assert set(_shipped_surfaces()) == self.ENTRY_POINTS | self.RELAYS | self.FLOOR | self.OWNER

    def test_each_entry_point_checks_the_port(self) -> None:
        surfaces = _shipped_surfaces()
        adrift = sorted(name for name in self.ENTRY_POINTS if not surfaces[name][0])
        assert adrift == [], f"entry point(s) not checking policy_port: {adrift}"

    def test_each_relay_hands_the_port_on(self) -> None:
        surfaces = _shipped_surfaces()
        adrift = sorted(name for name in self.RELAYS if not surfaces[name][1])
        assert adrift == [], f"relay(s) not forwarding policy_port: {adrift}"

    def test_the_scanner_sees_an_unchecked_entry_point(self) -> None:
        """Non-vacuity: an entry point that drops the check is reported."""
        planted = (
            "class Robot:\n"
            "    def start_task(self, instruction, policy_port=None):\n"
            "        return self._drive(instruction, policy_port)\n"
        )
        assert _policy_port_surfaces(planted) == {"start_task": (False, True)}
