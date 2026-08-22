# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A rollout and the mesh's readers share one conversation on the motor bus.

:mod:`strands_robots.bus_access` puts one ``RLock`` on the device so every
reader and writer that goes *through it* takes the serial motor bus in turn.
The mesh modules were converted to it; :mod:`strands_robots.hardware_robot` was
not, and imported it nowhere -- it drove the same device directly at five call
sites (the ROS 2 telemetry publish, the policy preflight read, the rollout
loop's read and write, and the ``send_action`` facade that teleop and inbound
ROS 2 commands both come through).

So the lock serialised the mesh's readers against *each other*, which is what it
was built and tested for, and not against a policy rollout -- while one process
holds both. ``hardware_robot`` has its own ``threading.Lock`` (``_task_admission``)
that admits one rollout at a time, but that is admission control over *tasks*:
the two locks do not know about each other, and neither one stops a mesh sensor
loop and a rollout from reaching for the bus at the same instant.

The failure is asymmetric, and the caller that loses is the one not holding the
lock. Measured against the real ``run_policy`` loop with a mesh reader polling
the same device, and a bus double that refuses overlap exactly as the feetech
SDK does::

    before: status='error'  steps=0   commands=0   mesh reads=54  (unharmed)
            'Policy rollout error: 0 steps in 0.0s'
            'Error: Failed to initialize policy'
    after:  status='success' steps=20  commands=20  mesh reads=85

One refused read is enough, and either of the rollout's two reads will do it.
The preflight read in ``_initialize_policy`` is its first touch of the bus, and
that method's ``except Exception`` turns a refusal into ``return False``, so the
rollout is reported as a *policy* fault -- ``Failed to initialize policy`` -- for
a run that never commanded the arm once. Lose the race one step later instead and
the loop's read propagates, so the rollout ends with the SDK's own
``Port is in use!`` after a step or two. Which one loses varies with the timing;
that the rollout ends does not. The mesh reader, holding the lock, never notices.

These tests pin both halves: the behaviour, against the real rollout; and the
root cause, as a package-wide rule that no caller reaches an operation
``bus_access`` owns through a device it holds. That rule needs no list of
exempt files -- it grades the three modules that hold a lerobot device
(``hardware_robot`` and the mesh's ``core`` and ``input``), and the mesh two
already pass.

No serial port is opened and no arm is commanded: the bus double is the one the
:mod:`bus_access` tests already use, extended with the connect surface a rollout
drives.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

import strands_robots
from strands_robots.bus_access import read_observation
from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from strands_robots.policies.base import Policy
from tests._daemon_executor import DaemonThreadExecutor

from .test_bus_access_serializes_motor_reads import RefusingBusRobot

#: Upper bound on any wait, so a broken contract fails instead of hanging.
DEADLINE = 20.0

#: Steps the measured rollout runs. Long enough that a mesh reader polling
#: throughout overlaps it many times over, short enough to stay fast.
STEPS = 20

#: Package root, for the source-level rule.
PACKAGE = pathlib.Path(strands_robots.__file__).parent


class ConnectableRefusingBus(RefusingBusRobot):
    """The bus double from the :mod:`bus_access` tests, plus a connect surface.

    Subclassed rather than rewritten so both files agree on what an overlap
    costs: the refusal, its message and the concurrency audit all come from the
    shipped double, and only the attributes a rollout's bring-up reads are added.
    """

    def __init__(self, *, read_seconds: float = 0.01) -> None:
        super().__init__(read_seconds=read_seconds)
        self.name = "arm"
        self.robot_type = "arm"
        self.is_calibrated = True
        self.config = SimpleNamespace(cameras={})
        self._connected = False
        self.connect_calls = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = False) -> None:  # noqa: ARG002 - driver signature
        self.connect_calls += 1
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False


class OneJointPolicy(Policy):
    """Commands one declared joint every step."""

    @property
    def provider_name(self) -> str:
        return "one_joint"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        pass

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"shoulder_pan.pos": 0.5}] * 2


def make_robot(bus: Any) -> HwRobot:
    """Build a Robot around ``bus``, bypassing hardware init (the tests/ pattern).

    The rollout executor is :class:`tests._daemon_executor.DaemonThreadExecutor`
    because a test here can abandon a work item (``future.result(timeout=...)``):
    a ``ThreadPoolExecutor`` worker is not a daemon and the interpreter joins it
    at exit, so a wedged item would deliver a failing test as a hung job.
    """
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "arm"
    hw.action_horizon = 2
    hw.data_config = None
    hw.control_frequency = 200.0
    hw.action_sleep_time = 1.0 / 200.0
    hw._task_state = RobotTaskState()
    hw._executor = DaemonThreadExecutor(thread_name_prefix="arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = bus
    return hw


@pytest.fixture
def bus() -> ConnectableRefusingBus:
    return ConnectableRefusingBus()


@pytest.fixture
def hw(bus: ConnectableRefusingBus) -> Any:
    robot = make_robot(bus)
    yield robot
    robot.cleanup()


class _MeshReader:
    """A mesh probe: reads the same device through ``bus_access`` in a loop."""

    def __init__(self, device: Any, *, period: float = 0.002) -> None:
        self._device = device
        self._period = period
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reads = 0
        self.failures = 0

    def __enter__(self) -> _MeshReader:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=DEADLINE)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                read_observation(self._device)
                self.reads += 1
            except Exception:  # noqa: BLE001 - a refusal is the measurement
                self.failures += 1
            time.sleep(self._period)


class TestARolloutSharesTheBusWithTheMeshsReaders:
    """The seam the two locks left open, driven through the real rollout."""

    def test_a_rollout_completes_beside_a_mesh_reader(self, hw: HwRobot, bus: ConnectableRefusingBus) -> None:
        """The whole rollout runs: every step commanded, no refusal on the bus."""
        with _MeshReader(hw.robot) as mesh:
            result = hw.run_policy(policy_object=OneJointPolicy(), instruction="reach", n_steps=STEPS)

        assert mesh.reads > 0, "premise: the mesh reader never got a turn on the bus"
        assert result.get("status") == "success", result
        assert hw._task_state.step_count == STEPS, result
        assert bus.writes == STEPS, f"commands that reached the wire: {bus.writes} of {STEPS}"
        assert bus.refusals == 0, f"the bus refused {bus.refusals} overlapping transaction(s)"

    def test_a_commanded_action_is_delivered_while_the_mesh_reads(
        self, hw: HwRobot, bus: ConnectableRefusingBus
    ) -> None:
        """The ``send_action`` facade -- teleop and inbound ROS 2 both come here."""
        bus.connect()
        writers, readers = 3, 3
        gate = threading.Barrier(writers + readers)
        verdicts: list[dict[str, Any]] = []

        def write() -> None:
            gate.wait(DEADLINE)
            verdicts.append(hw.send_action({"shoulder_pan.pos": 0.5}))

        def read() -> None:
            gate.wait(DEADLINE)
            read_observation(hw.robot)

        with ThreadPoolExecutor(max_workers=writers + readers) as pool:
            list(pool.map(lambda fn: fn(), [write] * writers + [read] * readers))

        assert [v.get("status") for v in verdicts] == ["success"] * writers, verdicts
        assert bus.writes == writers, f"commands that reached the wire: {bus.writes} of {writers}"
        assert bus.refusals == 0, f"the bus refused {bus.refusals} overlapping transaction(s)"

    def test_the_telemetry_publish_shares_the_bus(
        self, hw: HwRobot, bus: ConnectableRefusingBus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``publish_ros_observation`` reads the arm to publish it."""
        bus.connect()
        hw._ros_bridge = object()
        hw._ros2_domain = 0
        monkeypatch.setattr(hw, "_publish_ros_telemetry", lambda *a, **k: None)

        with _MeshReader(hw.robot):
            gate = threading.Barrier(2)

            def publish() -> dict[str, Any]:
                gate.wait(DEADLINE)
                return hw.publish_ros_observation()

            def read() -> None:
                gate.wait(DEADLINE)
                read_observation(hw.robot)

            with ThreadPoolExecutor(max_workers=2) as pool:
                published = pool.submit(publish)
                pool.submit(read)
                verdict = published.result(timeout=DEADLINE)

        assert verdict.get("status") == "success", verdict
        assert bus.refusals == 0, f"the bus refused {bus.refusals} overlapping transaction(s)"

    def test_the_policy_preflight_read_shares_the_bus(self, hw: HwRobot, bus: ConnectableRefusingBus) -> None:
        """The rollout's FIRST touch of the bus, and the one that aborted it."""
        bus.connect()
        with _MeshReader(hw.robot):
            ready = asyncio.run(hw._initialize_policy(OneJointPolicy()))

        assert ready is True, "the preflight read was refused, so the rollout reports a policy fault"
        assert bus.refusals == 0, f"the bus refused {bus.refusals} overlapping transaction(s)"


class TestNoCallerDrivesAHeldDeviceOutsideBusAccess:
    """The root cause, as a rule over the source rather than one call site.

    Derived twice over, so neither half is a list that can fall out of step: the
    guarded operations come from what :mod:`bus_access` itself calls on a device,
    and the graded population is every module that holds one.
    """

    def test_no_caller_drives_a_held_device_outside_bus_access(self) -> None:
        offenders = _direct_device_touches()
        assert offenders == [], (
            "these reach an operation bus_access owns through a device the module holds, "
            "so they take no bus lock and can collide with every other reader: "
            + ", ".join(f"{path}:{line} ({expr})" for path, line, expr in offenders)
        )

    def test_the_guarded_operations_are_derived_from_bus_access(self) -> None:
        """A derivation that lost the two public wrappers would grade nothing."""
        operations = _owned_bus_operations()
        assert {"get_observation", "send_action"} <= operations, operations

    def test_the_scan_reaches_every_module_that_holds_a_device(self) -> None:
        """A scan that reached no device holder would report a clean tree."""
        holders = _device_holding_modules()
        assert len(holders) >= 3, holders
        assert any(path.name == "hardware_robot.py" for path in holders), holders
        assert any(path.parent.name == "mesh" for path in holders), holders


class TestNothingElseChanges:
    """What the rule does not claim, and what must keep working."""

    def test_a_rollout_alone_is_unchanged(self, hw: HwRobot, bus: ConnectableRefusingBus) -> None:
        """With nothing else on the bus, the rollout is what it always was."""
        result = hw.run_policy(policy_object=OneJointPolicy(), instruction="reach", n_steps=STEPS)

        assert result.get("status") == "success", result
        assert hw._task_state.step_count == STEPS
        assert bus.writes == STEPS
        assert bus.refusals == 0

    def test_the_meshs_own_readers_still_take_turns(self, bus: ConnectableRefusingBus) -> None:
        """The contract bus_access already had: readers of one device serialise."""
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: read_observation(bus), range(6)))

        assert bus.reads == 6
        assert bus.refusals == 0

    def test_connect_is_not_claimed_by_this_rule(self) -> None:
        """bus_access wraps reads and writes; opening the port is not its call.

        ``self.robot.connect(...)`` stays direct, and the rule says nothing about
        it -- which is why the operation set is derived from what ``bus_access``
        drives rather than from a list of method names that look bus-shaped.
        """
        assert "connect" not in _owned_bus_operations()
        source = (PACKAGE / "hardware_robot.py").read_text(encoding="utf-8")
        assert "self.robot.connect" in source

    def test_a_real_bus_failure_still_reaches_the_caller(self, hw: HwRobot) -> None:
        """Taking the lock must not swallow what the device reports."""

        class Broken(ConnectableRefusingBus):
            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                raise ConnectionError("serial port disappeared")

        hw.robot = Broken()
        hw.robot.connect()

        verdict = hw.send_action({"shoulder_pan.pos": 0.5})

        assert verdict.get("status") == "error", verdict
        assert "serial port disappeared" in str(verdict)


def _owned_bus_operations() -> set[str]:
    """Every operation :mod:`bus_access` drives on the device it is handed.

    Read off that module rather than restated here, so an operation it learns to
    serialise later is graded without touching this file.
    """
    tree = ast.parse((PACKAGE / "bus_access.py").read_text(encoding="utf-8"))
    operations: set[str] = set()
    for function in [node for node in tree.body if isinstance(node, ast.FunctionDef)]:
        # The device parameter, plus any name bound from it (``bus = getattr(device, "bus", ...)``).
        held = {argument.arg for argument in function.args.args}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                is_getattr = ast.unparse(node.value.func) == "getattr"
                if is_getattr and node.value.args and ast.unparse(node.value.args[0]) in held:
                    held.update(t.id for t in node.targets if isinstance(t, ast.Name))
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in held:
                operations.add(node.attr)
    return operations


def _device_holding_modules() -> list[pathlib.Path]:
    """Every package module that assigns ``self.robot`` -- i.e. holds a device."""
    holders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a module that does not parse
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and ast.unparse(node) == "self.robot"
            ):
                holders.append(path)
                break
    return holders


def _direct_device_touches() -> list[tuple[str, int, str]]:
    """Owned bus operations reached through a device the module holds.

    Matched on the expression rather than on a file list: reaching
    ``get_observation`` through ``self.robot`` is driving the wrapped hardware
    bus, while a simulation backend's identically named method drives no serial
    port at all and is never in scope.
    """
    operations = _owned_bus_operations()
    touches: list[tuple[str, int, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "bus_access.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a module that does not parse
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in operations:
                continue
            holder = ast.unparse(node.value)
            if holder == "self.robot" or holder.startswith("self.robot."):
                touches.append((str(path.relative_to(PACKAGE.parent)), node.lineno, ast.unparse(node)))
    return sorted(touches)
