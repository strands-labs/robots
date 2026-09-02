"""An emergency stop on a simulation halts teleoperation as well as policies.

``SimulationDeviceDriver.onEmergencyStop`` stopped every robot's policy over
``world.robots.values()`` (today via ``request_policy_stop``) and stopped there. A ``Simulation`` mixes in
:class:`~strands_robots.teleop_mixin.TeleopMixin`, so a leader arm can be driving
it from a thread that flag says nothing about: the teleop loop polls
``get_action()`` and applies the result through ``send_action`` on its own
thread. So an operator's emergency stop on a teleoperated simulation reported the
halt and left the leader driving the follower.

The simulation's own ``cleanup`` already stops teleoperation, under the very
guard this handler now uses, because teleoperation is a motion source it cannot
leave running. And both sibling handlers hold an emergency stop to every source
they own: ``reachy_mini_driver.onEmergencyStop`` attempts torque-off AND
stop-motion and logs a failure at CRITICAL, and ``robot_driver.onEmergencyStop``
reads the stop verdict rather than discarding it.

Why the existing suites were silent: the three files that drive this handler
assert the POLICY flag went down (``robot.policy_running is False``) or that a
rogue caller was refused. None of them attaches a teleoperator, so the second
motion source is outside every fixture -- ``grep -c teleop`` over them is 0.

Scope: the ordinary ``stop`` RPC is left alone. Its docstring says "Stop all
running policies", which is what it does; an emergency stop is the one that
promises everything.
"""

import ast
import asyncio
import inspect
import logging
from typing import Any

import pytest

from strands_robots.simulation.models import SimRobot

# The autouse fixture below is what makes this import safe. A sibling test file
# replaces the device_connect_edge submodules with MagicMocks at import time,
# and this helper restores the real ones. Importing the helper does NOT bring the
# sibling's autouse fixture with it -- an autouse fixture is bound to the module
# that declares it -- so this file declares its own.
from tests.test_device_connect_hardening import _force_real_device_connect_edge

_ESTOP_SOURCE = "safety-controller-1"


@pytest.fixture(autouse=True)
def _real_device_connect(monkeypatch):
    """Restore the real extra, clear the allowlists, admit the estop source."""
    _force_real_device_connect_edge()
    for var in ("DEVICE_CONNECT_RPC_ALLOW", "DEVICE_CONNECT_ESTOP_ALLOW", "DEVICE_CONNECT_ALLOW_INSECURE"):
        monkeypatch.delenv(var, raising=False)
    import strands_robots.device_connect.sim_driver as sim_driver

    monkeypatch.setattr(
        sim_driver,
        "is_authorized_caller",
        lambda device_id, scope="rpc": device_id == _ESTOP_SOURCE,
    )


def _run(coro: Any) -> Any:
    """Drive one coroutine to completion."""
    return asyncio.run(coro)


def _Robot(name: str) -> SimRobot:
    """A world robot with a rollout in flight.

    The real record rather than a stand-in: the policy stop is a call on it
    (``request_policy_stop``), not a flag assignment, so a local fake carrying
    only ``policy_running`` would pass this file while the handler failed
    against a real world.
    """
    robot = SimRobot(name=name, urdf_path="")
    robot.policy_running = True
    return robot


class _World:
    def __init__(self, *names: str) -> None:
        self.robots = {n: _Robot(n) for n in names}


class _Sim:
    """A simulation stand-in with BOTH motion sources.

    Models the two things the handler must reach: the per-robot policy flag, and
    the teleop session whose stop answers an envelope. ``stop_teleoperate``
    records that it was called and answers the shape ``TeleopMixin`` really
    produces -- a ``json`` block carrying ``stopped`` -- so a cell can tell a
    reported refusal from a clean stop. A ``MagicMock`` here would answer every
    call with a truthy ``MagicMock`` and carry no verdict to read, which is the
    reason the pre-existing fixtures could not see this.
    """

    def __init__(
        self,
        *,
        robots: tuple[str, ...] = ("arm",),
        teleop_running: bool = True,
        teleop_stops: bool = True,
        teleop_errored_frames: bool = False,
        teleop_raises: BaseException | None = None,
    ) -> None:
        self._world = _World(*robots)
        self._teleop_running = teleop_running
        self._teleops = {"leader": object()} if teleop_running else {}
        self._teleop_stops = teleop_stops
        self._teleop_errored_frames = teleop_errored_frames
        self._teleop_raises = teleop_raises
        self.stop_teleoperate_calls = 0

    def stop_teleoperate(self) -> dict[str, Any]:
        self.stop_teleoperate_calls += 1
        if self._teleop_raises is not None:
            raise self._teleop_raises
        if self._teleop_stops:
            self._teleop_running = False
            if self._teleop_errored_frames:
                # The shape ``_stop_reported_stopped`` exists for: the join was
                # clean, and ``_teleop_stats`` derives "error" from the session
                # counters, so status and the flag disagree.
                return {
                    "status": "error",
                    "content": [
                        {"text": "Teleoperation stopped: 40 frames, 40 errors."},
                        {"json": {"stopped": True, "thread_alive": False, "frames": 40}},
                    ],
                }
            return {
                "status": "success",
                "content": [
                    {"text": "Teleoperation stopped: 40 frames, 0 errors."},
                    {"json": {"stopped": True, "thread_alive": False, "frames": 40}},
                ],
            }
        return {
            "status": "error",
            "content": [
                {"text": "Teleoperation did not stop within 3.0s: the leader is still being polled."},
                {"json": {"stopped": False, "thread_alive": True, "frames": 40}},
            ],
        }


def _driver(sim: _Sim) -> Any:
    from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

    return SimulationDeviceDriver(sim)


class TestTheStopReachesTeleoperationToo:
    """The regression: the second motion source is halted, and reported."""

    def test_a_teleoperated_simulation_has_its_teleop_loop_stopped(self):
        sim = _Sim()
        _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        assert sim.stop_teleoperate_calls == 1, "the teleop loop was left driving the robot"
        assert sim._teleop_running is False

    def test_a_teleop_loop_that_did_not_stop_is_logged_at_critical(self, caplog):
        sim = _Sim(teleop_stops=False)
        with caplog.at_level(logging.DEBUG):
            _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert critical, "a teleop loop still polling its leader was reported below CRITICAL"

    def test_the_critical_record_names_teleoperation_as_the_source(self, caplog):
        sim = _Sim(teleop_stops=False)
        with caplog.at_level(logging.DEBUG):
            _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.CRITICAL)
        assert "teleoperation" in text

    def test_the_critical_record_carries_the_reason_the_stop_gave(self, caplog):
        sim = _Sim(teleop_stops=False)
        with caplog.at_level(logging.DEBUG):
            _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.CRITICAL)
        assert "still being polled" in text

    def test_a_raising_stop_teleoperate_is_reported_rather_than_escaping(self, caplog):
        """A handler that crashes out reports nothing and stops nothing after."""
        sim = _Sim(teleop_raises=RuntimeError("zenoh publisher already closed"))
        with caplog.at_level(logging.DEBUG):
            _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.CRITICAL)
        assert "zenoh publisher already closed" in text

    def test_a_simulation_with_no_world_is_still_asked_to_stop_teleop(self):
        sim = _Sim()
        sim._world = None
        _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        assert sim.stop_teleoperate_calls == 1

    def test_the_receipt_is_a_log_record_not_stdout(self, caplog, capsys):
        """A safety receipt written to stdout carries no level to alert on."""
        sim = _Sim()
        with caplog.at_level(logging.DEBUG):
            _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        assert any(_ESTOP_SOURCE in r.getMessage() for r in caplog.records)
        assert "[estop]" not in capsys.readouterr().out


class TestEverySourceIsAttempted:
    """One failing source must not skip the others.

    The policy-flag cell below holds either way -- pre-fix there was only one
    source, so nothing could be skipped. It is the over-reach guard: adding the
    teleop stop must not cost the policy stop.
    """

    def test_the_policy_flag_still_goes_down_when_the_teleop_stop_raises(self):
        sim = _Sim(teleop_raises=RuntimeError("boom"))
        _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        assert all(r.policy_running is False for r in sim._world.robots.values())

    def test_the_teleop_stop_still_runs_when_the_policy_sweep_raises(self):
        class _Exploding(_Sim):
            @property
            def _world(self):  # type: ignore[override]
                raise RuntimeError("scene torn down mid-estop")

            @_world.setter
            def _world(self, value):  # type: ignore[misc]
                pass

        sim = _Exploding()
        _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        assert sim.stop_teleoperate_calls == 1, "a failing policy sweep skipped the teleop stop"

    def test_the_policy_sweep_precedes_the_bounded_teleop_join(self):
        """Structural: the flag write cannot block, so it lands first.

        ``stop_teleoperate`` joins a thread with a bounded budget. Ordering the
        unblockable kill first means a leader wedged on a serial read cannot
        delay it.
        """
        from strands_robots.device_connect import sim_driver

        handler = _handler_ast(sim_driver)
        policy_line = min(
            node.lineno
            for node in ast.walk(handler)
            if isinstance(node, ast.Call) and "request_policy_stop" in ast.unparse(node)
        )
        teleop_line = min(
            node.lineno
            for node in ast.walk(handler)
            if isinstance(node, ast.Call) and "stop_teleoperate" in ast.unparse(node)
        )
        assert policy_line < teleop_line


class TestWhatIsUnchanged:
    """Controls: the pre-fix contract this must not disturb."""

    def test_an_unauthorized_source_still_stops_nothing(self):
        sim = _Sim()
        _run(_driver(sim).onEmergencyStop("rogue-device", "emergencyStop", {}))
        assert all(r.policy_running is True for r in sim._world.robots.values())
        assert sim.stop_teleoperate_calls == 0

    def test_every_robot_in_the_world_still_has_its_policy_stopped(self):
        sim = _Sim(robots=("arm", "gripper", "base"))
        _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        assert all(r.policy_running is False for r in sim._world.robots.values())

    def test_a_simulation_with_no_teleop_attached_is_not_asked_to_stop_one(self):
        sim = _Sim(teleop_running=False)
        _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        assert sim.stop_teleoperate_calls == 0

    def test_a_teleop_session_whose_frames_errored_is_not_reported_as_still_running(self, caplog):
        """A clean join with unhealthy counters is not a live loop.

        ``_teleop_stats`` derives its status from the session counters, so a
        session whose every frame errored answers ``status="error"`` after a
        perfectly clean join. Keying the verdict on the status rather than on the
        ``stopped`` flag would raise a CRITICAL for a loop that did stop, and a
        false alarm on the safety path trains operators to ignore the warning.
        """
        sim = _Sim(teleop_errored_frames=True)
        with caplog.at_level(logging.DEBUG):
            _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        assert [r for r in caplog.records if r.levelno >= logging.CRITICAL] == []

    def test_a_clean_stop_reports_nothing_at_critical(self, caplog):
        """A false alarm on the safety path trains operators to ignore it."""
        sim = _Sim()
        with caplog.at_level(logging.DEBUG):
            _run(_driver(sim).onEmergencyStop(_ESTOP_SOURCE, "emergencyStop", {}))
        assert [r for r in caplog.records if r.levelno >= logging.CRITICAL] == []


class TestThePremise:
    """The in-tree facts this fix rests on, read from the tree."""

    def test_a_simulation_really_mixes_in_the_teleop_surface(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
        from strands_robots.teleop_mixin import TeleopMixin

        assert issubclass(MuJoCoSimEngine, TeleopMixin)

    def test_the_simulation_stops_teleop_in_its_own_cleanup(self):
        """The guard used here is the one the simulation already uses."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        source = inspect.getsource(MuJoCoSimEngine.cleanup)
        assert "stop_teleoperate" in source
        assert "_teleop_running" in source

    def test_the_shared_reader_grades_the_stopped_flag_not_the_status(self):
        """A status-keyed read would call a healthy teardown a live loop."""
        from strands_robots.teleop_mixin import _stop_reported_stopped

        errored_but_stopped = {
            "status": "error",
            "content": [{"json": {"stopped": True, "frames": 3}}],
        }
        assert _stop_reported_stopped(errored_but_stopped) is True

    def test_the_sibling_handler_escalates_a_failed_stop(self):
        """The accounting precedent, read rather than described."""
        from strands_robots.device_connect import reachy_mini_driver

        source = inspect.getsource(reachy_mini_driver.ReachyMiniDriver.onEmergencyStop)
        assert "logger.critical" in source


class TestTheHandlerConsultsBothSources:
    """Structural: neither source may be dropped silently.

    The discarded-verdict cell below holds either way -- pre-fix there was no
    call to discard. It is the drift guard: a future edit that calls the stop
    and drops its answer would report a halt it did not establish, which is the
    defect ``robot_driver`` carried until it read the verdict.
    """

    def test_the_handler_names_the_teleop_stop(self):
        from strands_robots.device_connect import sim_driver

        handler = _handler_ast(sim_driver)
        assert "stop_teleoperate" in ast.unparse(handler)

    def test_the_teleop_stop_verdict_is_not_discarded(self):
        """A bare-statement call cannot report a loop that kept running."""
        from strands_robots.device_connect import sim_driver

        handler = _handler_ast(sim_driver)
        discarded = [
            ast.unparse(node.value)
            for node in ast.walk(handler)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "stop_teleoperate"
        ]
        assert discarded == [], f"the teleop stop's verdict is discarded: {discarded}"


def _handler_ast(module: Any) -> ast.AsyncFunctionDef:
    """The ``onEmergencyStop`` definition in ``module``."""
    tree = ast.parse(inspect.getsource(module))
    return next(
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "onEmergencyStop"
    )
