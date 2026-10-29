"""N14: ``driver="memory"`` is a real body with no wire, so the operator gate, approve= and the watchdog's
real-body rule can be shown and tested without an arm (docs, notebooks, CI)."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots import Robot
from strands_robots.robot.drivers import Driver, MemoryDriver
from strands_robots.robot.run.source import from_agent
from strands_robots.robot.safety import RequestRefused, Watchdog


def _real(**kw):
    robot = Robot("so101", driver="memory", twin=False, **kw)
    return robot, robot.body


def test_driver_memory_is_a_real_body_whose_reads_return_its_writes():
    robot, bus = _real()
    assert isinstance(bus, MemoryDriver) and isinstance(bus, Driver) and robot.is_real
    assert bus.names == robot.action_keys and np.all(robot.observe().joints.positions == 0.0)
    robot.act({"shoulder_pan": 0.2})
    assert robot.observe().joints["shoulder_pan"] == pytest.approx(0.2)
    assert len(bus.writes) == 1 and bus.extras() == {"writes": 1, "holds": 0, "estops": 0}


@pytest.mark.parametrize(("confirm", "written"), [(None, 0), (lambda targets: False, 0), (lambda targets: True, 1)])
def test_the_operator_gate_decides_what_reaches_the_memory_bus(confirm, written):
    robot, bus = _real(confirm=confirm)
    if written:
        robot.confirm_act({"shoulder_pan": 0.3})
    else:
        with pytest.raises(RequestRefused):
            robot.confirm_act({"shoulder_pan": 0.3})
    assert len(bus.writes) == written


def test_estop_reaches_the_memory_bus_and_the_watchdog_takes_it_as_hardware():
    robot, bus = _real()
    robot.estop()
    assert bus.estops == 1
    t = [0.0]
    dog = Watchdog(robot, 100, clock=lambda: t[0])
    t[0] = 1.0
    assert dog.poll() and bus.holds == 1  # so101 declares on_disconnect="hold"


@pytest.mark.parametrize(("answer", "written"), [(False, 0), (True, 3)])
def test_approve_decides_whether_an_agent_run_reaches_the_memory_bus(answer, written):
    plans = []
    robot, bus = _real(confirm=lambda targets: True, approve=lambda plan: plans.append(plan) or answer)
    with from_agent():
        if answer:
            robot.run(lambda obs: {"shoulder_pan": 0.1}, steps=3)
        else:
            with pytest.raises(RequestRefused):
                robot.run(lambda obs: {"shoulder_pan": 0.1}, steps=3)
    assert len(plans) == 1 and len(bus.writes) == written


@pytest.mark.parametrize("kw", [{"port": "/dev/ttyACM0"}, {"cameras": {"front": 0}}])
def test_the_memory_driver_refuses_a_port_or_cameras_it_cannot_have(kw):
    with pytest.raises(TypeError, match="memory"):
        Robot("so101", driver="memory", twin=False, **kw)
