"""Gate row 4: a real (fake-bus) Robot refuses an agent's act without the operator's callable; with it, passes.

A tool argument can never stand in for the operator: ``confirm``/``confirmed`` are not arguments of the tool.
"""

from __future__ import annotations

import json

import pytest
from scripted_model import ScriptedModel
from strands import Agent
from test_drivers import make_driver
from test_operator_gate import goal_writes


def _turn(robot, calls):
    agent = Agent(model=ScriptedModel(calls), tools=[robot.tool], callback_handler=None)
    agent("go")
    return [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]


def _act(target: float, **extra):
    return ("robot_so101", {"rpc": "act", "act": {"action": {"shoulder_pan": target}, **extra}})


def test_without_the_operator_callable_an_agent_act_is_refused_and_writes_nothing():
    from strands_robots import Robot

    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False)
    assert real.is_real
    before = len(goal_writes(serial))
    (r,) = _turn(real, [_act(0.2)])
    assert r["status"] == "error" and "confirm=" in r["content"][0]["text"]
    assert len(goal_writes(serial)) == before


def test_the_models_own_claim_is_not_a_confirmation():
    from strands_robots import Robot

    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False)
    before = len(goal_writes(serial))
    r1, r2 = _turn(real, [_act(0.2, confirmed=True), _act(0.2, confirm=True)])
    for r in (r1, r2):
        assert r["status"] == "error" and "does not take" in r["content"][0]["text"], r
    assert len(goal_writes(serial)) == before


@pytest.mark.parametrize("answer", [True, False], ids=["operator-yes", "operator-no"])
def test_with_the_operator_callable_the_operator_decides(answer):
    from strands_robots import Robot

    asked = []
    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False, confirm=lambda targets: asked.append(targets) or answer)
    before = len(goal_writes(serial))
    (r,) = _turn(real, [_act(0.2)])
    assert len(asked) == 1
    if answer:
        assert r["status"] == "success", r
        assert json.loads(r["content"][0]["text"])["joints"]["shoulder_pan"] == pytest.approx(0.2, abs=1e-2)
        assert len(goal_writes(serial)) > before
    else:
        assert r["status"] == "error" and "refused" in r["content"][0]["text"]
        assert len(goal_writes(serial)) == before


def _run(steps: int = 3):
    return ("robot_so101", {"rpc": "run", "run": {"policy": "tests.hold", "steps": steps}})


@pytest.fixture
def hold_policy(monkeypatch):
    """A policy spec the factory resolves to 'hold where you are' (no checkpoint, no network)."""
    from strands_robots.policy import factory

    real_make = factory.make
    monkeypatch.setattr(
        factory, "make", lambda spec, **kw: (lambda obs: {}) if spec == "tests.hold" else real_make(spec)
    )


@pytest.mark.parametrize("rpc", ["run", "start"])
def test_an_agent_run_on_a_real_robot_goes_through_the_gate_every_step(hold_policy, rpc):
    from strands_robots import Robot

    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False)
    before = len(goal_writes(serial))
    call = _run()
    if rpc == "start":
        call = ("robot_so101", {"rpc": "start", "start": call[1]["run"]})
    (r,) = _turn(real, [call])
    if real._task is not None:
        real.stop()
    assert len(goal_writes(serial)) == before, "an agent's run wrote goal frames with confirm=None"
    if rpc == "run":
        assert r["status"] == "error" and "confirm=" in r["content"][0]["text"]


def test_with_the_operator_callable_an_agent_run_asks_every_step(hold_policy):
    from strands_robots import Robot

    asked = []
    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False, confirm=lambda t: asked.append(t) or True)
    (r,) = _turn(real, [_run(3)])
    assert r["status"] == "success", r
    assert json.loads(r["content"][0]["text"])["stopped_by"] == "steps" and len(asked) == 3


def test_a_background_start_inside_an_agent_call_carries_the_gate_to_its_thread():
    from strands_robots import Robot
    from strands_robots.robot.run.source import from_agent

    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False)
    before = len(goal_writes(serial))
    with from_agent():
        task = real.start(lambda obs: {}, steps=3)
    from strands_robots.robot.safety import RequestRefused

    with pytest.raises(RequestRefused, match="confirm="):
        task.stop()  # the refusal is loud: it re-raises where the task is collected
    assert len(goal_writes(serial)) == before


@pytest.mark.parametrize("rpc", ["calibrate", "resume"])
def test_calibrate_and_resume_are_the_operators_and_never_reach_the_bus(rpc):
    from strands_robots import Robot

    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False)
    frames = len(serial.frames)
    (r,) = _turn(real, [("robot_so101", {"rpc": rpc})])
    assert r["status"] == "error" and "operator" in r["content"][0]["text"]
    assert len(serial.frames) == frames  # not a byte on the bus: no torque off, no offsets rewritten
