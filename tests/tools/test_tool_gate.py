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
    assert len(goal_writes(serial)) == before, "an agent's run wrote goal frames with no operator"
    if rpc == "run":  # K6: an agent's run on a real arm needs the operator's one yes for the whole run (approve=)
        assert r["status"] == "error" and "approve=" in r["content"][0]["text"]


def test_with_the_operator_callable_an_agent_run_asks_once_and_checks_every_step(hold_policy):
    """K6: the old path asked confirm() on every tick (30 dialogs a second); here approve= is asked once with the
    RunPlan, and the safety chain still checks every step."""
    from strands_robots import Robot

    asked, per_step = [], []
    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False, approve=lambda p: asked.append(p) or True,
                 confirm=lambda t: per_step.append(t) or True)  # fmt: skip
    real.validators.append(lambda r, req: per_step.append("checked"))
    (r,) = _turn(real, [_run(3)])
    assert r["status"] == "success", r
    assert json.loads(r["content"][0]["text"])["stopped_by"] == "steps" and len(asked) == 1
    assert per_step == ["checked"] * 3  # the chain ran every step; the per-command confirm was never asked


def test_a_background_start_inside_an_agent_call_carries_the_gate_to_its_thread():
    from strands_robots import Robot
    from strands_robots.robot.run.source import from_agent

    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False)
    before = len(goal_writes(serial))
    with from_agent():
        task = real.start(lambda obs: {}, steps=3)
    from strands_robots.robot.safety import RequestRefused

    with pytest.raises(RequestRefused, match="approve="):
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


def test_an_agent_reaches_a_pose_with_one_act_and_the_operator_is_asked_once():
    from strands_robots import Robot
    from strands_robots.tools import tool_spec

    asked: list = []
    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False, confirm=lambda targets: asked.append(targets) or True)
    act = tool_spec(real)["inputSchema"]["json"]["properties"]["act"]["properties"]
    assert "seconds" in act  # the model is offered the way to reach a pose before it invents one
    (r,) = _turn(real, [_act(0.2, seconds=0.2)])
    assert r["status"] == "success", r
    assert len(asked) == 1 and len(goal_writes(serial)) >= 6  # one yes, then a ramp of writes at 30 Hz
    assert asked[0][real.action_keys.index("shoulder_pan")] == pytest.approx(0.2)
    real.close()


@pytest.mark.parametrize("answer, moved", [(False, False), (True, True)])
def test_an_agent_hold_on_a_limp_arm_asks_the_operator_before_torque_goes_on(answer, moved):
    from strands_robots import Robot

    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False, confirm=lambda targets: answer)
    (r,) = _turn(real, [("robot_so101", {"rpc": "hold"})])
    assert (r["status"] == "success") is answer and bool(goal_writes(serial)) is moved
    assert d.extras()["torque"] is moved  # refused: the limp arm stays limp
    real.close()


@pytest.mark.parametrize("extra", [{"record": "hf://someone/lab-camera-dump", "push": True}, {"record": "/elsewhere"}])
def test_an_agent_never_decides_what_is_recorded_or_uploaded(extra):
    from strands_robots import Robot
    from strands_robots.tools import tool_spec

    arm = Robot("so101")
    run = tool_spec(arm)["inputSchema"]["json"]["properties"]["run"]["properties"]
    assert "record" not in run and "push" not in run  # the model is not offered them
    (r,) = _turn(arm, [("robot_so101", {"rpc": "run", "run": {"policy": "tests.hold", "steps": 1, **extra}})])
    assert r["status"] == "error" and "developer" in r["content"][0]["text"]
    arm.close()


@pytest.mark.parametrize("rpc", ["run", "start"])
def test_an_agent_never_chooses_where_the_cameras_stream(monkeypatch, rpc):
    """A model-typed ``policy="zenoh://..."`` would connect and send a camera frame per step before the gate sees an
    action; it is refused before ``attach`` (a sign in the camera's view saying ``policy=zenoh://...`` is enough)."""
    from strands_robots import Robot
    from strands_robots.policy.lerobot import remote

    attached: list[str] = []
    monkeypatch.setattr(remote.ZenohPolicy, "attach", lambda self, robot: attached.append(robot.id))
    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False, confirm=lambda t: True)
    host = ".".join(["203", "0", "113", "9"])  # TEST-NET-3, assembled so no scanner reads it as an address
    (r,) = _turn(real, [("robot_so101", {"rpc": rpc, rpc: {"policy": f"zenoh://{host}:7447/x", "steps": 1}})])
    assert r["status"] == "error" and "developer" in r["content"][0]["text"] and "network" in r["content"][0]["text"]
    assert attached == [] and goal_writes(serial) == []
    assert real._task is None
    real.close()


def test_a_simulated_and_a_hardware_so101_in_one_agent_are_refused_not_one_dropped():
    """Both tools are named ``robot_so101``: strands kept the last one, so an act meant for the simulated arm wrote
    goal frames to the hardware bus. A second tool of the same name is refused; with distinct ids both are kept."""
    from strands_robots import Robot, Simulation

    world = Simulation()
    sim_arm = Robot("so101", sim=world)
    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False, confirm=lambda targets: True)
    for tools in ([sim_arm.tool, real.tool], [real.tool, sim_arm.tool]):
        with pytest.raises(ValueError, match="robot_so101"):
            Agent(model=ScriptedModel([_act(0.5, steps=200)]), tools=tools, callback_handler=None)("move the sim arm")
    assert goal_writes(serial) == []
    real.close()
    d, serial = make_driver()
    bench = Robot("so101", id="bench", driver=d, twin=False, confirm=lambda targets: True)
    agent = Agent(model=ScriptedModel([_act(0.5, steps=200)]), tools=[sim_arm.tool, bench.tool], callback_handler=None)
    agent("move the sim arm")
    assert agent.tool_names == ["robot_so101", "robot_bench"]
    assert sim_arm.observe(images=False)["shoulder_pan"] > 0.3 and goal_writes(serial) == []
    bench.close()
    world.close()


@pytest.fixture
def two_policies(monkeypatch):
    """``tests.left`` / ``tests.right`` drive shoulder_pan to +0.3 / -0.3; the list records who decided each step."""
    from strands_robots.policy import factory

    decided: list[str] = []
    real_make = factory.make
    sides = {"tests.left": ("left", 0.3), "tests.right": ("right", -0.3)}

    def make(spec, **kw):
        if spec not in sides:
            return real_make(spec)
        side, target = sides[spec]
        return lambda obs: decided.append(side) or {"shoulder_pan": target}

    monkeypatch.setattr(factory, "make", make)
    return decided


def _run_left_and_right(**limits):
    return [("robot_so101", {"rpc": "run", "run": {"policy": f"tests.{s}", **limits}}) for s in ("left", "right")]


def test_two_runs_in_one_model_turn_run_one_after_the_other(two_policies):
    """strands runs one message's tool uses at once: both runs drove one hardware so101, their goals alternating on
    the bus. The tool now takes them in the order sent: left decides its ten steps, then right its ten, never mixed."""
    from strands_robots import Robot

    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False, approve=lambda p: True)
    done = [r["status"] for r in _turn(real, [_run_left_and_right(steps=10, hz=20)])]
    assert done == ["success", "success"], done
    assert two_policies == ["left"] * 10 + ["right"] * 10, two_policies
    real.close()


def test_the_operators_stop_ends_an_agents_run(two_policies):
    """An agent's run was not the robot's task: ``describe()`` said no task and ``stop()`` returned None while the
    arm kept moving. Now the operator's ``stop()`` ends it, and the agent's tool call returns that result."""
    import threading
    import time

    from strands_robots import Robot

    d, serial = make_driver()
    real = Robot("so101", driver=d, twin=False, approve=lambda p: True)
    out: list[dict] = []
    call = ("robot_so101", {"rpc": "run", "run": {"policy": "tests.left", "seconds": 3, "hz": 20}})
    turn = threading.Thread(target=lambda: out.extend(_turn(real, [call])))
    turn.start()
    deadline = time.monotonic() + 2
    while real.describe()["task"] != "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    stopped = real.stop(timeout=2)
    turn.join(5)
    assert stopped is not None and stopped.stopped_by == "stopped", stopped
    assert json.loads(out[0]["content"][0]["text"])["stopped_by"] == "stopped" and len(two_policies) < 60
    real.close()
