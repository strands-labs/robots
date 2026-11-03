"""Gate row 2: a scripted strands Agent turn calls observe then act on a sim so101, and the joint moves."""

from __future__ import annotations

import json

import pytest
from scripted_model import ScriptedModel
from strands import Agent
from strands.types.tools import AgentTool

from strands_robots import Robot


def _results(agent: Agent) -> list[dict]:
    out = []
    for m in agent.messages:
        for block in m["content"]:
            if "toolResult" in block:
                out.append(block["toolResult"])
    return out


def test_scripted_turn_observes_then_acts_and_the_joint_moves():
    arm = Robot("so101")
    before = arm.observe()["shoulder_pan"]
    assert isinstance(arm.tool, AgentTool) and arm.tool.tool_name == "robot_so101"
    model = ScriptedModel(
        [
            ("robot_so101", {"rpc": "observe"}),
            ("robot_so101", {"rpc": "act", "act": {"action": {"shoulder_pan": 0.4}, "steps": 200}}),
        ]
    )
    agent = Agent(model=model, tools=[arm.tool], callback_handler=None)
    reply = agent("move the base joint")
    assert "done" in str(reply)
    observed, acted = _results(agent)
    assert observed["status"] == "success" and acted["status"] == "success", (observed, acted)
    seen = json.loads(observed["content"][0]["text"])
    assert seen["joints"]["shoulder_pan"] == pytest.approx(before, abs=1e-6)
    after = json.loads(acted["content"][0]["text"])["joints"]["shoulder_pan"]
    assert after == pytest.approx(0.4, abs=0.05) and arm.observe()["shoulder_pan"] == pytest.approx(after, abs=1e-3)
    assert model.seen[0][0]["name"] == "robot_so101"


def test_errors_reach_the_model_as_text_not_exceptions():
    arm = Robot("so101")
    agent = Agent(
        model=ScriptedModel(
            [
                ("robot_so101", {"rpc": "act", "act": {"action": {"nope": 1.0}}}),
                ("robot_so101", {"rpc": "fly"}),
                ("robot_so101", {"rpc": "observe", "observe": {"bogus": 1}}),
            ]
        ),
        tools=[arm.tool],
        callback_handler=None,
    )
    agent("try things")
    texts = [(r["status"], r["content"][0]["text"]) for r in _results(agent)]
    assert texts[0][0] == "error" and "nope" in texts[0][1]
    assert texts[1][0] == "error" and "unknown rpc 'fly'" in texts[1][1]
    assert texts[2][0] == "error" and "bogus" in texts[2][1]


def test_an_agent_can_estop_but_never_resume():
    arm = Robot("so101")
    agent = Agent(
        model=ScriptedModel(
            [
                ("robot_so101", {"rpc": "estop"}),
                ("robot_so101", {"rpc": "act", "act": {"action": {"shoulder_pan": 0.4}}}),
                ("robot_so101", {"rpc": "resume", "resume": {"approved_by": "me, the model"}}),
            ]
        ),
        tools=[arm.tool],
        callback_handler=None,
    )
    agent("stop everything")
    (s0, _), (s1, t1), (s2, t2) = [(r["status"], r["content"][0]["text"]) for r in _results(agent)]
    assert s0 == "success" and arm.estopped
    assert s1 == "error" and "estop" in t1.lower()
    assert s2 == "error" and "operator" in t2 and arm.estopped


def test_a_policy_spec_string_runs_the_same_from_python_and_from_the_tool(monkeypatch):
    from strands_robots.policy import factory

    real_make = factory.make
    monkeypatch.setattr(
        factory, "make", lambda s, **kw: (lambda obs: {"elbow_flex": 0.3}) if s == "t.x" else real_make(s)
    )
    arm = Robot("so101")
    assert arm.run("t.x", steps=2).stopped_by == "steps"  # a spec string is a policy in Python too
    (r,) = _results(_agent_after(arm, [("robot_so101", {"rpc": "run", "run": {"policy": "t.x", "steps": 2}})]))
    assert r["status"] == "success" and json.loads(r["content"][0]["text"])["steps"] == 2


def _agent_after(arm, calls):
    agent = Agent(model=ScriptedModel(calls), tools=[arm.tool], callback_handler=None)
    agent("go")
    return agent
