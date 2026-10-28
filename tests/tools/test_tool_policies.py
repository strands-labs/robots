"""``robot.tool(policies=[...])``: the developer registers policy handles; the model runs them by name, not address."""

from __future__ import annotations

import numpy as np
import pytest
from scripted_model import ScriptedModel
from strands import Agent
from test_drivers import make_driver
from test_operator_gate import goal_writes

from strands_robots import Robot
from strands_robots.tools.agent import handles


def _turn(tool, calls):
    agent = Agent(model=ScriptedModel(calls), tools=[tool], callback_handler=None)
    agent("go")
    return [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]


def _run(**args):
    return ("robot_so101", {"rpc": "run", "run": {"steps": 3, **args}})


class _Named:
    name = "nudge"

    def __init__(self):
        self.calls = 0

    def act(self, obs, *, instruction=None):
        self.calls += 1
        return np.asarray(obs.joints.positions) + 0.01

    def reset(self):
        pass


def test_handles_are_named_by_the_mapping_the_name_or_the_deployment():
    pytest.importorskip("lerobot")
    from strands_robots import Policy

    far = Policy("zenoh://zenoh.example:7447/d_ttt")
    assert list(handles([_Named(), far, lambda obs: obs.joints.positions])) == ["nudge", "d_ttt", "policy3"]
    assert list(handles({"ttt": far})) == ["ttt"]
    with pytest.raises(TypeError, match=r"policies\['x'\] must be a policy object"):
        handles({"x": "zenoh://h:7447/d"})


def test_two_handles_with_one_name_are_refused_not_collapsed():
    pytest.importorskip("lerobot")
    from strands_robots import Policy

    a, b = Policy("zenoh://host-a:7447/d_pick"), Policy("zenoh://host-b:7447/d_pick")
    with pytest.raises(ValueError, match=r"two handles are both named 'd_pick'.*pass a \{name: policy\} mapping"):
        handles([a, b])
    with pytest.raises(ValueError, match=r"two handles are both named 'nudge'"):
        handles([_Named(), _Named()])
    assert list(handles({"pick_a": a, "pick_b": b})) == ["pick_a", "pick_b"]


def test_the_tool_is_the_same_tool_with_handles_and_the_spec_names_them():
    arm = Robot("so101")
    plain, with_one = arm.tool, arm.tool(policies=[_Named()])
    assert type(plain) is type(with_one) and plain.tool_name == with_one.tool_name == "robot_so101"
    run = with_one.tool_spec["inputSchema"]["json"]["properties"]["run"]
    assert (
        "Registered policies, by name: nudge (the only one: may be left out)"
        in run["properties"]["policy"]["description"]
    )
    assert (
        "policy" not in run.get("required", [])
        and "policy" in plain.tool_spec["inputSchema"]["json"]["properties"]["run"]["required"]
    )
    two = arm.tool(policies={"a": _Named(), "b": _Named()}).tool_spec["inputSchema"]["json"]["properties"]["run"]
    assert "a, b" in two["properties"]["policy"]["description"] and "policy" in two["required"]


def test_an_agent_runs_a_registered_handle_by_name_or_the_only_one_unnamed():
    arm = Robot("so101")
    pol = _Named()
    r1, r2 = _turn(arm.tool(policies=[pol]), [_run(policy="nudge"), _run()])
    assert r1["status"] == "success" and r2["status"] == "success", (r1, r2)
    assert pol.calls == 6  # three steps each, through the same object the developer built
    (r3,) = _turn(arm.tool(policies={"a": _Named(), "b": _Named()}), [_run()])
    assert r3["status"] == "error" and "one of the registered ['a', 'b']" in r3["content"][0]["text"]


def test_a_remote_handle_runs_on_a_real_arm_through_the_approved_gate_but_a_typed_address_never(monkeypatch):
    """The developer's ``Policy("zenoh://…")`` is a handle like any other: the operator approves the run once (K6)
    and every step passes the chain. The model typing the same address is refused before ``attach``."""
    pytest.importorskip("lerobot")
    from strands_robots import Policy
    from strands_robots.policy.lerobot import remote

    attached: list[str] = []
    monkeypatch.setattr(remote.ZenohPolicy, "attach", lambda self, robot: attached.append(robot.id))
    monkeypatch.setattr(remote.ZenohPolicy, "detach", lambda self: None)
    monkeypatch.setattr(remote.ZenohPolicy, "act", lambda self, obs, instruction=None: np.asarray(obs.joints.positions))
    monkeypatch.setattr(
        remote.ZenohPolicy, "requirements", lambda self: __import__("strands_robots").policy.Requirements()
    )
    monkeypatch.setattr(remote.ZenohPolicy, "units_for", lambda self, d: None)
    d, serial = make_driver()
    plans = []
    real = Robot("so101", driver=d, twin=False, confirm=lambda t: True, approve=lambda plan: plans.append(plan) or True)
    ttt = Policy("zenoh://zenoh.example:7447/d_ttt")
    (ok, refused) = _turn(
        real.tool(policies=[ttt]),
        [_run(policy="d_ttt", instruction="pick up an x"), _run(policy="zenoh://zenoh.example:7447/d_ttt")],
    )
    assert ok["status"] == "success", ok
    assert attached == ["so101"] and len(goal_writes(serial)) >= 3 and plans[0].instruction == "pick up an x"
    assert refused["status"] == "error" and "developer" in refused["content"][0]["text"]
    assert "registered: ['d_ttt']" in refused["content"][0]["text"] and attached == ["so101"]
