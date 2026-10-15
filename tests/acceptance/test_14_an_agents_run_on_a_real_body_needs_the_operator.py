"""Check 14: an agent's ``run`` on a real body, with no operator callable, writes nothing to the bus.

The call goes through a real strands ``Agent`` and the robot's generated tool, the way a model calls it.
The body is a feetech driver on a recorded serial line, so a goal write would be visible frame by frame.
The policy spec resolves to "hold where you are", so no checkpoint or network is needed.
"""

from __future__ import annotations

import pytest
from checks import needs_module
from test_drivers import make_driver
from test_operator_gate import goal_writes

HOLD = "acceptance.hold"


@pytest.fixture
def hold_policy(monkeypatch):
    from strands_robots.policy import factory

    make = factory.make
    monkeypatch.setattr(factory, "make", lambda spec, **kw: (lambda obs: {}) if spec == HOLD else make(spec, **kw))


@pytest.mark.parametrize("rpc", ["run", "start"])
def test_an_agents_run_on_a_real_body_needs_the_operator(hold_policy, rpc):
    needs_module("strands", "the agent tools extra")
    needs_module("strands_robots.tools", "part 8: agent tools")
    from strands import Agent

    from strands_robots import Robot

    d, serial = make_driver()
    real = Robot("so101", driver=d, confirm=None, twin=False)
    agent = Agent(tools=[real.tool], callback_handler=None)
    serial.frames.clear()
    result = getattr(agent.tool, real.tool.tool_name)(rpc=rpc, **{rpc: {"policy": HOLD, "steps": 3}})
    if real._task is not None:
        real.stop()
    real.close()
    assert goal_writes(serial) == [], "an agent's run wrote goal frames with confirm=None"
    if rpc == "run":
        assert result["status"] == "error" and "confirm=" in result["content"][0]["text"], result
