"""What a person reads at the safety surface: the run they approve, who pressed stop, where a body is (#4749)."""

from __future__ import annotations

import re

import pytest
from test_drivers import make_driver

from strands_robots import Policy, Robot, Simulation
from strands_robots.robot.run.source import from_agent
from strands_robots.tools.call import dispatch


def sway(obs):
    return None


def hardware(plans: list) -> Robot:
    d, _ = make_driver()
    return Robot("so101", driver=d, twin=False, approve=lambda plan: plans.append(plan) or False)


@pytest.mark.parametrize(
    "policy, shown",
    [(sway, "sway"), (Policy(sway), "sway"), (lambda obs: None, "<lambda>")],
    ids=["function", "callable-policy", "lambda"],
)
def test_the_run_plan_names_the_policy_without_a_memory_address(policy, shown):
    plans: list = []
    real = hardware(plans)
    with from_agent(), pytest.raises(PermissionError, match=f"policy='{shown}'"):
        real.run(policy, hz=30, steps=3)
    assert plans[0].policy == shown and " at 0x" not in str(plans[0])
    real.close()


@pytest.mark.parametrize("tool_input", [{"policy": "wave", "seconds": 0.1}, {"seconds": 0.1}], ids=["named", "only"])
def test_an_agents_run_plan_names_the_handle_the_model_used(tool_input):
    plans: list = []
    real = hardware(plans)
    with from_agent(), pytest.raises(PermissionError, match="policy='wave'"):
        dispatch(real, {"rpc": "run", "run": tool_input}, policies={"wave": lambda obs: None})
    assert [p.policy for p in plans] == ["wave"]
    real.close()


def test_from_agent_is_a_name_on_the_robot_package():
    import strands_robots.robot

    assert strands_robots.robot.from_agent is from_agent


@pytest.mark.parametrize("by, said", [("alice", "engaged by alice (estop())"), (None, "engaged (estop())")])
def test_an_estop_refusal_says_who_pressed_it(by, said):
    arm = Robot("so101", sim=Simulation())
    arm.estop(by=by)
    with pytest.raises(PermissionError, match=re.escape(f"e-stop is {said}")):
        arm.act({"shoulder_pan": 0.1})
    arm.sim.close()
