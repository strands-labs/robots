"""What a model is told: every rpc described in whole sentences, joints with unit and range, sizes with units,
argument names that say what they take, and only the presets that drive the robot the tool is for."""

from __future__ import annotations

import pytest

from strands_robots import Robot, Simulation
from strands_robots.policy.factory import spec_help
from strands_robots.tools import tool_spec
from strands_robots.tools.call import dispatch
from strands_robots.tools.world import WorldDevice


@pytest.fixture(scope="module")
def specs():
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    props = {
        "robot": tool_spec(arm)["inputSchema"]["json"]["properties"],
        "world": tool_spec(WorldDevice(sim), prefix="sim")["inputSchema"]["json"]["properties"],
    }
    yield arm, props
    sim.close()


def _rpcs(description: str) -> dict[str, str]:
    return dict(part.split(": ", 1) for part in description.split(" | "))


@pytest.mark.parametrize("tool", ["robot", "world"])
def test_every_rpc_is_a_whole_plain_sentence(specs, tool):
    for rpc, text in _rpcs(specs[1][tool]["rpc"]["description"]).items():
        assert text.strip(), f"{rpc}: empty"
        assert text.rstrip().endswith((".", ")")), f"{rpc}: cut mid-sentence: {text!r}"
        assert "``" not in text and "self." not in text, f"{rpc}: RST or self leaks: {text!r}"


def test_act_carries_each_joints_unit_and_range(specs):
    arm, props = specs
    joints = props["robot"]["act"]["properties"]["action"]["properties"]
    for key in arm.action_keys:
        lo, hi = arm.manifest.control(f"{key}.pos").range
        j = joints[key]
        assert j["description"] == "rad" and lo <= j["minimum"] < j["maximum"] <= hi and j["minimum"] - lo < 1e-3
    assert "seconds" in props["robot"]["act"]["description"]  # the rest of the docstring rides on the arguments


def test_add_object_says_size_is_metres_from_the_centre_out(specs):
    text = _rpcs(specs[1]["world"]["rpc"]["description"])["add_object"]
    assert "size m" in text and "half" in text and "10 cm cube" in text


def test_remove_takes_a_name(specs):
    assert specs[1]["world"]["remove"]["required"] == ["name"]


@pytest.mark.parametrize(
    ("robot", "offered", "refused"),
    [
        ("so101", (), ("'holosoma'", "'microduck'", "<preset>")),
        ("unitree_g1", ("'holosoma'",), ("'microduck'",)),
        ("microduck", ("'microduck'",), ("'holosoma'",)),
        (None, ("'holosoma' (unitree_g1)", "'microduck' (microduck)"), ()),
    ],
)
def test_presets_offered_are_those_that_drive_the_robot(robot, offered, refused):
    text = spec_help(robot)
    assert all(o in text for o in offered) and not any(r in text for r in refused), text


def test_the_arm_tool_offers_no_humanoid_preset(specs):
    text = specs[1]["robot"]["run"]["properties"]["policy"]["description"]
    assert "none drives so101" in text and "holosoma" not in text


def test_a_joint_without_a_hard_limit_is_a_number_with_no_bounds():
    sim = Simulation()
    try:
        panda = Robot("panda", sim=sim)  # actuator8 drives a tendon gripper: no range in the manifest
        assert panda.manifest.control("actuator8.pos").range is None
        joints = tool_spec(panda)["inputSchema"]["json"]["properties"]["act"]["properties"]["action"]["properties"]
        assert joints["actuator8"] == {"type": "number", "description": "rad"}
        assert "minimum" in joints["joint1"]
    finally:
        sim.close()


@pytest.mark.parametrize("end", ["minimum", "maximum"])
def test_every_advertised_bound_is_one_the_gate_accepts(end):
    sim = Simulation()
    try:
        arm = Robot("so101", sim=sim)
        joints = tool_spec(arm)["inputSchema"]["json"]["properties"]["act"]["properties"]["action"]["properties"]
        for key in arm.action_keys:  # rounded to nearest, -0.1745 became -0.175: past the hard range, refused
            dispatch(arm, {"rpc": "act", "act": {"action": {key: joints[key][end]}}})
    finally:
        sim.close()
