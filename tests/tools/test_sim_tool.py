"""Gate row 5: ``sim.tool`` covers the world verbs, generated from the Simulation's public methods."""

from __future__ import annotations

import json

from scripted_model import ScriptedModel
from strands import Agent
from strands.types.tools import AgentTool

from strands_robots import Robot
from strands_robots.simulation import Simulation


def _turn(tool, calls):
    agent = Agent(model=ScriptedModel(calls), tools=[tool], callback_handler=None)
    agent("build a scene")
    return [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]


def test_sim_tool_spec_is_generated_from_the_simulations_public_verbs():
    sim = Simulation()
    try:
        tool = sim.tool
        assert isinstance(tool, AgentTool) and tool.tool_name == "sim_world"
        props = tool.tool_spec["inputSchema"]["json"]["properties"]
        verbs = set(props["rpc"]["enum"])
        assert {"add_robot", "add_object", "camera", "step", "reset", "render", "describe", "remove"} <= verbs
        assert not verbs & {"close", "add", "manual_time", "default", "unique_id", "editing"}  # no context managers
        assert props["add_object"]["properties"]["position"]["type"] == "array"
        assert props["step"]["properties"]["n"] == {"type": "integer", "default": 1}
        add_robot = props["add_robot"]["properties"]
        assert "name" in props["add_robot"]["required"]
        assert not set(add_robot) & {"port", "driver", "confirm", "cameras", "sim"}  # an agent never mounts hardware
    finally:
        sim.close()


def test_an_agent_builds_a_scene_steps_and_renders_an_image_block():
    sim = Simulation()
    try:
        r = _turn(
            sim.tool,
            [
                ("sim_world", {"rpc": "add_robot", "add_robot": {"name": "so101", "position": [0, 0, 0]}}),
                (
                    "sim_world",
                    {"rpc": "add_object", "add_object": {"name": "cube", "shape": "box", "position": [0.2, 0, 0.05]}},
                ),
                (
                    "sim_world",
                    {"rpc": "camera", "camera": {"name": "front", "position": [1, 0, 0.5], "size": [64, 48]}},
                ),
                ("sim_world", {"rpc": "step", "step": {"n": 10}}),
                ("sim_world", {"rpc": "render", "render": {"camera": "front"}}),
                ("sim_world", {"rpc": "reset"}),
                ("sim_world", {"rpc": "describe"}),
            ],
        )
        assert [x["status"] for x in r] == ["success"] * 7, [x["content"][0]["text"] for x in r]
        render = r[4]["content"]
        assert json.loads(render[0]["text"]) == {"image_shape": [48, 64, 3]}
        assert render[1]["image"]["format"] == "png" and render[1]["image"]["source"]["bytes"][:4] == b"\x89PNG"
        d = json.loads(r[6]["content"][0]["text"])
        assert "so101" in d["robots"] and "cube" in json.dumps(d["objects"])
        assert isinstance(sim.robots["so101"], Robot) or "so101" in sim.robots
    finally:
        sim.close()


def _logged(obj, verb, done, delay=0.0):
    """``obj.<verb>`` that sleeps ``delay`` s, runs, then appends ``verb`` to ``done`` (the order calls finish in)."""
    import functools
    import time

    real = getattr(obj, verb)

    @functools.wraps(real)  # the tool reads the verb's signature
    def logged(*a, **k):
        time.sleep(delay)
        out = real(*a, **k)
        done.append(verb)
        return out

    return logged


def test_one_message_that_edits_the_world_and_renders_it_runs_in_the_order_sent():
    # strands' default ConcurrentToolExecutor starts every call of one message at once; the slow camera and act must
    # still finish before the render sent after them, across the world's tool and the robot's tool
    sim = Simulation()
    try:
        arm, done = Robot("so101", sim=sim), []
        sim.camera = _logged(sim, "camera", done, delay=0.2)  # type: ignore[method-assign]
        arm.act = _logged(arm, "act", done, delay=0.4)  # type: ignore[method-assign]
        sim.render = _logged(sim, "render", done)  # type: ignore[method-assign]
        cam = {"rpc": "camera", "camera": {"name": "c", "position": [0.5, -0.5, 0.4], "size": [32, 24]}}
        act = {"rpc": "act", "act": {"action": {"shoulder_pan": 0.3}}}
        render = {"rpc": "render", "render": {"camera": "c"}}
        agent = Agent(
            model=ScriptedModel([[("sim_world", cam), (arm.tool.tool_name, act), ("sim_world", render)]]),
            tools=[sim.tool, arm.tool],
            callback_handler=None,
        )
        agent("put a camera on the arm, move it, show me")
        r = [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]
        assert [x["status"] for x in r] == ["success"] * 3, [x["content"][0]["text"] for x in r]
        assert done == ["camera", "act", "render"]
    finally:
        sim.close()
