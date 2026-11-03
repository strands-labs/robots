"""notebooks/examples/agent_builds_a_scene.py runs end to end with a scripted model: the agent builds the scene,
moves the arm and renders, through the generated sim.tool and robot.tool only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from scripted_model import ScriptedModel

EXAMPLE = Path(__file__).resolve().parents[2] / "notebooks" / "examples" / "agent_builds_a_scene.py"
CUBES = {"red": (0.2, 0.15), "green": (0.25, 0.0), "blue": (0.2, -0.15)}
TURNS = [
    [
        (
            "sim_world",
            {
                "rpc": "add_object",
                "add_object": {
                    "name": f"{c}_cube",
                    "shape": "box",
                    "size": 0.015,
                    "position": [x, y, 0.015],
                    "color": rgba,
                },
            },
        )
        for (c, (x, y)), rgba in zip(CUBES.items(), ([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]), strict=True)
    ]
    + [
        (
            "sim_world",
            {"rpc": "camera", "camera": {"name": "hero", "position": [0.6, -0.5, 0.4], "target": [0, 0, 0.1]}},
        )
    ],
    ("robot_so101", {"rpc": "observe"}),
    ("robot_so101", {"rpc": "act", "act": {"action": {"shoulder_pan": 0.3}, "steps": 400}}),
    ("robot_so101", {"rpc": "observe"}),
    ("sim_world", {"rpc": "render", "render": {"camera": "hero"}}),
]


def _example():
    spec = importlib.util.spec_from_file_location("agent_builds_a_scene", EXAMPLE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_example_builds_the_scene_moves_the_arm_and_renders(capsys):
    ex = _example()
    # one scripted turn per ask: each ask ends with "done", so the calls are spread over the three asks
    model = ScriptedModel([TURNS[0], *TURNS[1:4], TURNS[4]])
    sim, arm, agent = ex.main(model)
    calls = ex.trace(agent)
    assert [f"{t}.{r}" for t, r, _ in calls] == [
        *["sim_world.add_object"] * 3,
        "sim_world.camera",
        "robot_so101.observe",
        "robot_so101.act",
        "robot_so101.observe",
        "sim_world.render",
    ]
    assert {s for *_, s in calls} == {"success"}, calls
    scene = sim.describe()
    assert sorted(o["name"] for o in scene["objects"]) == ["blue_cube", "green_cube", "red_cube"]
    assert [c["name"] for c in scene["cameras"]] == ["hero"]
    assert arm.observe()["shoulder_pan"] == pytest.approx(0.3, abs=0.05)
    rendered = [b for m in agent.messages for b in m["content"] if "toolResult" in b][-1]["toolResult"]["content"]
    assert json.loads(rendered[0]["text"]) == {"image_shape": [480, 640, 3]} and "image" in rendered[1]
    assert "robot_so101.act: success" in capsys.readouterr().out
    sim.close()
