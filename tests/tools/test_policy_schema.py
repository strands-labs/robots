"""An agent reads the shipped presets and every wire form of a policy spec from the tool spec, before its first call."""

from __future__ import annotations

import pytest

from strands_robots import Robot, Simulation
from strands_robots.policy import PolicyLike
from strands_robots.policy.factory import FORMS
from strands_robots.policy.onnx import presets
from strands_robots.tools import parameters

WIRE_FORMS = [f.split()[0] for f in FORMS if not f.startswith("<callable>")]


def _robot_policy() -> dict:
    return Robot("so101").tool.tool_spec["inputSchema"]["json"]["properties"]["run"]["properties"]["policy"]


def _world_policy() -> dict:
    run = Simulation.default().tool.tool_spec["inputSchema"]["json"]["properties"]["run"]
    return run["properties"]["policies"]["additionalProperties"]


@pytest.mark.parametrize(
    ("schema", "robot"), [(_robot_policy, "so101"), (_world_policy, None)], ids=["Robot.run", "Simulation.run"]
)
def test_the_policy_schema_names_every_preset_that_drives_the_robot_and_every_wire_form(schema, robot):
    s = schema()
    assert s["type"] == "string"
    names = presets.preset_names(robot)
    for name in names:
        assert repr(name) in s["description"]
    for form in WIRE_FORMS:
        assert (form in s["description"]) == (bool(names) or form != "<preset>[?k=v&k2=v2]")
    assert "<callable>" not in s["description"]  # a function does not cross a wire


def test_a_preset_added_to_the_registry_reaches_the_schema(monkeypatch):
    table = {**presets._table(), "newduck": {"class": "microduck:MicroduckPolicy", "robots": ["microduck"]}}
    monkeypatch.setattr(presets, "_table", lambda: table)

    def run(policy: PolicyLike) -> None: ...

    assert "'newduck'" in parameters(run, joints=())["properties"]["policy"]["description"]
