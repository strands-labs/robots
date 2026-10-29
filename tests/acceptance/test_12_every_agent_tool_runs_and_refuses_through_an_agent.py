"""Check 12: every agent tool answers through a real strands ``Agent``, the way a model calls it.

The tools are generated (``robot.tool``, ``sim.tool``, ``mesh.tool`` = use_mesh). Every verb a tool offers has a
row below that runs it for real on a sim world; a refused call must come back as a failed tool result with its
reason in the text. A verb added to a tool without a row here fails the walk.
"""

from __future__ import annotations

import uuid

import pytest
from checks import needs_module

pytest.importorskip("mujoco")
pytestmark = pytest.mark.timeout(300)
HOLD = "acceptance.hold"

# (tool, verb, arguments under the verb, expected status, text that must be in the result)
SIM = [
    ("add_robot", {"name": "so101", "position": [1.0, 0.0, 0.0], "id": "guest"}, "success", ""),
    ("add_object", {"name": "cube", "shape": "box", "position": [0.2, 0.0, 0.05]}, "success", ""),
    ("camera", {"name": "front", "position": [1.0, 0.0, 0.5], "size": [64, 48]}, "success", ""),
    ("camera", {"name": "front", "position": [0.0, -1.0, 0.5]}, "success", ""),  # the same name re-aims it
    ("light", {"name": "lamp", "position": [0.0, 0.0, 1.5], "color": [0.2, 0.2, 1.0]}, "success", ""),
    ("light", {"name": "lamp", "color": [2.0, 0.0, 0.0]}, "error", "RGB"),
    ("set_pose", {"body": "cube", "position": [0.25, 0.0, 0.05]}, "success", ""),
    ("pose", {"body": "cube"}, "success", "0.25"),
    ("pose", {"body": "nobody"}, "error", "nobody"),
    ("run", {"policies": {"guest": HOLD}, "steps": 3}, "success", "guest"),
    ("run", {"policies": {"ghost": HOLD}, "steps": 3}, "error", "ghost"),
    ("step", {"n": 5}, "success", ""),
    ("control_step", None, "success", ""),
    ("contacts", None, "success", ""),
    ("randomize", {"friction": 1.0, "seed": 0}, "success", ""),
    ("render", {"camera": "front"}, "success", "image_shape"),
    ("render", {"camera": "nowhere"}, "error", "nowhere"),
    ("describe", None, "success", "cube"),
    ("remove", {"robot": "guest"}, "success", ""),
    ("reset", None, "success", ""),
]
ROBOT = [
    ("describe", None, "success", "device_id"),
    ("observe", None, "success", "shoulder_pan"),
    ("act", {"action": {"shoulder_pan": 0.2}}, "success", "shoulder_pan"),
    ("act", {"action": {"elbow": 0.1}}, "error", "elbow"),
    ("run", {"policy": HOLD, "steps": 3}, "success", ""),
    ("stop", None, "success", ""),
    ("hold", None, "success", ""),
    ("calibrate", None, "error", "calibrat"),
    ("estop", None, "success", ""),
    ("act", {"action": {"shoulder_pan": 0.0}}, "error", ""),  # latched
]
MESH = [
    ("list", {"selector": "device(role:arm)"}, "success", "mesh-arm"),
    ("describe", {"device_id": "mesh-arm"}, "success", "device_id"),
    ("observe", {"device_id": "mesh-arm"}, "success", "shoulder_pan"),
    ("act", {"device_id": "mesh-arm", "act": {"shoulder_pan": 0.2}}, "success", ""),
    ("act", {"device_id": "mesh-arm", "act": {"shoulder_pan": 99.0}}, "error", "shoulder_pan"),
    ("run", {"device_id": "mesh-arm", "run": {"policy": HOLD, "steps": 3}}, "success", "started"),
    ("stop", {"device_id": "mesh-arm"}, "success", ""),
    ("estop", {"selector": "device(role:arm)"}, "success", "mesh-arm"),
    ("resume", {"device_id": "mesh-arm"}, "error", "operator"),  # no operator key on this mesh
]


@pytest.fixture
def hold_policy(monkeypatch):
    from strands_robots.policy import factory

    make = factory.make
    monkeypatch.setattr(factory, "make", lambda spec, **kw: (lambda obs: {}) if spec == HOLD else make(spec, **kw))


def _walk(agent, tool, rows, wrap):
    offered = set(_verbs(tool))
    missing = offered - {r[0] for r in rows}
    assert not missing, f"{tool.tool_name} offers {sorted(missing)} with no acceptance row"
    for verb, args, status, text in rows:
        result = getattr(agent.tool, tool.tool_name)(**wrap(verb, args))
        body = " ".join(str(c.get("text", "")) for c in result["content"])
        assert result["status"] == status and text in body, (tool.tool_name, verb, args, result["status"], body[:300])


def _verbs(tool):
    props = tool.tool_spec["inputSchema"]["json"]["properties"]
    return (props.get("rpc") or props.get("action"))["enum"]


def test_every_agent_tool_runs_and_refuses_through_an_agent(hold_policy):
    needs_module("strands", "the agent tools extra")
    needs_module("strands_robots.tools", "part 8: agent tools")
    needs_module("strands_robots.mesh", "part 7: the mesh")
    from strands import Agent

    from strands_robots import Robot
    from strands_robots.mesh import Mesh

    arm = Robot("so101")
    bus = f"mem://acceptance-{uuid.uuid4().hex[:8]}"
    device_side, ctl = Mesh(bus, tenant="acceptance"), Mesh(bus, tenant="acceptance")
    meshed = Robot("so101", id="mesh-arm", position=[0.0, 1.0, 0.0])
    device_side.mount(meshed, device_id="mesh-arm", labels={"role": "arm"}, watchdog_ms=None)
    assert ctl.discover("device(role:arm)", min_count=1, timeout=3.0)
    tools = [arm.sim.tool, arm.tool, ctl.tool]
    agent = Agent(tools=tools, callback_handler=None)
    try:
        _walk(agent, tools[0], SIM, lambda v, a: {"rpc": v, **({v: a} if a else {})})
        _walk(agent, tools[1], ROBOT, lambda v, a: {"rpc": v, **({v: a} if a else {})})
        _walk(agent, tools[2], MESH, lambda v, a: {"action": v, **(a or {})})
    finally:
        ctl.close()
        device_side.close()
