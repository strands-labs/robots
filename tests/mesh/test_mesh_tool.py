"""mesh.tool (use_mesh): one agent tool over every device on the mesh, generated from their manifests."""

from __future__ import annotations

import json
import sys

import pytest

from strands_robots import Robot, Simulation
from strands_robots.mesh import Mesh, authority

pytest.importorskip("strands")


@pytest.fixture
def lab():
    m = Mesh("mem://tool-lab", tenant="t-tool")
    sim = Simulation()
    arm = Robot("so101", sim=sim, id="arm")
    m.mount(arm, labels={"role": "arm"}, watchdog_ms=None)
    yield m, arm
    m.close()
    sim.close()


def call(tool, **inp):
    res = tool._tool_func({"toolUseId": "t1", "input": inp})
    text = res["content"][0]["text"]
    return res["status"], (json.loads(text) if res["status"] == "success" else text)


def test_spec_is_generated_from_the_manifests(lab):
    m, arm = lab
    tool = m.tool
    assert tool.tool_name == "use_mesh"
    schema = tool.tool_spec["inputSchema"]["json"]
    assert set(schema["properties"]["action"]["enum"]) == {
        "list",
        "describe",
        "observe",
        "act",
        "run",
        "stop",
        "estop",
        "resume",
    }
    assert set(schema["properties"]["act"]["properties"]) == set(arm.action_keys)
    assert "arm" in tool.tool_spec["description"] and "role:arm" in tool.tool_spec["description"]


def test_list_observe_act_through_the_tool(lab):
    m, arm = lab
    status, devices = call(m.tool, action="list", selector="device(role:arm)")
    assert status == "success" and [d["id"] for d in devices] == ["arm"]
    status, obs = call(m.tool, action="observe", device_id="arm")
    assert status == "success" and set(obs["joints"]) == set(arm.action_keys)
    status, obs = call(m.tool, action="act", device_id="arm", act={"shoulder_pan": 0.3})
    assert status == "success"
    assert arm.observe().joints["shoulder_pan"] > 0.01


def test_a_refusal_goes_back_to_the_model_as_text(lab):
    m, _ = lab
    status, text = call(m.tool, action="act", device_id="arm", act={"shoulder_pan": 99.0})
    assert status == "error" and "outside its range" in text


def test_estop_from_the_tool_and_resume_is_the_operators(lab):
    m, arm = lab
    status, acked = call(m.tool, action="estop", selector="device(*)")
    assert status == "success" and acked == ["arm"] and arm.estopped
    status, text = call(m.tool, action="resume", device_id="arm", approved_by=m.operator)
    assert status == "error" and "operator" in text and arm.estopped
    m.resume_key, public = authority.keygen()  # even with the operator's key on this mesh, a model never resumes
    m.resume_authority = authority.Authority(public_keys=(public,))
    status, text = call(m.tool, action="resume", device_id="arm", approved_by="op")
    assert status == "error" and arm.estopped
    m.invoke("arm", "resume", approved_by="op")  # the operator's own code does
    assert not arm.estopped


def test_a_new_device_shows_up_without_tool_code(lab):
    m, _ = lab
    arm2 = Robot("so101", sim=Simulation(), id="arm2")
    m.mount(arm2, labels={"role": "spare"}, watchdog_ms=None)
    assert "role:spare" in m.tool.tool_spec["description"]
    m.unmount("arm2")
    arm2.sim.close()


def test_strands_is_imported_only_when_the_tool_is_built():
    import subprocess

    code = "import strands_robots.mesh, sys; m = strands_robots.mesh.Mesh('mem://x'); print('strands' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
