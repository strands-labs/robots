"""Connect: a peer's robot attaches as a card (gated act, e-stop through the mesh, device-side resume); a hardware
robot's camera is its own frame, not a render; a connected policy spec shows in /api/policies and runs."""

from __future__ import annotations

import uuid

import httpx
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402
from strands_robots.dashboard.connect import mesh_url
from strands_robots.dashboard.connect_policy import policy_spec  # noqa: E402
from strands_robots.mesh import Mesh  # noqa: E402
from strands_robots.mesh.authority import Authority  # noqa: E402
from strands_robots.robot.drivers.memory import MemoryDriver  # noqa: E402

YES = {"X-Confirm": "yes"}


@pytest.fixture
def peer():
    """Another peer on a mem:// bus serving a simulated arm (it trusts operator names: resume works)."""
    bus, tenant = f"mem://connect-{uuid.uuid4().hex[:8]}", f"t-{uuid.uuid4().hex[:6]}"
    world = Simulation()
    arm = Robot("so101", sim=world, id="far_arm")
    world.camera("wrist_view", attach=arm, size=(64, 48))
    mesh = Mesh(bus, tenant=tenant, resume_authority=Authority(trust_names=True))
    mesh.mount(arm, labels={"site": "lab"})
    yield bus, tenant, arm
    mesh.close()
    world.close()


def test_address_forms() -> None:
    assert mesh_url("lan") == "zenoh://"
    assert mesh_url("10.0.0.5:7447") == "zenoh://?connect=tcp/10.0.0.5:7447"
    assert mesh_url("tls/host.example:7447") == "zenoh://?connect=tls/host.example:7447"
    with pytest.raises(ValueError, match="host:port"):
        mesh_url("nonsense")
    assert policy_spec("127.0.0.1:7450", "lerobot-async", "act_so101") == "zenoh://127.0.0.1:7450/act_so101"
    assert policy_spec("zenoh://h:1/d", "lerobot-async", None) == "zenoh://h:1/d"
    with pytest.raises(ValueError, match="deployment"):
        policy_spec("h:1", "lerobot-async", None)
    with pytest.raises(ValueError, match="kind"):
        policy_spec("x", "grpc", None)


def test_a_peers_robot_attaches_moves_only_with_confirm_and_estops_through_the_mesh(peer) -> None:
    bus, tenant, arm = peer
    sim = Simulation()
    dash = Dashboard(sim)
    server = dash.start()
    try:
        u = server.url
        assert httpx.post(f"{u}/api/connect/mesh", json={"address": bus, "tenant": tenant}).status_code == 200
        devs = httpx.get(f"{u}/api/connect/peers").json()["devices"]
        assert [(d["id"], d["attached"]) for d in devs] == [("far_arm", False)]
        r = httpx.post(f"{u}/api/connect/robot", json={"address": bus})  # one device there: it is the one
        assert r.status_code == 201, r.text
        card = r.json()
        assert card["kind"] == "remote" and "shoulder_pan" in card["joints"] and not card["estop"]
        assert [c["name"] for c in card["cameras"]] == ["wrist_view"]

        act = f"{u}/api/robots/far_arm/act"
        r = httpx.post(act, json={"targets": {"shoulder_pan": 0.3}})
        assert r.status_code == 403 and "X-Confirm" in r.json()["detail"]
        r = httpx.post(act, json={"targets": {"shoulder_pan": 0.3}}, headers=YES)
        assert r.status_code == 200, r.text
        assert arm.observe(images=False).joints["shoulder_pan"] != 0.0  # the peer's own world moved
        reach = f"{u}/api/connect/robot/far_arm/act"
        assert httpx.post(reach, json={"targets": {"elbow_flex": 0.4}, "seconds": 0.5}).status_code == 403
        r = httpx.post(reach, json={"targets": {"elbow_flex": 0.4}, "seconds": 0.5}, headers=YES)
        assert r.status_code == 200, r.text
        assert r.json()["joints"]["elbow_flex"] == pytest.approx(0.4, abs=0.1)  # reached over 0.5 s, not one step

        png = httpx.get(f"{u}/api/robots/far_arm/cameras/wrist_view.png")
        assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert httpx.get(f"{u}/api/robots/far_arm/cameras/nope.png").status_code == 404

        assert httpx.post(f"{u}/api/robots/far_arm/estop", json={}).status_code == 200
        assert arm.estopped  # latched on the device, not just on this page's handle
        assert httpx.get(f"{u}/api/connect/robots").json()["robots"][0]["estop"] is True
        assert httpx.post(act, json={"targets": {"shoulder_pan": 0.0}}, headers=YES).status_code == 409

        res = f"{u}/api/connect/robot/far_arm/resume"
        assert httpx.post(res, json={"approved_by": "op"}).status_code == 403  # no X-Confirm: the device is not asked
        assert arm.estopped
        r = httpx.post(res, json={"approved_by": "op"}, headers=YES)
        assert r.status_code == 403 and "not a resume authority" in r.json()["detail"]  # the device's word, verbatim
        r = httpx.post(res, json={}, headers=YES)  # as this page's operator, who stopped it
        assert r.status_code == 200, r.text
        assert not arm.estopped and r.json()["estop"] is False

        assert httpx.post(f"{u}/api/connect/robot/far_arm/detach", json={}).status_code == 200
        assert httpx.get(f"{u}/api/connect/robots").json()["robots"] == []
        assert httpx.post(act, json={"targets": {"shoulder_pan": 0.0}}, headers=YES).status_code == 404
    finally:
        server.stop()
        dash.connections.close()
        sim.close()


class CameraDriver(MemoryDriver):
    """A bus with no wire whose one camera sees a fixed gradient (a real camera's frame, never a sim render)."""

    def images(self):
        img = np.zeros((24, 32, 3), np.uint8)
        img[..., 0] = np.arange(32, dtype=np.uint8) * 8
        return {"front": img}


def test_a_hardware_robots_card_streams_its_own_camera() -> None:
    sim = Simulation()
    from strands_robots import Description

    real = Robot("so101", driver=CameraDriver(Description.load("so101")), id="bench_arm", twin=False)
    server = Dashboard(sim, robots=[real]).start()
    try:
        cards = httpx.get(f"{server.url}/api/connect/robots").json()["robots"]
        assert [(c["id"], c["kind"], [k["name"] for k in c["cameras"]]) for c in cards] == [
            ("bench_arm", "hardware", ["front"])
        ]
        png = httpx.get(f"{server.url}{cards[0]['cameras'][0]['url']}")
        assert png.status_code == 200 and png.headers["content-type"] == "image/png"
        assert httpx.get(f"{server.url}/api/robots/bench_arm/cameras/front.png").content == png.content
    finally:
        server.stop()
        real.close()
        sim.close()


def test_a_connected_policy_is_listed_with_the_runnable_ones() -> None:
    sim = Simulation()
    Robot("so101", sim=sim, id="arm")
    server = Dashboard(sim).start()
    try:
        u = server.url
        r = httpx.post(f"{u}/api/connect/policy", json={"endpoint": "h:1", "kind": "grpc"})
        assert r.status_code == 400 and "kind" in r.json()["detail"]
        r = httpx.post(f"{u}/api/connect/policy", json={"endpoint": "lerobot/act_so101", "kind": "lerobot"})
        assert r.status_code == 200, r.text
        names = [p["name"] for p in httpx.get(f"{u}/api/policies").json()["policies"]]
        assert "lerobot/act_so101" in names
        assert httpx.get(f"{u}/api/connect/policies").json()["policies"][0]["kind"] == "lerobot"
    finally:
        server.stop()
        sim.close()
