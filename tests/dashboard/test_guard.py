"""Cross-site requests are refused: another web page the user has open cannot resume, act, spawn or chat."""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

import test_drivers  # noqa: E402

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

EVIL = {"Origin": "https://evil.example", "Content-Type": "text/plain"}


@pytest.fixture
def served():
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    d, _ = test_drivers.make_driver()
    real = Robot("so101", driver=d, id="real_arm", twin=False)
    server = Dashboard(sim, robots=[real]).start()
    yield sim, arm, real, server
    server.stop()
    real.close()
    sim.close()


def test_the_critics_cross_site_requests_are_refused(served) -> None:
    sim, arm, _, server = served
    u = server.url
    arm.estop()
    r = httpx.post(f"{u}/api/robots/so101/resume", content='{"approved_by":"evil"}', headers=EVIL)
    assert r.status_code == 403 and arm.estopped, r.text
    arm.resume(approved_by="test")
    before = arm.observe(images=False)["shoulder_pan"]
    r = httpx.post(f"{u}/api/robots/so101/act", content='{"targets":{"shoulder_pan":0.4}}', headers=EVIL)
    assert r.status_code == 403 and arm.observe(images=False)["shoulder_pan"] == before
    r = httpx.post(f"{u}/api/robots", content='{"name":"unitree_g1"}', headers=EVIL)
    assert r.status_code == 403 and list(sim.robots) == ["so101"]


@pytest.mark.parametrize(
    "headers, why",
    [
        ({"Origin": "https://evil.example", "Content-Type": "application/json"}, "Origin"),
        ({"Content-Type": "text/plain"}, "Content-Type"),
        ({"Host": "evil.example", "Content-Type": "application/json"}, "Host"),  # DNS rebinding
    ],
)
def test_each_check_alone_refuses(served, headers, why) -> None:
    _, arm, _, server = served
    arm.estop()
    r = httpx.post(f"{server.url}/api/robots/so101/resume", content='{"approved_by":"x"}', headers=headers)
    assert r.status_code == 403 and why in r.json()["detail"] and arm.estopped


def test_same_origin_json_passes_and_a_rebound_get_is_refused(served) -> None:
    _, arm, _, server = served
    arm.estop()
    ok = {"Origin": server.url, "Content-Type": "application/json"}
    r = httpx.post(f"{server.url}/api/robots/so101/resume", json={"approved_by": "me"}, headers=ok)
    assert r.status_code == 200 and not arm.estopped
    assert httpx.get(f"{server.url}/api/scene", headers={"Host": "evil.example"}).status_code == 403


def test_cross_site_websocket_is_refused(served) -> None:
    import websockets.sync.client as wsc
    from websockets.exceptions import InvalidStatus

    _, _, _, server = served
    with pytest.raises(InvalidStatus):
        wsc.connect(server.url.replace("http", "ws") + "/ws/chat", origin="https://evil.example", open_timeout=5)
    with wsc.connect(server.url.replace("http", "ws") + "/ws/state", origin=server.url, open_timeout=5) as ws:
        assert "bodies" in __import__("json").loads(ws.recv(timeout=5))


def test_resuming_a_real_body_needs_x_confirm(served) -> None:
    _, _, real, server = served
    real.estop()
    url = f"{server.url}/api/robots/real_arm/resume"
    r = httpx.post(url, json={"approved_by": "me"})
    assert r.status_code == 403 and "X-Confirm" in r.json()["detail"] and real.estopped
    r = httpx.post(url, json={"approved_by": "me"}, headers={"X-Confirm": "yes"})
    assert r.status_code == 200 and not real.estopped


def test_a_robot_written_g1_is_served_at_its_own_name(served) -> None:
    sim, _, _, server = served
    g1 = Robot("g1", sim=sim, position=[1.0, 0.0, 0.0])
    r = httpx.post(f"{server.url}/api/robots/g1/estop", json={})
    assert r.status_code == 200 and g1.estopped, r.text  # was 404: "robots are ['so101', 'unitree_g1']"
