"""Check 8: the dashboard serves THE world: both robots in ``/api/scene``, and ``WS /ws/state`` carries a
server-side act to the browser with the exact float the robot reads itself. A page on another origin can
neither move a robot nor clear its e-stop. Real uvicorn, real WebSocket, real MuJoCo.
"""

from __future__ import annotations

import json
import time

import pytest
from checks import needs_module

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
pytestmark = pytest.mark.timeout(120)
EVIL = {"Origin": "https://evil.example", "Content-Type": "text/plain"}


def test_the_dashboard_shows_a_live_robots_joints():
    needs_module("strands_robots.dashboard", "part 10: dashboard")
    from websockets.sync.client import connect

    from strands_robots import Robot, Simulation
    from strands_robots.dashboard import Dashboard

    sim = Simulation()
    arm, g1 = Robot("so101", sim=sim), Robot("unitree_g1", sim=sim, position=[1.0, 0.0, 0.0])
    server = Dashboard(sim).start()
    try:
        scene = httpx.get(f"{server.url}/api/scene").json()
        assert [r["id"] for r in scene["robots"]] == [arm.id, g1.id]

        with connect(server.url.replace("http://", "ws://") + "/ws/state") as ws:
            json.loads(ws.recv(timeout=5))
            arm.act({"shoulder_lift": 0.4}, steps=200)
            mine, t0 = arm.observe(images=False)["shoulder_lift"], time.perf_counter()
            key, seen = f"{arm.id}/{arm.description.unlabel('shoulder_lift')}", None  # model joint names
            while time.perf_counter() - t0 < 2.0:
                seen = json.loads(ws.recv(timeout=2))["bodies"][arm.id]["q"][key]
                if seen == mine:
                    break
            lag = time.perf_counter() - t0
            print(f"[measured] server act -> /ws/state frame with the robot's own float in {lag * 1000:.0f} ms")
            assert seen == mine and lag < 0.5, (seen, mine, lag)

        arm.estop()
        r = httpx.post(f"{server.url}/api/robots/{arm.id}/resume", content='{"approved_by":"x"}', headers=EVIL)
        assert r.status_code == 403 and arm.estopped, r.text
        arm.resume(approved_by="operator")
        before = arm.observe(images=False)["shoulder_pan"]
        act = f"{server.url}/api/robots/{arm.id}/act"
        r = httpx.post(act, content='{"targets":{"shoulder_pan":0.4}}', headers=EVIL)
        assert r.status_code == 403 and arm.observe(images=False)["shoulder_pan"] == before, r.text
        r = httpx.post(act, json={"targets": {"shoulder_pan": 0.4}})
        assert r.status_code == 200, r.text
    finally:
        server.stop()
        sim.close()
