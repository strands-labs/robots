"""POST act on a real body: refused (403, with the reason) without X-Confirm; with it, it passes the gate."""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

import test_drivers  # noqa: E402
from test_operator_gate import goal_writes  # noqa: E402

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402


def test_real_body_needs_x_confirm() -> None:
    sim = Simulation()
    Robot("so101", sim=sim, id="sim_arm")
    d, serial = test_drivers.make_driver()  # a fake serial bus: nothing leaves this process
    real = Robot("so101", driver=d, id="real_arm", twin=False)
    server = Dashboard(sim, robots=[real]).start()
    try:
        url = f"{server.url}/api/robots/real_arm/act"
        before = len(goal_writes(serial))
        r = httpx.post(url, json={"targets": {"shoulder_pan": 0.2}})
        assert r.status_code == 403
        assert "X-Confirm" in r.json()["detail"]
        assert len(goal_writes(serial)) == before  # refused before the bus

        r = httpx.post(url, json={"targets": {"shoulder_pan": 0.2}}, headers={"X-Confirm": "yes"})
        assert r.status_code == 200, r.text
        assert len(goal_writes(serial)) > before
        assert r.json()["joints"]["shoulder_pan"] == pytest.approx(0.2, abs=1e-2)

        r = httpx.post(f"{server.url}/api/robots/sim_arm/act", json={"targets": {"shoulder_pan": 0.2}})
        assert r.status_code == 200  # a simulated body is never gated
        r = httpx.post(url, json={"targets": {"no_such_joint": 1}}, headers={"X-Confirm": "yes"})
        assert r.status_code == 400
    finally:
        server.stop()
        real.close()
        sim.close()
