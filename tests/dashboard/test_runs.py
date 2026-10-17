"""Run a policy from the page's routes: start, read the task, stop; a real body needs X-Confirm before it runs."""

from __future__ import annotations

import time

import httpx
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

import test_drivers  # noqa: E402
from test_operator_gate import goal_writes  # noqa: E402

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402


def test_run_status_stop_on_a_sim_body_and_the_real_gate() -> None:
    sim = Simulation()
    Robot("so101", sim=sim)
    d, serial = test_drivers.make_driver()
    real = Robot("so101", driver=d, id="real_arm", twin=False)
    server = Dashboard(sim, robots=[real]).start()
    try:
        u = server.url
        pol = httpx.get(f"{u}/api/policies").json()
        assert pol["policies"][0]["name"] == "hold" and pol["forms"]
        assert httpx.get(f"{u}/api/robots/so101/run").json()["status"] == "idle"
        r = httpx.post(f"{u}/api/robots/so101/run", json={"policy": "hold", "seconds": None})
        assert r.status_code == 202, r.text
        assert httpx.get(f"{u}/api/robots/so101/run").json()["status"] == "running"
        assert httpx.post(f"{u}/api/robots/so101/run", json={"policy": "hold"}).status_code == 409  # one task each
        time.sleep(0.3)
        out = httpx.post(f"{u}/api/robots/so101/stop", json={}).json()
        assert out["status"] == "stopped" and out["result"]["stopped_by"] == "stopped" and out["result"]["steps"] > 0
        assert httpx.post(f"{u}/api/robots/so101/run", json={"policy": "not a spec"}).status_code == 400

        before = len(goal_writes(serial))
        r = httpx.post(f"{u}/api/robots/real_arm/run", json={"policy": "hold", "seconds": 0.2})
        assert r.status_code == 403 and "X-Confirm" in r.json()["detail"]
        assert len(goal_writes(serial)) == before and real._task is None  # refused before the loop and the bus
    finally:
        server.stop()
        real.close()
        sim.close()
