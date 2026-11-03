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


def test_a_page_run_on_hardware_passes_the_safety_chain_on_every_step_not_only_the_first(monkeypatch) -> None:
    """One X-Confirm is the operator's yes for the run; the e-stop, range, safe band, interlocks and validators still
    see every action (the same policy through ``robot.tool run`` is gated per step, and so is this). Before, only the
    first action was gated and step 2 on reached the bus through the developer path."""
    import strands_robots.policy.factory as factory
    from strands_robots.dashboard import runs
    from strands_robots.robot.safety import RequestRefused

    d, serial = test_drivers.make_driver()
    real = Robot("so101", driver=d, id="real_arm", twin=False)
    seen: list[float] = []

    def small_pan(robot, req):
        v = dict(req.items())["shoulder_pan"]
        seen.append(v)
        if abs(v) > 0.1:
            raise RequestRefused(f"shoulder_pan={v:.2f} is over the 0.1 rad fence")

    real.validators.append(small_pan)
    outputs = iter([0.05, 0.05, 0.5])
    monkeypatch.setattr(factory, "make", lambda spec: lambda obs: {"shoulder_pan": next(outputs, 0.5)})
    try:
        with pytest.raises(PermissionError, match="X-Confirm"):
            runs.start(real, "fenced", instruction=None, seconds=0.5, hz=20, confirmed=False)
        before = len(goal_writes(serial))
        out = runs.start(real, "fenced", instruction=None, seconds=0.5, hz=20, confirmed=True)
        assert out["status"] == "running"
        for _ in range(100):
            if not real._task.running:
                break
            time.sleep(0.05)
        st = runs.status(real)  # the refusal is the run's end, loudly (the task's error is read once, here)
        assert st["status"] == "error" and "RequestRefused" in st["error"] and "0.1 rad fence" in st["error"], st
        assert len(seen) >= 3 and seen[0] == pytest.approx(0.05) and seen[-1] == pytest.approx(0.5)  # every step
        assert len(goal_writes(serial)) - before <= 2  # the two 0.05 writes; 0.5 never reached the bus
        assert abs(float(real.observe(images=False).joints.positions[0])) <= 0.1 + 1e-6
    finally:
        real.close()
