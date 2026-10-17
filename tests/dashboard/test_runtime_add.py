"""A robot added at runtime (by a script, or POST /api/robots) recompiles the twin without a page reload."""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

from .browser import page_on  # noqa: E402
from .test_twin_page import MEDIA  # noqa: E402


def test_runtime_add_recompiles_the_twin_in_place() -> None:
    sim = Simulation()
    Robot("so101", sim=sim)
    server = Dashboard(sim).start()
    try:
        with page_on(server.url) as page:
            page.evaluate("window.__sameDocument = 'yes'")
            loads = page.evaluate("window.__sceneLoaded")
            Robot("unitree_g1", sim=sim, position=[1.0, 0.0, 0.0])  # a script adds a body to THE world
            page.wait_for_function(f"window.__sceneLoaded > {loads}", timeout=30_000)
            page.wait_for_function("window.__twin.model.jnt_names.some(n => n.startsWith('unitree_g1/'))")
            r = httpx.post(f"{server.url}/api/robots", json={"name": "so100", "position": [0, 1, 0]})
            assert r.status_code == 201, r.text
            assert r.json()["id"] in sim.robots
            page.wait_for_function("window.__twin.model.jnt_names.some(n => n.startsWith('so100/'))", timeout=30_000)
            assert page.evaluate("window.__sameDocument") == "yes"  # no reload: the same document
            assert page.evaluate("window.__twin.revision") == sim.revision
            page.screenshot(path=str(MEDIA / "row3-runtime-add.png"))
            assert not page.errors, page.errors
            bad = httpx.post(f"{server.url}/api/robots", json={"name": "no_such_robot"})
            assert bad.status_code in (400, 404), bad.text
    finally:
        server.stop()
        sim.close()
