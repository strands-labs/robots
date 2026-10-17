"""The page paints THE world: robots from /api/scene appear in the browser's MuJoCo twin."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

from .browser import page_on  # noqa: E402

MEDIA = Path(os.environ.get("DASHBOARD_MEDIA", Path.home() / ".tiny/v1-local-20261015/media/dashboard"))


@pytest.fixture
def served():
    sim = Simulation()
    Robot("so101", sim=sim)
    Robot("unitree_g1", sim=sim, position=[1.0, 0.0, 0.0])
    server = Dashboard(sim).start()
    yield sim, server
    server.stop()
    sim.close()


def test_two_robots_are_painted_from_the_scene(served) -> None:
    sim, server = served
    with page_on(server.url) as page:
        assert not page.evaluate("window.__error || null"), page.evaluate("window.__error")
        joints = page.evaluate("window.__twin.model.jnt_names")
        assert any(j.startswith("so101/") for j in joints)
        assert any(j.startswith("unitree_g1/") for j in joints)
        assert page.evaluate("window.__twin.model.nmesh") == Dashboard(sim).scene_json()["stats"]["nmesh"]
        page.wait_for_function("window.__twin.frames > 5", timeout=20_000)
        assert page.evaluate("window.__twin.geoms.length") > 0
        MEDIA.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(MEDIA / "row1-two-robots.png"))
        assert not page.errors, page.errors
