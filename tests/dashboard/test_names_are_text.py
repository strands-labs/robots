"""A name the server sends is text on the page, never markup: an agent chooses camera names and robot ids through
``sim.tool``, and a script running in this page could set ``X-Confirm`` itself."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

from .browser import page_on  # noqa: E402

PAYLOAD = '<img id="pwned" src=x onerror="window.__pwned=1">cam'


def test_an_agent_chosen_camera_name_cannot_inject_markup_into_the_page() -> None:
    sim = Simulation()
    Robot("so101", sim=sim)
    sim.camera(PAYLOAD, position=[0.25, 0, 1.0], target=[0.25, 0, 0], size=(64, 48))
    dash = Dashboard(sim)
    server = dash.start()
    try:
        assert any(c["name"] == PAYLOAD for c in dash.cameras_json())  # the server sends the name raw
        with page_on(server.url) as page:
            page.locator(".card.cam").first.wait_for(timeout=10_000)
            assert page.locator("#pwned").count() == 0, "the name was parsed as HTML"
            assert page.evaluate("window.__pwned") is None
            assert PAYLOAD in page.locator(".card.cam h3").first.inner_text()  # shown verbatim, as text
            assert not page.errors, page.errors
    finally:
        server.stop()
        sim.close()
