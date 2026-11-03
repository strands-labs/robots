"""The page sees what a policy sees (a camera's picture) and changes the world with the model's own verbs."""

from __future__ import annotations

import io

import httpx
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402


@pytest.fixture
def served():
    sim = Simulation()
    Robot("so101", sim=sim)
    sim.camera("overhead", position=[0.25, 0, 1.0], target=[0.25, 0, 0], size=(96, 64))
    server = Dashboard(sim).start()
    yield sim, server.url
    server.stop()
    sim.close()


def test_a_camera_is_listed_and_its_png_is_the_frame_sim_render_returns(served) -> None:
    from PIL import Image

    sim, url = served
    assert httpx.get(f"{url}/api/scene").json()["cameras"] == [
        {"name": "overhead", "size": [96, 64], "url": "/api/cameras/overhead.png"}
    ]
    r = httpx.get(f"{url}/api/cameras/overhead.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    served_img, rendered = np.asarray(Image.open(io.BytesIO(r.content))).astype(int), sim.render("overhead").astype(int)
    assert served_img.shape == rendered.shape == (64, 96, 3)
    assert np.abs(served_img - rendered).mean() < 0.05  # the same picture; the GPU may differ by one level on a pixel
    assert httpx.get(f"{url}/api/cameras/nope.png").status_code == 404


@pytest.mark.parametrize(
    ("call", "status"),
    [
        ({"rpc": "add_object", "add_object": {"name": "cube", "shape": "box", "size": [0.02] * 3}}, 200),
        ({"rpc": "camera", "camera": {"name": "side", "position": [1, 0, 0.5], "target": [0, 0, 0]}}, 200),
        ({"rpc": "close"}, 400),
        ({"rpc": "add_object", "add_object": {"nme": "typo"}}, 400),
    ],
)
def test_the_page_changes_the_world_with_the_verbs_sim_tool_offers(served, call, status) -> None:
    sim, url = served
    r = httpx.post(f"{url}/api/world", json=call)
    assert r.status_code == status, r.text
    if status == 200:
        assert call[call["rpc"]]["name"] in {o.name for o in sim.scene.objects} | {c.name for c in sim.scene.cameras}


def test_the_page_shows_each_camera_as_the_world_renders_it(served) -> None:
    from .browser import page_on
    from .test_twin_page import MEDIA

    _, url = served
    with page_on(url) as page:
        page.wait_for_function("document.querySelector('.card.cam img')?.naturalWidth > 0", timeout=30_000)
        size = page.evaluate("(i => [i.naturalWidth, i.naturalHeight])(document.querySelector('.card.cam img'))")
        page.screenshot(path=str(MEDIA / "row5-camera-card.png"))
        assert size == [96, 64]
        assert not page.errors, page.errors
