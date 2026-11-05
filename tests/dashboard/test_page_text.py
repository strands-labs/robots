"""Text that reaches the page from its address or a scene description stays text, and inline markup cannot run.

``?scene=`` and ``?cli=`` come from whoever wrote the link; a scene's strings come from whoever wrote the file.
None of them may become script on the dashboard's origin, where a script could press any button the operator can.
"""

from __future__ import annotations

import json
import urllib.parse

import httpx
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

from .browser import page_on  # noqa: E402

PAYLOAD = '<img src=x onerror="window.__ran = 1">'


@pytest.fixture
def server():
    sim = Simulation()
    Robot("so101", sim=sim)
    server = Dashboard(sim).start()
    yield server
    server.stop()
    sim.close()


def test_a_scene_address_with_markup_is_shown_as_text(server) -> None:
    with page_on(f"{server.url}/?scene={urllib.parse.quote(PAYLOAD)}") as page:
        page.wait_for_timeout(500)
        assert page.evaluate("() => window.__ran") is None
        assert page.locator("#notice img").count() == 0
        assert "HTTP 404" in page.locator("#notice").inner_text()


def test_a_scene_on_another_site_is_not_loaded(server) -> None:
    with page_on(f"{server.url}/?scene={urllib.parse.quote('https://example.invalid/scene.json')}") as page:
        assert page.evaluate("() => window.__state.mode") == "server"  # ignored: the page talks to its own server


def test_a_scene_revision_that_is_not_a_number_never_reaches_the_page(server) -> None:
    scene = {"xml": "<mujoco><worldbody/></mujoco>", "assets": [], "robots": [], "objects": [], "cameras": []}
    scene |= {"physics": {"name": "x", "timestep": 0.002, "substeps": 1}, "revision": PAYLOAD}
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            )
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chromium unavailable: {exc}")
        page = browser.new_page()
        page.route(
            "**/planted.json", lambda route: route.fulfill(body=json.dumps(scene), content_type="application/json")
        )
        page.goto(f"{server.url}/?scene=planted.json")
        page.wait_for_function("() => window.__twin.model || window.__error", timeout=60_000)
        page.wait_for_timeout(300)
        assert page.evaluate("() => [window.__ran, window.__twin.revision]") == [None, None]
        assert page.locator("#status img").count() == 0
        browser.close()


def test_a_cli_address_that_is_not_loopback_host_port_is_ignored(server) -> None:
    with page_on(f"{server.url}/?cli={urllib.parse.quote(PAYLOAD)}") as page:
        assert page.evaluate("() => window.__ran") is None
        assert page.evaluate("() => window.__state.mode") == "server"


def test_markup_injected_into_the_page_cannot_run_script(server) -> None:
    with page_on(server.url) as page:
        page.evaluate(f"() => document.body.insertAdjacentHTML('beforeend', {json.dumps(PAYLOAD)})")
        page.wait_for_timeout(500)
        assert page.evaluate("() => window.__ran") is None  # the page's policy refuses inline handlers
        assert page.evaluate("() => document.documentElement.dataset.scheme") in (
            "paper",
            "dark",
        )  # its own inline script ran


def test_only_html_pages_carry_the_script_policy(server) -> None:
    page = httpx.get(f"{server.url}/").headers["content-security-policy"]
    assert "script-src 'self' 'sha256-" in page and "object-src 'none'" in page and page.startswith("frame-ancestors")
    worker = httpx.get(f"{server.url}/mujoco-worker.js").headers["content-security-policy"]
    assert "script-src" not in worker  # MuJoCo's bindings compile functions in the worker
