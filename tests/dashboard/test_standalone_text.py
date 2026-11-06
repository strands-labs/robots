"""The page served by a plain file server (no dashboard, so no policy header): ``?standalone=`` and the bundle's own
strings stay text, and the page's own ``<meta>`` policy keeps inline markup from running.

``?standalone=`` comes from whoever wrote the link, like ``?scene=``: only a file on the page's own origin is loaded.
"""

from __future__ import annotations

import functools
import http.server
import json
import re
import threading
import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots.dashboard import guard, standalone  # noqa: E402

PAYLOAD = '<img src=x onerror="window.__ran = 1">'
STATIC = Path(guard.__file__).parent / "static"


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass


@contextmanager
def _browser() -> Iterator[Any]:
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            )
        except Exception as exc:  # noqa: BLE001 — no browser installed is a skip, not a failure
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            yield browser.new_page()
        finally:
            browser.close()


def _foreign(asked: list[str], request: Any) -> None:
    if request.url.startswith(("http://elsewhere", "data:")):
        asked.append(request.url)


@pytest.fixture(scope="module")
def site() -> Iterator[str]:
    handler = functools.partial(_Quiet, directory=str(STATIC))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/index.html?cli=127.0.0.1:1"
    httpd.shutdown()


def test_the_page_carries_the_same_script_policy_as_the_server_sends() -> None:
    html = (STATIC / "index.html").read_text()
    meta = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)">', html)
    assert meta is not None and meta.group(1) == guard.PAGE_POLICY  # the inline script's hash included
    assert html.index("Content-Security-Policy") < html.index("<script")  # before anything it governs


def test_a_standalone_bundle_on_another_site_is_not_loaded(site: str) -> None:
    other = "http://elsewhere.invalid/scene.json"
    for value in (other, "//elsewhere.invalid/scene.json", "data:application/json,{}"):
        with _browser() as page:
            asked: list[str] = []
            page.on("request", functools.partial(_foreign, asked))
            page.goto(f"{site}&standalone={urllib.parse.quote(value)}")
            page.wait_for_function("() => ['offline', 'standalone', 'error'].includes(window.__state.mode)")
            page.wait_for_timeout(500)
            assert asked == [], value
            assert page.evaluate("() => window.__ran") is None


def test_a_bundles_physics_strings_are_text_and_inline_markup_cannot_run(site: str) -> None:
    try:
        standalone.build()
    except (OSError, ImportError) as exc:  # no cached weights and no network is a skip, not a failure
        pytest.skip(f"standalone bundle unavailable: {exc}")
    scene = json.loads((STATIC / "scenes" / "microduck" / "scene.json").read_text())
    scene["physics"]["substeps"] = PAYLOAD
    scene["standalone"]["hz"] = PAYLOAD
    with _browser() as page:
        body = json.dumps(scene)
        page.route("**/scenes/microduck/scene.json", lambda r: r.fulfill(body=body, content_type="application/json"))
        page.goto(site)
        ready = "() => window.__standalone && window.__standalone.stats || window.__error"
        page.wait_for_function(ready, timeout=120_000)
        assert page.locator("#notice img").count() == 0
        assert PAYLOAD in page.locator("#notice").inner_text()
        page.evaluate(f"() => document.body.insertAdjacentHTML('beforeend', {json.dumps(PAYLOAD)})")
        page.wait_for_timeout(300)
        assert page.evaluate("() => window.__ran") is None
