"""Headless Chromium on a served dashboard (skips when playwright or its browser is missing)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from conftest import dashboard_token

from strands_robots.dashboard.auth.session import COOKIE


def sign_in(context: Any, url: str) -> None:
    """Give a browser context the session cookie a passkey sign-in on ``url``'s dashboard would have set."""
    origin = "/".join(url.split("/")[:3])
    context.add_cookies([{"name": COOKIE, "value": dashboard_token(), "url": origin}])


@contextmanager
def page_on(
    url: str, *, timeout_ms: int = 60_000, viewport: tuple[int, int] = (1400, 860), signed_in: bool = True
) -> Iterator[Any]:
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            )
        except Exception as exc:  # noqa: BLE001 — no browser installed is a skip, not a failure
            pytest.skip(f"chromium unavailable: {exc}")
        page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        if signed_in:
            sign_in(page.context, url)
        errors: list[str] = []
        requests: list[tuple[float, str]] = []  # (monotonic s, url): every request the page makes, from load
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append((time.monotonic(), r.url)))
        page.requests = requests
        page.goto(url)
        page.wait_for_function("() => window.__sceneLoaded >= 1 || window.__error", timeout=timeout_ms)
        page.errors = errors
        try:
            yield page
        finally:
            browser.close()
