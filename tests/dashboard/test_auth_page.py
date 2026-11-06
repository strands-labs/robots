"""The page's passkey screens in Chromium, with its virtual authenticator: first-time setup from the printed link,
sign-in after the cookie is gone, and the Passkeys panel."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
pytest.importorskip("webauthn")
sync_api = pytest.importorskip("playwright.sync_api")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

pytestmark = pytest.mark.signed_out


def test_setup_link_then_sign_in_then_the_passkeys_panel() -> None:
    sim = Simulation()
    Robot("so101", sim=sim)
    dash = Dashboard(sim)
    server = dash.start()
    try:
        with sync_api.sync_playwright() as p:
            try:
                browser = p.chromium.launch(
                    args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
                )
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 860})
            cdp = page.context.new_cdp_session(page)
            cdp.send("WebAuthn.enable")
            options = {"protocol": "ctap2", "transport": "internal", "hasResidentKey": True}
            options |= {"hasUserVerification": True, "isUserVerified": True, "automaticPresenceSimulation": True}
            cdp.send("WebAuthn.addVirtualAuthenticator", {"options": options})

            page.goto(server.url)  # no setup code in the link: nothing can be enrolled
            page.wait_for_selector("#signin")
            assert page.locator("#signin h1").inner_text() == "Create the first passkey"
            assert page.evaluate("() => window.__sceneLoaded") is None
            page.click("#signin button")
            page.wait_for_function("() => document.querySelector('#signin .err').textContent.length > 0")
            assert "setup code" in page.locator("#signin .err").inner_text()

            page.goto(dash.auth.setup_url(server.url))
            page.click("#signin button")
            page.wait_for_function("() => window.__sceneLoaded >= 1", timeout=60_000)
            assert "setup=" not in page.url and not dash.auth.setup_required

            page.context.clear_cookies()
            page.reload()
            page.wait_for_selector("#signin")
            assert page.locator("#signin h1").inner_text() == "Sign in"
            page.click("#signin button")
            page.wait_for_function("() => window.__sceneLoaded >= 1", timeout=60_000)
            page.click('#tabs button[data-tab="passkeys"]')
            page.wait_for_function("() => document.querySelectorAll('[data-pane=passkeys] li').length === 1")
            assert "this computer" in page.locator("[data-pane=passkeys] li").inner_text()
            browser.close()
    finally:
        server.stop()
        sim.close()
