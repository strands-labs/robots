"""Standalone mode: with no server, the browser steps the bundled world and runs the ONNX skills itself.

S1 no server → standalone: scene compiled, the default skill loaded (61 in, 14 out), every skill offered, honest text
S2 stands 10 s at the control rate (trunk above stand_z, no fall)
S3 W walks it: vx = 0.3 for 10 s moves the trunk ≥ 1 m forward, still standing
S4 switching skill changes the motion: velstand with wz = 1 turns the body, without a reset
S5 a quiet console: no e-stop panel and no /api/mesh poll without a server; the server re-probe backs off 5 s -> 60 s
S6 with a server the page never steps physics: no standalone loop, no policy worker, no ONNX runtime fetched
"""

from __future__ import annotations

import functools
import http.server
import math
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard, standalone  # noqa: E402

from .browser import page_on  # noqa: E402
from .test_twin_page import MEDIA  # noqa: E402

PRESET, FACTS = standalone.browser_preset()
SPEC = FACTS["browser"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def site() -> Iterator[str]:
    """The dashboard's static folder on a plain file server (no API), with the bundle built into it."""
    try:
        standalone.build()
    except (OSError, ImportError) as exc:  # no cached weights and no network is a skip, not a failure
        pytest.skip(f"standalone bundle unavailable: {exc}")
    handler = functools.partial(_Quiet, directory=str(standalone.STATIC))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/index.html?cli=127.0.0.1:{_free_port()}"
    httpd.shutdown()


@pytest.fixture(scope="module")
def page(site: str) -> Iterator[Any]:
    with page_on(site, timeout_ms=120_000) as pg:
        pg.wait_for_function("window.__standalone && window.__standalone.stats || window.__error", timeout=120_000)
        yield pg


def _samples(page: Any, seconds: int) -> list[dict[str, float]]:
    out = []
    for _ in range(seconds):
        time.sleep(1.0)
        out.append(page.evaluate("window.__standalone.stats"))
    return out


def test_s1_no_server_lands_in_standalone_with_every_skill(page: Any) -> None:
    got = page.evaluate(
        """({mode: document.querySelector('#mode').dataset.mode,
            notice: document.querySelector('#notice').textContent,
            ready: window.__standalone.ready, skills: window.__standalone.skills,
            pad: !document.querySelector('#pad').hidden, errors: window.__standalone.errors,
            options: [...document.querySelectorAll('#skill option')].map((o) => o.value),
            fetched: performance.getEntriesByType('resource').map((r) => r.name)})"""
    )
    fetched = " ".join(got["fetched"])
    assert "policy-worker" in fetched and "/scenes/" in fetched  # what S5 must never see
    assert got["mode"] == "standalone", got
    assert "physics in your browser" in got["notice"] and "strands-robots dashboard" in got["notice"]
    assert (got["ready"]["obsWidth"], got["ready"]["actionWidth"]) == (61, 14)
    assert got["ready"]["skill"] == SPEC["skill"]
    assert got["skills"] == got["options"] == list(FACTS["files"])
    assert got["pad"] and not got["errors"] and not page.errors, (got["errors"], page.errors)


def test_s2_stands_ten_seconds_at_the_control_rate(page: Any) -> None:
    s = _samples(page, 10)
    assert min(x["z"] for x in s) > SPEC["stand_z"], s
    assert s[-1]["falls"] == 0
    assert sum(x["hz"] for x in s[1:]) / (len(s) - 1) >= 0.9 * SPEC["hz"], s


def test_s3_w_walks_it_a_metre(page: Any) -> None:
    x0 = page.evaluate("window.__standalone.stats.x")
    page.keyboard.down("w")
    try:
        s = _samples(page, 10)
    finally:
        page.keyboard.up("w")
    assert s[-1]["x"] - x0 >= 1.0, s
    assert min(x["z"] for x in s) > SPEC["stand_z"] and s[-1]["falls"] == 0, s


def test_s4_switching_skill_changes_the_motion(page: Any) -> None:
    t0 = page.evaluate("window.__standalone.stats.t")
    page.select_option("#skill", "velstand")
    page.wait_for_function("window.__standalone.skill === 'velstand'", timeout=20_000)
    page.evaluate("window.__standalone.command(0, 0, 1.0)")
    s = _samples(page, 6)
    page.evaluate("window.__standalone.command(0, 0, 0)")
    turned = sum(abs(math.remainder(b["yaw"] - a["yaw"], math.tau)) for a, b in zip(s, s[1:], strict=False))
    assert turned >= math.radians(90), s
    assert s[0]["t"] > t0 and s[-1]["falls"] == 0, s  # no reset: world time carries on
    MEDIA.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(MEDIA / "standalone.png"))


def test_s5_no_server_means_a_quiet_console(page: Any) -> None:
    """Each failed request is a red browser console error, so without a server the page must not keep asking."""
    page.wait_for_timeout(1_000)  # let the 50 ms safety loop see the final mode
    hidden = page.evaluate("document.querySelector('#safety').hidden")
    t0, t1 = page.requests[0][0], time.monotonic()
    mesh = [u for _, u in page.requests if "/api/mesh" in u]
    probes = [t - t0 for t, u in page.requests if u.endswith("/api/scene")]  # one same-origin GET per probe
    # Backing off 5, 10, 20, 40, 60 s, probes after the first land by 5, 15, 35, 75, 135 s (a fixed 5 s tick: far more).
    due = sum(1 for at in (5, 15, 35, 75, 135, 195, 255) if at <= t1 - t0)
    assert hidden and not mesh, (hidden, mesh)
    assert t1 - t0 >= 15 and len(probes) <= 1 + due + 1, (t1 - t0, probes)


def test_s6_with_a_server_the_page_never_steps_physics(page: Any) -> None:
    sim = Simulation()
    Robot(SPEC["robot"], sim=sim)
    server = Dashboard(sim).start()
    page.goto("about:blank")  # the standalone world stops; one browser, one heavy page at a time
    pg = page.context.browser.new_page(viewport={"width": 1400, "height": 860})
    try:
        pg.goto(server.url, wait_until="commit", timeout=60_000)
        pg.wait_for_function("window.__sceneLoaded >= 1 || window.__error", timeout=60_000)
        time.sleep(2.0)
        got = pg.evaluate(
            """({mode: document.querySelector('#mode').dataset.mode, loop: window.__standalone || null,
                pad: !document.querySelector('#pad').hidden,
                fetched: performance.getEntriesByType('resource').map((r) => r.name)})"""
        )
        assert got["mode"] == "server" and got["loop"] is None and not got["pad"], got
        assert not [u for u in got["fetched"] if "policy-worker" in u or "/vendor/ort" in u or "/scenes/" in u], got
    finally:
        pg.close()
        server.stop()
        sim.close()
