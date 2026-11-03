"""The Connect panel in a browser: join a peer's mesh by address, attach its arm, move it (the operator is asked), see
its camera, e-stop it with one click (the device latches), resume it; a policy spec typed in shows in the run picker."""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402
from strands_robots.mesh import Mesh  # noqa: E402
from strands_robots.mesh.authority import Authority  # noqa: E402

from .browser import page_on  # noqa: E402

ARM = '#cx-robots .rcard[data-robot="far_arm"]'


def _robot(page, rid="far_arm"):
    return page.evaluate(f"window.__connect.robots.find((r) => r.id === {rid!r}) || null")


def test_the_connect_panel_attaches_moves_estops_and_resumes_a_peers_arm() -> None:
    bus, tenant = f"mem://page-{uuid.uuid4().hex[:8]}", f"t-{uuid.uuid4().hex[:6]}"
    far = Simulation()
    arm = Robot("so101", sim=far, id="far_arm")
    far.camera("wrist_view", attach=arm, size=(96, 72))
    peer = Mesh(bus, tenant=tenant, resume_authority=Authority(trust_names=True))
    peer.mount(arm)
    sim = Simulation()
    Robot("so101", sim=sim, id="local_arm")
    dash = Dashboard(sim)
    dash.connections.join(bus, tenant)  # this test's own tenant: the join box would use the login name
    server = dash.start()
    asked: list[str] = []
    try:
        with page_on(server.url) as page:
            page.on("dialog", lambda d: (asked.append(d.message), d.accept()))
            page.wait_for_function("window.__connect && window.__connect.ready", timeout=120_000)
            page.evaluate("window.SR.select('connect')")  # a side-column tab: hidden until picked
            page.locator('#cx-peers [data-attach="far_arm"]').dispatch_event("click")
            page.wait_for_selector(f"{ARM} img", timeout=30_000)
            page.wait_for_function(
                "document.querySelector('#cx-robots img').naturalWidth === 96", timeout=30_000
            )  # the peer's own camera frame, not a render of this world

            page.locator(f'{ARM} input[data-joint="shoulder_pan"]').evaluate(
                "el => { el.value = '0.6'; el.dispatchEvent(new Event('change', { bubbles: true })); }"
            )
            page.wait_for_function("(window.__connect.acts || []).length === 1", timeout=30_000)
            assert asked and "real robot" in asked[0]  # the operator is asked before a remote arm moves
            assert page.evaluate("window.__connect.acts[0].status") == 200
            assert arm.observe(images=False).joints["shoulder_pan"] == pytest.approx(0.6, abs=0.1)

            n = len(asked)
            page.locator(f'{ARM} [data-estop="far_arm"]').dispatch_event("click")
            page.wait_for_function(
                "window.__connect.robots.find((r) => r.id === 'far_arm').estop === true", timeout=30_000
            )
            assert arm.estopped and len(asked) == n  # latched on the device; stopping never asks

            page.locator(f'{ARM} [data-resume="far_arm"]').dispatch_event("click")
            page.wait_for_function(
                "window.__connect.robots.find((r) => r.id === 'far_arm').estop === false", timeout=30_000
            )
            assert not arm.estopped and "Release the e-stop" in asked[-1]

            page.fill("#cx-endpoint", "lerobot/act_so101")
            page.select_option("#cx-kind", "lerobot")
            page.locator("#cx-policy").dispatch_event("click")
            page.wait_for_selector('#cx-run-policy option[value="lerobot/act_so101"]', state="attached", timeout=30_000)
            robots = page.eval_on_selector_all("#cx-run-robot option", "os => os.map((o) => o.value)")
            assert set(robots) == {"local_arm", "far_arm"}
            assert not page.errors, page.errors
    finally:
        server.stop()
        dash.connections.close()
        peer.close()
        far.close()
        sim.close()
