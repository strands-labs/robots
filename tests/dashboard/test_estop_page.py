"""The page's E-STOP stops a run (stopped_by="estop"), shows LATCHED, and resume needs an operator's name."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

from .browser import page_on  # noqa: E402
from .test_twin_page import MEDIA  # noqa: E402


def test_estop_button_stops_the_run_latches_and_resume_needs_approval() -> None:
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    server = Dashboard(sim).start()
    try:
        with page_on(server.url) as page:
            row = page.locator('.card .estop-ctl[data-robot="so101"]')
            row.wait_for(timeout=10_000)
            assert row.locator(".estop-status").inner_text() == "armed"
            task = arm.start(lambda obs: {"shoulder_pan": 0.2}, hz=50, seconds=60, realtime=True)
            assert task.running
            row.locator("button.estop").click()
            page.wait_for_function("window.__safety.notices.length > 0", timeout=10_000)
            assert arm.estopped, page.evaluate("window.__safety.notices")
            page.wait_for_function(
                "document.querySelector('.estop-ctl[data-robot=\"so101\"]').dataset.latched === 'yes'", timeout=10_000
            )
            assert row.locator(".estop-status").inner_text() == "LATCHED"
            result = task.stop(timeout=5.0)
            assert result is not None and result.stopped_by == "estop"
            page.screenshot(path=str(MEDIA / "row4-estop-latched.png"))

            page.once("dialog", lambda d: d.accept(""))  # no operator name: refused
            row.locator("button.resume").click()
            page.wait_for_function("window.__safety.notices.some(n => n.includes('approved_by'))", timeout=5_000)
            assert arm.estopped and row.locator(".estop-status").inner_text() == "LATCHED"

            page.once("dialog", lambda d: d.accept("cagatay"))
            row.locator("button.resume").click()
            page.wait_for_function(
                "document.querySelector('.estop-ctl[data-robot=\"so101\"]').dataset.latched === 'no'", timeout=5_000
            )
            assert not arm.estopped and arm.estop_latch.resumed_by == "cagatay"
            assert not page.errors, page.errors
    finally:
        server.stop()
        sim.close()


def test_a_hardware_cards_latch_is_the_real_robots_not_its_twins() -> None:
    """The card of a hardware robot is its twin. E-STOP on that card latches the REAL robot (``_robot`` resolves it),
    so the frame the panel paints must report the real latch too, or a latched arm shows ``armed``."""
    import test_drivers

    d, _serial = test_drivers.make_driver()
    real = Robot("so101", driver=d, id="real_arm", twin=True)
    sim = real.twin.sim
    dash = Dashboard(sim, robots=[real])
    try:
        card = real.twin.id
        assert dash.frame()["estop"] == {card: False}
        assert dash.estop(card)["estop"] is True
        assert real.estopped and not sim.robots[card].estopped  # the latch lives on the real robot only
        assert dash.frame()["estop"] == {card: True}, "the frame must resolve the card to the real robot"
        dash.resume(card, approved_by="alice", confirmed=True)  # a real body: the page sends X-Confirm
        assert dash.frame()["estop"] == {card: False}
    finally:
        real.close()
        sim.close()
