"""/ws/chat: a scripted model (no network) turns a typed prompt into a tool_use on robot.tool; the arm moves."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
pytest.importorskip("strands")

from tools.scripted_model import ScriptedModel  # noqa: E402

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

from .browser import page_on  # noqa: E402
from .test_twin_page import MEDIA  # noqa: E402


def test_chat_produces_a_tool_use_and_the_arm_moves() -> None:
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    model = ScriptedModel([("robot_so101", {"rpc": "act", "act": {"action": {"shoulder_pan": 0.4}, "steps": 200}})])
    server = Dashboard(sim, model=model).start()
    try:
        with page_on(server.url) as page:
            page.fill("#prompt", "turn the base a little")
            page.press("#prompt", "Enter")
            page.wait_for_function("(window.__chatEvents || []).some(e => e.type === 'done')", timeout=30_000)
            events = page.evaluate("window.__chatEvents")
            kinds = [e["type"] for e in events]
            assert "tool_use" in kinds and "tool_result" in kinds, events
            use = next(e for e in events if e["type"] == "tool_use")
            assert use["name"] == "robot_so101" and use["input"]["rpc"] == "act"
            result = next(e for e in events if e["type"] == "tool_result")
            assert result["status"] == "success", result
            assert arm.observe(images=False)["shoulder_pan"] == pytest.approx(0.4, abs=0.05)
            offered = {t["name"] for t in model.seen[0]}
            assert "robot_so101" in offered and any(n.startswith("sim") for n in offered), offered
            page.screenshot(path=str(MEDIA / "row5-chat-tool-use.png"))
            assert not page.errors, page.errors
    finally:
        server.stop()
        sim.close()


def test_chat_without_a_model_says_so() -> None:
    sim = Simulation()
    Robot("so101", sim=sim)
    server = Dashboard(sim).start()
    try:
        with page_on(server.url) as page:
            page.fill("#prompt", "hello")
            page.press("#prompt", "Enter")
            page.wait_for_function("(window.__chatEvents || []).some(e => e.type === 'done')", timeout=10_000)
            errors = [e for e in page.evaluate("window.__chatEvents") if e["type"] == "error"]
            assert errors and "model=" in errors[0]["message"]
    finally:
        server.stop()
        sim.close()
