"""/ws/voice: microphone PCM in, a BidiAgent holding the chat tools, transcripts + tool calls + audio out.

A scripted bidi model (no network) hears the first audio and calls the tools; the world changes as it would for
the real models (a live run against gpt-realtime-2 or Nova Sonic is manual, not in CI).
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
pytest.importorskip("strands.bidi")
websockets_sync = pytest.importorskip("websockets.sync.client")

from tools.scripted_bidi import ScriptedBidi  # noqa: E402

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard, voice  # noqa: E402

SPEECH = (b"\x10\x27" * 320) * 5  # 100 ms of non-silent PCM16 at 16 kHz


def talk(url: str, *, frames: bytes = SPEECH, until: str = "turn_done", n: int = 2, wait: float = 30.0) -> list[Any]:
    """Stream ``frames`` like the page does; collect events until ``n`` ``until`` events (or ``wait`` s)."""
    events: list[Any] = []
    with websockets_sync.connect(url.replace("http", "ws") + "/ws/voice", max_size=None) as ws:
        events.append(json.loads(ws.recv(timeout=10)))
        if events[0]["type"] != "ready":
            return events
        ws.send(frames)
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait and sum(e.get("type") == until for e in events) < n:
            try:
                m = ws.recv(timeout=1)
            except TimeoutError:
                continue
            events.append({"type": "audio", "bytes": len(m)} if isinstance(m, bytes) else json.loads(m))
        ws.send(json.dumps({"type": "bye"}))
    return events


def test_voice_opens_the_gripper_through_robot_tool() -> None:
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    models: list[ScriptedBidi] = []

    def model() -> ScriptedBidi:
        m = ScriptedBidi(
            heard="open the gripper of so101",
            calls=[("robot_so101", {"rpc": "act", "act": {"action": {"gripper": 1.5}, "seconds": 0.5}})],
            say="The gripper is open.",
        )
        models.append(m)
        return m

    server = Dashboard(sim, voice=model).start()
    try:
        events = talk(server.url)
        kinds = [e["type"] for e in events]
        assert events[0] == {"type": "ready", "provider": "custom", "model": "scripted-bidi",
                             "input_rate": 16000, "output_rate": 16000}  # fmt: skip
        assert "tool_use" in kinds and "tool_result" in kinds and "audio" in kinds, events
        use = next(e for e in events if e["type"] == "tool_use")
        assert use["name"] == "robot_so101" and use["input"]["rpc"] == "act"
        res = next(e for e in events if e["type"] == "tool_result")
        assert res["status"] == "success" and res["name"] == "robot_so101", res
        said = [e for e in events if e["type"] == "transcript" and e.get("final")]
        assert {e["role"] for e in said} == {"user", "assistant"}, said
        assert arm.observe(images=False)["gripper"] == pytest.approx(1.5, abs=0.05)
        offered = set(models[0].tools_seen)
        assert {"robot_so101", "sim_world", "dashboard_run", "dashboard_stop", "dashboard_estop"} <= offered
        assert "English" in (models[0].system_prompt or "")
    finally:
        server.stop()
        sim.close()


def test_voice_starts_a_background_walk_and_stops_it() -> None:
    pytest.importorskip("onnxruntime")
    sim = Simulation()
    g1 = Robot("unitree_g1", sim=sim, id="g1")

    def model() -> ScriptedBidi:
        return ScriptedBidi(
            heard="start walking forward in g1",
            calls=[("dashboard_run", {"robot": "g1", "policy": "holosoma?vx=0.4"})],
        )

    server = Dashboard(sim, voice=model).start()
    try:
        x0 = float(g1.observe(images=False).pose.position[0])
        events = talk(server.url)
        res = next(e for e in events if e["type"] == "tool_result")
        assert res["status"] == "success" and '"running"' in res["text"], res
        assert g1._task is not None and g1._task.running  # the call returned while the robot walks
        time.sleep(3.0)
        assert float(g1.observe(images=False).pose.position[0]) - x0 > 0.3
    finally:
        if g1._task is not None:
            g1._task.halt()
        server.stop()
        sim.close()


def test_voice_never_confirms_a_real_robot() -> None:
    import test_drivers
    from test_operator_gate import goal_writes

    sim = Simulation()
    d, serial = test_drivers.make_driver()
    real = Robot("so101", driver=d, id="real_arm", twin=False)
    calls = [
        ("robot_real_arm", {"rpc": "act", "act": {"action": {"shoulder_pan": 0.2}}}),
        ("dashboard_run", {"robot": "real_arm", "policy": "lerobot/act_so101"}),
    ]
    server = Dashboard(sim, robots=[real], voice=lambda: ScriptedBidi(heard="move", calls=calls)).start()
    try:
        before = len(goal_writes(serial))
        events = talk(server.url, until="tool_result", n=2)
        results = [e for e in events if e["type"] == "tool_result"]
        assert len(results) == 2 and all(r["status"] == "error" for r in results), results
        assert len(goal_writes(serial)) == before  # nothing reached the bus
        assert real._task is None or not real._task.running
    finally:
        server.stop()
        real.close()
        sim.close()


def test_voice_estop_tool_latches() -> None:
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    calls = [("dashboard_estop", {"robot": "so101"})]
    server = Dashboard(sim, voice=lambda: ScriptedBidi(heard="stop", calls=calls)).start()
    try:
        talk(server.url, until="tool_result", n=1)
        assert arm.estopped
    finally:
        server.stop()
        sim.close()


def test_voice_refuses_a_network_policy_from_the_model() -> None:
    sim = Simulation()
    Robot("so101", sim=sim)
    calls = [("dashboard_run", {"robot": "so101", "policy": "zenoh://10.0.0.1:7447/act"})]
    server = Dashboard(sim, voice=lambda: ScriptedBidi(heard="run", calls=calls)).start()
    try:
        res = next(e for e in talk(server.url, until="tool_result", n=1) if e["type"] == "tool_result")
        assert res["status"] == "error" and "network address" in res["text"], res
    finally:
        server.stop()
        sim.close()


def test_voice_refuses_a_hub_checkpoint_the_operator_did_not_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRANDS_POLICY_REPOS", raising=False)
    sim = Simulation()
    Robot("so101", sim=sim)
    calls = [("dashboard_run", {"robot": "so101", "policy": "someone/ckpt"})]
    server = Dashboard(sim, voice=lambda: ScriptedBidi(heard="run", calls=calls)).start()
    try:
        res = next(e for e in talk(server.url, until="tool_result", n=1) if e["type"] == "tool_result")
        assert res["status"] == "error" and "STRANDS_POLICY_REPOS" in res["text"], res
    finally:
        server.stop()
        sim.close()


def test_voice_off_says_how_to_turn_it_on() -> None:
    import httpx

    sim = Simulation()
    server = Dashboard(sim).start()
    try:
        info = httpx.get(f"{server.url}/api/voice").json()
        assert info["enabled"] is False and "voice=" in info["hint"]
        events = talk(server.url)
        assert events[0]["type"] == "error" and "voice=" in events[0]["message"]
    finally:
        server.stop()
        sim.close()


def test_voice_spec_is_checked_and_auto_picks_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = Simulation()
    with pytest.raises(ValueError, match="voice must be"):
        Dashboard(sim, voice="gpt-4")
    monkeypatch.delenv("STRANDS_VOICE_SECRETS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert voice.provider_of("auto") == "openai"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert voice.provider_of("auto") == "nova"
    assert voice.provider_of(lambda: None) == "custom"
    with pytest.raises(RuntimeError, match="voice="):
        voice.make_model(None)
    sim.close()


def test_page_events_map_the_agent_stream() -> None:
    names: dict[str, str] = {}
    use = {
        "type": "bidi_tool_use_blocks",
        "tool_uses": [{"toolUseId": "t1", "name": "robot_x", "input": {"rpc": "observe"}}],
    }
    assert voice.page_events(use, names) == [{"type": "tool_use", "name": "robot_x", "input": {"rpc": "observe"}}]
    res = {"type": "tool_result", "tool_result": {"toolUseId": "t1", "status": "success", "content": [{"text": "ok"}]}}
    assert voice.page_events(res, names)[0] == {
        "type": "tool_result",
        "name": "robot_x",
        "status": "success",
        "text": "ok",
    }
    audio = {"type": "bidi_audio_delta", "audio": "AQA=", "format": "pcm", "sample_rate": 24000, "channels": 1}
    assert voice.page_events(audio, names) == [b"\x01\x00"]
    assert voice.page_events({"type": "bidi_barge_in"}, names) == [{"type": "interrupt"}]
    assert voice.page_events({"type": "bidi_usage"}, names) == []


def test_page_mic_button_streams_and_logs_the_turn() -> None:
    """Chromium's fake microphone (a tone) through voice.js -> /ws/voice -> the scripted model -> the chat log."""
    sync_api = pytest.importorskip("playwright.sync_api")
    from .browser import sign_in
    from .test_twin_page import MEDIA

    sim = Simulation()
    arm = Robot("so101", sim=sim)
    calls = [("robot_so101", {"rpc": "act", "act": {"action": {"gripper": 1.2}, "seconds": 0.5}})]
    server = Dashboard(sim, voice=lambda: ScriptedBidi(heard="open the gripper", calls=calls, say="Opened.")).start()
    try:
        with sync_api.sync_playwright() as p:
            try:
                browser = p.chromium.launch(
                    args=[
                        "--use-gl=angle",
                        "--use-angle=swiftshader",
                        "--enable-unsafe-swiftshader",
                        "--use-fake-ui-for-media-stream",
                        "--use-fake-device-for-media-stream",
                        "--autoplay-policy=no-user-gesture-required",
                    ]
                )
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"chromium unavailable: {exc}")
            ctx = browser.new_context(viewport={"width": 1400, "height": 860}, permissions=["microphone"])
            sign_in(ctx, server.url)
            page = ctx.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(server.url)
            # app.js imports the feature modules (GET /api/panels) once the scene is up, so the scene comes first.
            page.wait_for_function("() => window.__sceneLoaded >= 1", timeout=90_000)
            page.wait_for_function("() => window.__voiceInfo && window.__voiceInfo.enabled", timeout=30_000)
            assert page.query_selector("#chat-slot [data-panel=voice] #voice-mic"), "voice not in the chat-input slot"

            def press(selector: str) -> None:  # mouse.click: .click()'s stability wait stalls at swiftshader's ~1 fps
                box = page.locator(selector).bounding_box()
                assert box, selector
                page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

            press("#voice-mic")
            page.wait_for_function(
                "() => (window.__voiceEvents || []).some(e => e.type === 'transcript' && e.role === 'assistant')",
                timeout=30_000,
            )
            assert page.get_attribute("#voice", "data-state") == "live"
            log = page.inner_text("#log")
            assert "open the gripper" in log and "robot_so101" in log and "Opened." in log, log
            assert arm.observe(images=False)["gripper"] == pytest.approx(1.2, abs=0.05)
            press("#voice-mute")
            assert page.get_attribute("#voice", "data-muted") == "true"
            page.screenshot(path=str(MEDIA / "voice-panel.png"))
            press("#voice-mic")
            page.wait_for_function("() => document.querySelector('#voice').dataset.state === 'off'", timeout=10_000)
            assert not errors, errors
            browser.close()
    finally:
        server.stop()
        sim.close()
