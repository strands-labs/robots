"""A REAL model turn through /ws/chat drives the sim arm (opt-in: STRANDS_ROBOTS_LIVE=1 + Bedrock credentials)."""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

MODEL_ID = os.environ.get("STRANDS_ROBOTS_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
REGION = os.environ.get("AWS_REGION", "us-west-2")


def _model():
    if os.environ.get("STRANDS_ROBOTS_LIVE") != "1":
        pytest.skip("live model turn is opt-in: set STRANDS_ROBOTS_LIVE=1 (needs Bedrock credentials)")
    from strands.models.bedrock import BedrockModel

    return BedrockModel(model_id=MODEL_ID, region_name=REGION, max_tokens=512, temperature=0.0)


def test_a_real_model_moves_the_arm_through_the_chat_socket() -> None:
    import websockets.sync.client as wsc

    sim = Simulation()
    arm = Robot("so101", sim=sim)
    server = Dashboard(sim, model=_model()).start()
    events: list[dict] = []
    try:
        with wsc.connect(server.url.replace("http", "ws") + "/ws/chat", open_timeout=10) as ws:
            ws.send(
                json.dumps({"prompt": "Set so101 shoulder_pan to 0.3 rad with one act call (steps 200). Then stop."})
            )
            while not events or events[-1]["type"] != "done":
                events.append(json.loads(ws.recv(timeout=120)))
    finally:
        server.stop()
    uses = [e for e in events if e["type"] == "tool_use"]
    errors = [e for e in events if e["type"] == "error"]
    if errors and "credential" in errors[0]["message"].lower():
        pytest.skip(f"needs creds (owner): {errors[0]['message']}")
    assert uses and not errors, events
    pan = arm.observe(images=False)["shoulder_pan"]
    sim.close()
    print(
        f"[measured] live chat: {len(uses)} tool_use ({[u['input'].get('rpc') for u in uses]}), shoulder_pan={pan:.3f}"
    )
    assert pan == pytest.approx(0.3, abs=0.06)
