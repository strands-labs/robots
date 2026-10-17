"""Gate row 3: a REAL model turn drives a sim so101 through robot.tool (opt-in; skipped with the reason otherwise).

Run with ``STRANDS_ROBOTS_LIVE=1`` and AWS credentials for Bedrock (``STRANDS_ROBOTS_MODEL_ID`` picks the model).
"""

from __future__ import annotations

import os

import pytest

from strands_robots import Robot

MODEL_ID = os.environ.get("STRANDS_ROBOTS_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
REGION = os.environ.get("AWS_REGION", "us-west-2")


def _model():
    if os.environ.get("STRANDS_ROBOTS_LIVE") != "1":
        pytest.skip("live model turn is opt-in: set STRANDS_ROBOTS_LIVE=1 (needs Bedrock credentials)")
    try:
        import boto3

        boto3.client("bedrock-runtime", region_name=REGION).converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": "ok"}]}],
            inferenceConfig={"maxTokens": 1},
        )
    except Exception as exc:  # noqa: BLE001 - any failure means no reachable model
        pytest.skip(f"no Bedrock model reachable ({MODEL_ID} in {REGION}): {type(exc).__name__}: {exc}")
    from strands.models.bedrock import BedrockModel

    return BedrockModel(model_id=MODEL_ID, region_name=REGION, max_tokens=512, temperature=0.0)


def test_a_real_model_observes_then_moves_the_shoulder_pan():
    from strands import Agent

    model = _model()
    arm = Robot("so101")
    agent = Agent(
        model=model,
        tools=[arm.tool],
        callback_handler=None,
        system_prompt="You control a simulated robot arm through one tool. Be brief.",
    )
    agent(
        "First observe the robot. Then act once to set shoulder_pan to 0.5 radians with steps=200. "
        "Reply with the final shoulder_pan value."
    )
    calls = [b["toolUse"] for m in agent.messages for b in m["content"] if "toolUse" in b]
    rpcs = [c["input"].get("rpc") for c in calls]
    assert "observe" in rpcs and "act" in rpcs and rpcs.index("observe") < rpcs.index("act"), rpcs
    assert arm.observe()["shoulder_pan"] == pytest.approx(0.5, abs=0.05)
