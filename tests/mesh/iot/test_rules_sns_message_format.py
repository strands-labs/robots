"""An SNS action delivers the rule's record, not a refusal nobody hears.

With ``messageFormat: "JSON"`` SNS treats the message as a per-protocol JSON envelope and refuses any body without a
``"default"`` key ("No default entry in JSON message body"). A rule's SELECT output never has one, so every publish
failed, and a topic rule with no error action drops that failure silently: measured on a live tenant, a real e-stop
reached S3 through the same rule set and 0 messages reached SNS in 90 s. With ``RAW`` the same e-stop reached the
topic (and an SQS subscriber) in 1.06 s.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh.iot import rules

ROLE = "arn:aws:iam::123456789012:role/strands-rules"
SNS = "arn:aws:sns:us-west-2:123456789012:alerts"


def _sns_actions(payloads: dict) -> list[tuple[str, dict]]:
    return [(name, a["sns"]) for name, p in payloads.items() for a in p["actions"] if "sns" in a]


@pytest.mark.parametrize("tenant", ["lab-a", "b17"])
def test_every_sns_action_sends_the_record_as_the_raw_message_body(tenant):
    actions = _sns_actions(rules.rules_for(tenant, ROLE, sns_topic_arn=SNS, s3_bucket="b"))
    assert {name.split("_")[1] for name, _ in actions} == {"estop", "fault"}, "both SNS paths are covered"
    for name, sns in actions:
        assert sns["targetArn"] == SNS and sns["roleArn"] == ROLE
        assert sns["messageFormat"] == "RAW", (
            f"{name}: messageFormat {sns['messageFormat']!r} makes SNS refuse a body with no 'default' key, "
            "which every rule record is"
        )


def test_without_a_topic_there_is_no_sns_action():
    assert _sns_actions(rules.rules_for("lab-a", ROLE, s3_bucket="b")) == []
