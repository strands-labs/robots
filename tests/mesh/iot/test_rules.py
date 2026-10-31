"""Rules as data (MAPPING 2.1–2.6): the exact SQL, idempotent apply, Basic Ingest telemetry bounded at 1 Hz."""

from __future__ import annotations

import json
import time
import uuid

import pytest

from strands_robots.mesh import wire
from strands_robots.mesh.iot import lifecycle, rules
from strands_robots.mesh.transports.iot_link import FakeIotBroker, FakeLink

ROLE = "arn:aws:iam::123456789012:role/strands-rules"  # placeholder: the user supplies the role
SNS = "arn:aws:sns:us-west-2:123456789012:x"


def _every_rule(tenant: str) -> dict:
    """All five rules a tenant can have: estop, events, fault and telemetry (``rules_for``) and lifecycle."""
    return {
        **rules.rules_for(tenant, ROLE, sns_topic_arn=SNS, s3_bucket="b", telemetry=True),
        **lifecycle.clear_presence_rule(tenant, ROLE),
    }


def test_rule_names_are_legal_and_sql_reads_the_tenant_keys():
    r = rules.rules_for("lab-a", ROLE, sns_topic_arn="arn:aws:sns:us-west-2:123456789012:x", s3_bucket="b")
    assert set(r) == {"strands_estop_lab_a", "strands_events_lab_a", "strands_fault_lab_a"}
    assert (
        r["strands_estop_lab_a"]["sql"] == "SELECT *, topic(2) AS tenant, timestamp() AS at FROM 'strands/lab-a/@estop'"
    )
    assert r["strands_estop_lab_a"]["actions"][0]["republish"]["topic"] == "strands/lab-a/@estop/latest"
    assert r["strands_events_lab_a"]["sql"].endswith("FROM 'strands/lab-a/+/event/+'")
    assert "WHERE result.severity = 'high'" in r["strands_fault_lab_a"]["sql"]
    assert all(n == n[:128] and n.replace("_", "a").isalnum() for n in r)


def test_telemetry_needs_a_sink_and_rides_basic_ingest():
    with pytest.raises(ValueError, match="needs a sink"):
        rules.rules_for("lab", ROLE, telemetry=True)
    r = rules.rules_for("lab", ROLE, timestream=("strands", "events"), telemetry=True)
    assert "strands_telemetry_lab" in r
    assert rules.basic_ingest_topic("lab", "arm") == "$aws/rules/strands_telemetry_lab/strands/lab/arm/state"


def test_apply_is_idempotent_and_replaces_on_change(aws):
    r = rules.rules_for("lab", ROLE, s3_bucket="bucket")
    assert rules.apply_rules(aws, r) == {"strands_estop_lab": "created", "strands_events_lab": "created"}
    assert rules.apply_rules(aws, r) == {"strands_estop_lab": "unchanged", "strands_events_lab": "unchanged"}
    r2 = rules.rules_for("lab", ROLE, s3_bucket="other-bucket")
    out = rules.apply_rules(aws, r2)
    assert out == {"strands_estop_lab": "unchanged", "strands_events_lab": "replaced"}
    got = aws.get_topic_rule(ruleName="strands_events_lab")["rule"]
    assert got["actions"][0]["s3"]["bucketName"] == "other-bucket"
    other = rules.rules_for("prod-lab", ROLE, s3_bucket="bucket")  # normalises to the suffix "_prod_lab"
    assert set(rules.apply_rules(aws, other)) == {"strands_estop_prod_lab", "strands_events_prod_lab"}
    assert rules.delete_rules(aws, "lab") == ["strands_estop_lab", "strands_events_lab"]
    assert sorted(r["ruleName"] for r in aws.list_topic_rules()["rules"]) == sorted(other)
    assert rules.delete_rules(aws, "prod-lab") == sorted(other)
    assert aws.list_topic_rules()["rules"] == []


def test_delete_rules_deletes_every_rule_of_its_tenant_however_long_the_tenant(aws):
    """A rule name stops at 128 characters, and the 9-letter kinds (telemetry, lifecycle) reach it first: the
    delete still finds all five of the tenant's rules."""
    for n in (110, 111, 115, 122, 200):
        mine = _every_rule("t" * n)
        rules.apply_rules(aws, mine)
        assert rules.delete_rules(aws, "t" * n) == sorted(mine), n
        assert aws.list_topic_rules()["rules"] == [], n


def test_telemetry_publishes_state_envelopes_at_most_1_hz():
    from strands_robots import Robot, Simulation

    broker = FakeIotBroker.named(f"tel-{uuid.uuid4().hex[:8]}")
    device, sink = FakeLink(broker, "arm"), FakeLink(broker, "sink")
    got: list[dict] = []
    sink.subscribe("$aws/rules/+/strands/lab/arm/state", lambda t, raw: got.append(wire.decode_envelope(raw)))
    arm = Robot("so101", sim=Simulation())
    with pytest.raises(ValueError, match="over 1 Hz"):
        rules.telemetry(device, "lab", "arm", arm, 10)
    undo = rules.telemetry(device, "lab", "arm", arm, 1.0)
    time.sleep(2.2)
    undo()
    assert 2 <= len(got) <= 3, len(got)
    env = got[-1]
    assert (env["tenant"], env["device_id"], env["kind"]) == ("lab", "arm", "state")  # the ids ride in the payload
    assert "shoulder_pan.pos" in env["result"]
    assert len(json.dumps(env)) < 2048
    device.close()
    sink.close()
    arm.sim.close()
