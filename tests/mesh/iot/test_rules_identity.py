"""What the storage rules file a record under comes from the topic, the connection or the rule, never the body."""

from __future__ import annotations

import json

from strands_robots.mesh.iot import rules

ROLE = "arn:aws:iam::123456789012:role/strands-rules"
SNS = "arn:aws:sns:us-west-2:123456789012:x"


def _all(tenant: str = "lab-a") -> dict:
    return rules.rules_for(tenant, ROLE, sns_topic_arn=SNS, s3_bucket="b", timestream=("db", "t"), telemetry=True)


def _rule(r: dict, kind: str, tenant: str = "lab-a") -> dict:
    return r[rules.rule_name(kind, tenant)]


def test_events_and_fault_rules_bind_the_tenant_and_device_to_the_topic():
    r = _all()
    events, fault = _rule(r, "events")["sql"], _rule(r, "fault")["sql"]
    assert events.startswith("SELECT *, topic(2) AS tenant, topic(3) AS device_id, topic(5) AS event, ")
    assert fault.startswith("SELECT *, topic(2) AS tenant, topic(3) AS device_id ")


def test_storage_actions_never_substitute_a_body_field():
    """Every ``${...}`` in a storage action is a broker function (topic, client id, time), never a body column."""
    r = _all()
    for kind, device in (("events", "${topic(3)}"), ("telemetry", "${clientid()}")):
        actions = _rule(r, kind)["actions"]
        s3 = next(a["s3"] for a in actions if "s3" in a)
        ts = next(a["timestream"] for a in actions if "timestream" in a)
        assert s3["key"].startswith(f"recordings/lab-a/{device}/"), s3["key"]
        assert ts["dimensions"] == [{"name": "tenant", "value": "lab-a"}, {"name": "device_id", "value": device}]
        text = json.dumps(actions)
        for column in ("${tenant}", "${device_id}", "${event}"):
            assert column not in text, (kind, column)


def test_telemetry_takes_no_identity_from_the_body():
    sql = _rule(_all(), "telemetry")["sql"]
    body = sql.split(" FROM ")[0].removeprefix("SELECT ")
    columns = [c.strip() for c in body.split(",")]
    assert columns == ["seq", "t", "result", "'lab-a' AS tenant", "clientid() AS device_id"]


def test_a_tenant_only_ever_files_under_itself():
    """Two tenants sharing a bucket: each rule writes only under its own prefix, whatever a device sends."""
    for tenant in ("lab-a", "lab-b", "acme:plant.1"):
        for name, payload in _all(tenant).items():
            for action in payload["actions"]:
                if "s3" in action:
                    assert action["s3"]["key"].startswith(f"recordings/{tenant}/"), name
                if "timestream" in action:
                    assert action["timestream"]["dimensions"][0] == {"name": "tenant", "value": tenant}, name
