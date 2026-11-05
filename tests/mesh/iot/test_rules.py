"""Rules as data (MAPPING 2.1–2.6): the exact SQL, idempotent apply, Basic Ingest telemetry bounded at 1 Hz."""

from __future__ import annotations

import itertools
import json
import re
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
        **lifecycle.clear_presence_rule(tenant, ROLE, things=["arm-1"]),
    }


def test_rule_names_are_legal_and_sql_reads_the_tenant_keys():
    r = rules.rules_for("lab-a", ROLE, sns_topic_arn="arn:aws:sns:us-west-2:123456789012:x", s3_bucket="b")
    # a rule name has no "-": "lab_a", then 12 hex digits of sha256("lab-a") (the tenant lab_a gets other digits)
    estop, events, fault = (f"strands_{kind}_lab_a_1c327e55b394" for kind in ("estop", "events", "fault"))
    assert set(r) == {estop, events, fault}
    assert r[estop]["sql"] == "SELECT *, topic(2) AS tenant, timestamp() AS at FROM 'strands/lab-a/@estop'"
    assert r[estop]["actions"][0]["republish"]["topic"] == "strands/lab-a/@estop/latest"
    assert r[events]["sql"].endswith("FROM 'strands/lab-a/+/event/+'")
    assert "WHERE result.severity = 'high'" in r[fault]["sql"]
    assert all(n == n[:128] and n.replace("_", "a").isalnum() for n in r)


def test_telemetry_needs_a_sink_and_rides_basic_ingest():
    with pytest.raises(ValueError, match="needs a sink"):
        rules.rules_for("lab", ROLE, telemetry=True)
    r = rules.rules_for("lab", ROLE, timestream=("strands", "events"), telemetry=True)
    assert "strands_telemetry_lab" in r
    assert rules.basic_ingest_topic("lab", "arm") == "$aws/rules/strands_telemetry_lab/strands/lab/arm/state"
    hashed = "strands_telemetry_lab_a_1c327e55b394"  # a device of lab-a publishes into the rule rules_for names
    assert hashed in rules.rules_for("lab-a", ROLE, s3_bucket="b", telemetry=True)
    assert rules.basic_ingest_topic("lab-a", "arm") == f"$aws/rules/{hashed}/strands/lab-a/arm/state"


def test_apply_is_idempotent_and_replaces_on_change(aws):
    r = rules.rules_for("lab", ROLE, s3_bucket="bucket")
    assert rules.apply_rules(aws, r) == {"strands_estop_lab": "created", "strands_events_lab": "created"}
    assert rules.apply_rules(aws, r) == {"strands_estop_lab": "unchanged", "strands_events_lab": "unchanged"}
    r2 = rules.rules_for("lab", ROLE, s3_bucket="other-bucket")
    out = rules.apply_rules(aws, r2)
    assert out == {"strands_estop_lab": "unchanged", "strands_events_lab": "replaced"}
    got = aws.get_topic_rule(ruleName="strands_events_lab")["rule"]
    assert got["actions"][0]["s3"]["bucketName"] == "other-bucket"
    other = rules.rules_for("prod-lab", ROLE, s3_bucket="bucket")  # "_prod_lab_" and 12 hex digits of its sha256
    assert set(rules.apply_rules(aws, other)) == {
        "strands_estop_prod_lab_934913edb48c",
        "strands_events_prod_lab_934913edb48c",
    }
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


class _Paged:
    """A boto3 ``iot`` client whose ``list_topic_rules`` answers ``page`` rules at a time, as the service may (moto
    answers every rule at once)."""

    def __init__(self, iot, page: int):
        self.iot, self.page = iot, page

    def list_topic_rules(self, nextToken: str = "0") -> dict:
        everything, start = self.iot.list_topic_rules()["rules"], int(nextToken)
        out = {"rules": everything[start : start + self.page]}
        if start + self.page < len(everything):
            out["nextToken"] = str(start + self.page)
        return out

    def delete_topic_rule(self, ruleName: str) -> dict:
        return self.iot.delete_topic_rule(ruleName=ruleName)


def test_delete_rules_reads_every_page_of_the_accounts_rules(aws):
    """20 rules, 4 to a page: the tenant whose rules come last still finds and deletes all five, and only those."""
    for t in ("alpha", "beta", "gamma", "zeta-lab"):
        rules.apply_rules(aws, _every_rule(t))
    assert rules.delete_rules(_Paged(aws, 4), "zeta-lab") == sorted(_every_rule("zeta-lab"))
    left = sorted(r["ruleName"] for r in aws.list_topic_rules()["rules"])
    assert left == sorted(n for t in ("alpha", "beta", "gamma") for n in _every_rule(t))


FOLD_ALIKE = [("lab-a", "lab_a"), ("lab_a", "lab-a"), ("lab.a", "lab-a"), ("lab:a", "lab_a"), ("john.doe", "john_doe")]


def _stored(iot, name: str) -> tuple:
    got = iot.get_topic_rule(ruleName=name)["rule"]
    return got["sql"], got["description"], got["actions"]


@pytest.mark.parametrize(("mine", "theirs"), FOLD_ALIKE)
def test_a_tenant_that_folds_alike_never_replaces_another_tenants_rules(aws, mine, theirs):
    """``- . :`` all become ``_`` in a rule name, yet each tenant's apply creates rules of its own and leaves the
    other tenant's as that tenant wrote them (its e-stop fan-out still reads its own ``@estop``)."""
    rules.apply_rules(aws, _every_rule(mine))
    assert set(rules.apply_rules(aws, _every_rule(theirs)).values()) == {"created"}
    for name, payload in _every_rule(mine).items():
        assert _stored(aws, name) == (payload["sql"], payload["description"], payload["actions"]), name


@pytest.mark.parametrize(("mine", "theirs"), FOLD_ALIKE)
def test_a_tenant_that_folds_alike_never_deletes_another_tenants_rules(aws, mine, theirs):
    rules.apply_rules(aws, _every_rule(mine))
    assert rules.delete_rules(aws, theirs) == []
    assert sorted(r["ruleName"] for r in aws.list_topic_rules()["rules"]) == sorted(_every_rule(mine))


def test_two_long_tenants_that_start_alike_have_rules_of_their_own(aws):
    """122 characters that differ only in the last one: past the 128-character cut each still has its five rules."""
    a, b = "t" * 120 + "-a", "t" * 120 + "-b"
    assert [len(n) for n in sorted(set(_every_rule(a)) & set(_every_rule(b)))] == []  # names both tenants have
    rules.apply_rules(aws, _every_rule(a))
    assert set(rules.apply_rules(aws, _every_rule(b)).values()) == {"created"}
    assert _stored(aws, rules.rule_name("estop", a))[0].endswith(f"FROM 'strands/{a}/@estop'")
    assert rules.delete_rules(aws, b) == sorted(_every_rule(b))
    assert sorted(r["ruleName"] for r in aws.list_topic_rules()["rules"]) == sorted(_every_rule(a))


def test_every_rule_name_is_legal_and_names_one_tenant():
    """Every legal tenant of up to 4 characters over ``a _ - . :``, and long ones: each name is a legal rule name and
    no two (kind, tenant) pairs share one. A tenant of letters and digits, up to 110 of them, keeps its name."""
    tenants = ["".join(p) for n in range(1, 5) for p in itertools.product("a_-.:", repeat=n)]
    tenants += [c * n + end for c in "tT" for n in (97, 109, 110, 111, 128, 200) for end in ("", "-a", "_a", ".a")]
    tenants += ["lab-a", "lab_a_1c327e55b394"]  # the second spells the first's hashed part: "_" is never plain
    names = [
        rules.rule_name(kind, t) for kind in ("estop", "events", "fault", "telemetry", "lifecycle") for t in tenants
    ]
    assert all(re.fullmatch(r"[A-Za-z0-9_]{1,128}", n) for n in names)
    assert len(names) - len(set(names)) == 0  # names another (kind, tenant) already has
    assert rules.rule_name("lifecycle", "t" * 110) == "strands_lifecycle_" + "t" * 110
    assert rules.rule_name("estop", "Lab2") == "strands_estop_Lab2"
    with pytest.raises(ValueError, match="unknown rule kind"):
        rules.rule_name("x", "lab")


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
