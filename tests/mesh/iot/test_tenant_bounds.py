"""Identity on an IoT link: the attributes the fleet policy reads cannot be set by a label, an operator link gets a
client id its tenant's policy allows, and a device answers rpcs only on its own tenant's topics."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import time

import pytest

from strands_robots.mesh.iot import claim, claim_hook, client_id, lifecycle, policy, rules
from strands_robots.mesh.transports import iot


def test_an_iam_link_makes_its_tenants_operator_client_id_and_refuses_another():
    assert client_id.link_client_id({}, "lab").startswith("strands-op-lab-")
    assert client_id.link_client_id({"client": "strands-op-lab-me"}, "lab") == "strands-op-lab-me"
    with pytest.raises(ValueError, match="strands-op-lab-"):
        client_id.link_client_id({"client": "strands-op-other-x"}, "lab")
    assert client_id.link_client_id({"client": "arm-1", "cert": "c.pem"}, "lab") == "arm-1"  # a device names its thing


def test_no_tenants_operator_ids_fall_under_another_tenants_prefix():
    tenants = ["lab", "lab-a", "lab-a-b", "laba", "lab_a", "a", "a-b", "lab.a", "lab:a"]
    prefix = {t: policy.operator_client_prefix(t) for t in tenants}
    for mine in tenants:
        own = client_id.default_client_id({}, mine)
        assert own.startswith(prefix[mine])
        for other in tenants:
            if other != mine:  # neither its policy's Connect nor its audit rule's startswith() admits my operators
                assert not own.startswith(prefix[other]) and not prefix[mine].startswith(prefix[other])
    assert prefix["lab-a"] == "strands-op-lab_a_1c327e55b394-" and prefix["lab"] == "strands-op-lab-"


def test_the_claim_template_writes_the_telemetry_rule_the_approval_names():
    attrs = claim.TEMPLATE["Resources"]["thing"]["Properties"]["AttributePayload"]
    assert attrs["telemetry_rule"] == {"Ref": "telemetry_rule"} and "telemetry_rule" in claim.TEMPLATE["Parameters"]
    ns: dict = {}
    exec(claim_hook.HOOK_SOURCE, ns)  # noqa: S102 - the shipped hook
    secret = "s3cret"
    entry = {"device_id": "arm-1", "tenant": "lab-a", "device_type": "robot_arm",
             "telemetry_rule": rules.rule_name("telemetry", "lab-a"),
             "secret_sha256": hashlib.sha256(secret.encode()).hexdigest()}  # fmt: skip
    ns.update(subject_cns=lambda pem: ["arm-1"], thing_exists=lambda n: False, allowed=lambda n: dict(entry),
              consume=lambda n: None)  # fmt: skip
    ask = {"device_id": "arm-1", "tenant": "lab-a", "device_type": "robot_arm", "claim_secret": secret,
           "telemetry_rule": "strands_telemetry_other"}  # fmt: skip
    out = ns["handler"]({"parameters": ask, "certificatePem": ""}, None)
    assert out["allowProvisioning"]  # a hyphenated tenant gets its real (hashed) rule, whatever the request said
    assert out["parameterOverrides"]["telemetry_rule"] == "strands_telemetry_lab_a_1c327e55b394"
    ns["allowed"] = lambda n: {k: v for k, v in entry.items() if k != "telemetry_rule"}
    assert not ns["handler"]({"parameters": ask, "certificatePem": ""}, None)["allowProvisioning"]


def test_the_operator_policy_admits_the_lifecycle_subscription_and_receives_only_its_devices():
    doc = policy.operator_policy("us-west-2", "123456789012", "lab", things=["arm-1"])
    acts = {a: [r for s in doc["Statement"] if s["Action"] == a for r in s["Resource"]]
            for a in ("iot:Subscribe", "iot:Receive")}  # fmt: skip
    subs = [r.split(":topicfilter/", 1)[1] for r in acts["iot:Subscribe"]]
    assert all(any(fnmatch.fnmatchcase(t, p) for p in subs) for t in lifecycle.TOPICS)  # "+" is literal in a policy
    recv = [r.split(":topic/", 1)[1] for r in acts["iot:Receive"]]
    assert any(fnmatch.fnmatchcase("$aws/events/presence/connected/arm-1", p) for p in recv)
    assert not any(fnmatch.fnmatchcase("$aws/events/presence/connected/arm-9", p) for p in recv)


def test_a_device_never_answers_an_rpc_on_another_tenants_topic():
    t = iot.IotTransport("iot://fake?broker=bounds&client=arm-1", tenant="lab")
    seen = []
    t.queryable("strands/*/arm-1/rpc/*", lambda key, body: seen.append(key) or b"{}", inline=True)
    caller = iot.IotTransport("iot://fake?broker=bounds&client=strands-op-other-1")
    for tenant in ("other", "lab"):
        body = {"id": "1", "reply": f"strands/{tenant}/@reply/strands-op-other-1", "body": "{}"}
        caller.publish(f"strands/{tenant}/arm-1/rpc/act", json.dumps(body).encode())  # "lab" last: proves both seen
    end = time.monotonic() + 5.0
    while not seen and time.monotonic() < end:
        time.sleep(0.02)
    time.sleep(0.2)
    assert seen == ["strands/lab/arm-1/rpc/act"]
    t.close()
    caller.close()


def test_a_device_certificate_link_sends_no_peer_beat_its_policy_cannot_publish():
    def beats(url: str, kind: str) -> list[bool]:
        heard: list[bool] = []
        ear = iot.IotTransport("iot://fake?broker=peerbeat&client=ear", presence_ttl=0.3)
        ear.watch_liveliness(f"strands/lab/{kind}/*", lambda key, alive: heard.append(alive))
        t = iot.IotTransport(url, presence_ttl=0.3)
        drop = t.liveliness(f"strands/lab/{kind}/arm-1")
        time.sleep(0.1)
        drop()  # its goodbye
        time.sleep(0.05)
        t.close()
        ear.close()
        return heard

    device = "iot://fake?broker=peerbeat&client=arm-1&cert=c.pem"
    assert beats(device, "@peer") == []  # nothing the fleet policy refuses ever reaches the broker
    assert beats(device, "@presence") == [True, False]  # its own presence still beats and says goodbye
    assert beats(device + "&peer=1", "@peer") == [True, False]  # a certificate that also carries controller_policy
    assert beats("iot://fake?broker=peerbeat&client=strands-op-lab-me", "@peer") == [True, False]  # an IAM operator
