"""Identity on an IoT link: the attributes the fleet policy reads cannot be set by a label, an operator link gets a
client id its tenant's policy allows, and a device answers rpcs only on its own tenant's topics."""

from __future__ import annotations

import json
import time

import pytest

from strands_robots.mesh.iot import claim, client_id, rules
from strands_robots.mesh.transports import iot


def test_an_iam_link_makes_its_tenants_operator_client_id_and_refuses_another():
    assert client_id.link_client_id({}, "lab").startswith("strands-op-lab-")
    assert client_id.link_client_id({"client": "strands-op-lab-me"}, "lab") == "strands-op-lab-me"
    with pytest.raises(ValueError, match="strands-op-lab-"):
        client_id.link_client_id({"client": "strands-op-other-x"}, "lab")
    assert client_id.link_client_id({"client": "arm-1", "cert": "c.pem"}, "lab") == "arm-1"  # a device names its thing


def test_the_claim_template_writes_the_telemetry_rule_the_policy_names():
    attrs = claim.TEMPLATE["Resources"]["thing"]["Properties"]["AttributePayload"]
    joined = "".join(p if isinstance(p, str) else "lab" for p in attrs["telemetry_rule"]["Fn::Join"][1])
    assert joined == rules.rule_name("telemetry", "lab")


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
