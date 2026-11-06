"""An IAM operator reaches a device by Direct Message, under the policy this package generates.

Measured against real AWS IoT Core: with ``iot:SendDirectMessage`` granted on ``connection/*`` every manifest
request came back 403, the fetch swallowed it, and ``discover()`` returned ``[]`` while the device's presence
heartbeats were arriving. AWS names the TARGET client as this action's resource; granting it on ``client/*``
(kept inside the tenant by an ``iot:Topic`` condition) discovered the device in 0.49 s.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import time

from strands_robots.mesh import wire
from strands_robots.mesh.iot import policy
from strands_robots.mesh.iot.client_id import default_client_id
from strands_robots.mesh.presence import Directory
from strands_robots.mesh.transports.base import RemoteError

ACCOUNT, REGION = "123456789012", "us-west-2"


def _doc():
    return json.loads(policy.policy_text(policy.operator_policy(REGION, ACCOUNT, "lab", things=["arm-1"])))


def test_direct_messages_are_granted_on_the_target_client_inside_the_tenant():
    stmt = [s for s in _doc()["Statement"] if s["Action"] == "iot:SendDirectMessage"]
    assert len(stmt) == 1
    assert stmt[0]["Resource"] == [f"arn:aws:iot:{REGION}:{ACCOUNT}:client/arm-1"]  # only the tenant's own things
    assert stmt[0]["Condition"] == {"StringLike": {"iot:Topic": ["strands/lab/*/rpc/*", "strands/lab/*/manifest"]}}


def test_a_direct_message_reaches_only_the_keys_an_operator_queries():
    """A direct message needs no subscription, so the topic condition is the only thing keeping an operator off
    the keys that are a device's word (presence, holder, state, events)."""
    (stmt,) = [s for s in _doc()["Statement"] if s["Action"] == "iot:SendDirectMessage"]
    allowed = stmt["Condition"]["StringLike"]["iot:Topic"]

    def admits(topic):  # IAM StringLike: * spans "/"
        return any(fnmatch.fnmatchcase(topic, p) for p in allowed)

    assert admits(wire.key("rpc", "lab", "arm-1", "act"))
    assert admits(wire.key("manifest", "lab", "arm-1"))
    for slot in sorted(set(wire.KEYS) - {"rpc", "manifest"}):
        pattern = wire.KEYS[slot]
        topic = pattern.format(tenant="lab", device_id="arm-1", name="x")
        assert not admits(topic), f"{slot}: {topic} must not be reachable by direct message"


def test_an_iam_link_connects_with_a_client_id_the_operator_policy_admits():
    connect = [s for s in _doc()["Statement"] if s["Action"] == "iot:Connect"][0]["Resource"]
    pattern = connect.split(":client/", 1)[1]
    assert fnmatch.fnmatchcase(default_client_id({}, "lab"), pattern)
    device = default_client_id({"cert": "cert.pem", "key": "key.pem"})
    assert device.startswith("strands-") and not fnmatch.fnmatchcase(device, pattern)


class _Refusing:
    """A transport whose device is present but whose manifest request IAM refuses."""

    def watch_liveliness(self, key_expr, cb):
        return lambda: None

    def query(self, key, payload, timeout):
        raise RemoteError(f"{key}: direct message refused (403)")


def test_a_refused_manifest_request_is_logged_rather_than_silent(caplog):
    caplog.set_level(logging.WARNING, logger="strands_robots.mesh.presence")
    d = Directory(_Refusing(), "lab")
    d._on_presence("strands/lab/@presence/arm-1", True)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not caplog.records:
        time.sleep(0.02)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "arm-1" in text and "refused" in text and "403" in text
