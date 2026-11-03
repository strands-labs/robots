"""Lifecycle events are an audit trail (MAPPING 4.3, EXPERIMENTS §5): parsed, filtered, remembered; not liveness."""

from __future__ import annotations

import json
import time
import uuid

from strands_robots.mesh.iot import lifecycle
from strands_robots.mesh.transports.iot_link import FakeIotBroker, FakeLink


def test_parse_reads_connected_and_disconnected_and_ignores_the_rest():
    ev = lifecycle.parse(
        "$aws/events/presence/connected/arm-1", json.dumps({"clientId": "arm-1", "timestamp": 1700000000123}).encode()
    )
    assert ev == {"client": "arm-1", "connected": True, "t": 1700000000.123, "reason": None, "client_initiated": None}
    raw = json.dumps(
        {
            "clientId": "arm-1",
            "timestamp": 1700000001000,
            "disconnectReason": "CONNECTION_LOST",
            "clientInitiatedDisconnect": False,
        }
    )
    ev = lifecycle.parse("$aws/events/presence/disconnected/arm-1", raw.encode())
    assert ev["connected"] is False and ev["reason"] == "CONNECTION_LOST" and ev["client_initiated"] is False
    assert lifecycle.parse("strands/lab/arm-1/state", b"{}") is None
    assert lifecycle.parse("$aws/events/presence/connected/arm-1", b"not json") is None


def test_watch_filters_to_the_tenants_clients_and_audit_remembers_the_last_event():
    broker = FakeIotBroker.named(f"lc-{uuid.uuid4().hex[:8]}")
    link, cloud = FakeLink(broker, "op"), FakeLink(broker, "aws")
    audit = lifecycle.Audit(keep=3)
    undo = lifecycle.watch(link, audit, tenant_clients=lambda c: c.startswith("lab-"))
    for client, kind in (
        ("lab-arm", "connected"),
        ("other-arm", "connected"),
        ("lab-arm", "disconnected"),
        ("lab-b", "connected"),
        ("lab-c", "connected"),
    ):
        cloud.publish(
            f"$aws/events/presence/{kind}/{client}",
            json.dumps(
                {"clientId": client, "timestamp": int(time.time() * 1000), "disconnectReason": "CONNECTION_LOST"}
            ).encode(),
        )
    end = time.monotonic() + 2
    while len(audit.events) < 3 and time.monotonic() < end:
        time.sleep(0.01)
    assert [e["client"] for e in audit.events] == [
        "lab-arm",
        "lab-b",
        "lab-c",
    ]  # other-arm filtered, first lab-arm aged out (keep=3)
    assert audit.since("lab-arm")["connected"] is False and audit.since("lab-arm")["reason"] == "CONNECTION_LOST"
    assert audit.since("nobody") is None
    undo()
    link.close()
    cloud.close()


def test_the_lifecycle_rule_only_audits_and_names_the_tenant():
    r = lifecycle.clear_presence_rule("lab", "arn:aws:iam::123456789012:role/x")
    assert list(r) == ["strands_lifecycle_lab"]
    assert r["strands_lifecycle_lab"]["sql"].startswith(
        "SELECT clientId, disconnectReason, timestamp FROM '$aws/events/presence/disconnected/+'"
    )
    assert r["strands_lifecycle_lab"]["actions"][0]["republish"]["topic"] == "strands/lab/@audit/disconnected"
