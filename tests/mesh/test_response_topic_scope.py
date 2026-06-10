"""Tests for response-topic responder scoping (pentest F-15 / B-09).

The response topic is strands/{operator}/response/{responder}/{turn} so
the IoT robot policy can pin a robot's response publishes to its OWN
ThingName -- closing the cross-robot response-spoof surface.
"""

from __future__ import annotations

import json
from typing import Any

from strands_robots.mesh.core import Mesh
from strands_robots.mesh.iot import provision


class _FakeRobot:
    def get_observation(self) -> dict[str, Any]:
        return {}

    def status(self) -> dict[str, Any]:
        return {"status": "idle"}


def _capture_puts(m: Mesh) -> list[tuple[str, dict]]:
    puts: list[tuple[str, dict]] = []
    m.publish = lambda key, payload, **kw: puts.append((key, payload))  # type: ignore
    return puts


def test_response_topic_includes_responder_id():
    """rkey must be strands/{operator}/response/{responder}/{turn}."""
    m = Mesh(_FakeRobot(), peer_id="robot-a")
    puts = _capture_puts(m)
    m._exec_cmd({"sender_id": "operator-1", "turn_id": "turn-xyz", "command": {"action": "status"}})
    keys = [k for k, _ in puts]
    assert "strands/operator-1/response/robot-a/turn-xyz" in keys, keys
    # the responder segment is the publisher's own peer_id
    assert all("/response/robot-a/" in k for k in keys if "/response/" in k)
    # F-15 invariant: the topic responder segment and the payload
    # responder_id field must convey the SAME identity. The broker ACL keys
    # off the topic; _on_response's expected_responders check keys off the
    # payload. If they ever drift, one layer passes while the other rejects
    # -- exactly the split this fix exists to prevent.
    responses = [d for k, d in puts if "/response/" in k]
    assert responses, puts
    assert all(d.get("responder_id") == "robot-a" for d in responses), responses


def test_robot_policy_response_scoped_to_own_thingname():
    """F-15: AllowResponseToAnyOperator must pin responder to ThingName."""
    policy = provision._ROBOT_POLICY_DOC
    stmt = next(s for s in policy["Statement"] if s.get("Sid") == "AllowResponseToAnyOperator")
    resources = stmt["Resource"]
    assert all("${iot:Connection.Thing.ThingName}" in r for r in resources), resources
    # Must NOT contain the old fleet-wide wildcard form.
    assert all(r != "arn:aws:iot:*:*:topic/strands/*/response/*" for r in resources)


def test_operator_receive_covers_new_depth():
    """Operator must still receive the deeper response topic via # filter."""
    policy = provision._OPERATOR_POLICY_DOC
    stmt = next(s for s in policy["Statement"] if s.get("Sid") == "OperatorReceiveResponses")
    joined = "\n".join(stmt["Resource"])
    # multi-level filter so {robot}/{turn} depth is covered
    assert "response/#" in joined


def test_policies_serialise_to_json():
    """Both policy docs must remain valid JSON."""
    for doc in (provision._ROBOT_POLICY_DOC, provision._OPERATOR_POLICY_DOC):
        json.dumps(doc)
