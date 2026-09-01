# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The command envelope's routing fields must be identifiers, not key expressions.

``Mesh._exec_cmd`` answers a command on
``strands/{sender_id}/response/{responder}/{turn_id}``, where ``sender_id`` and
``turn_id`` are read straight off the wire. Zenoh accepts a wildcard on a ``put``
and routes it by intersection, so a ``sender_id`` of ``**`` produces a key that
every peer's ``strands/{peer}/response/**`` subscription matches: the response,
which carries this robot's dispatch result, goes to the whole fleet instead of to
the peer that asked.

``validate_command`` guards a ``turn_id`` / ``sender_id`` found inside the
*command* dict (see ``test_security_control_char_gates.py``), but routing reads
the envelope's own copy. These cells pin the envelope pair, and the companion
file ``test_response_topic_scope.py`` pins the shape of the key they build.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from strands_robots.mesh.core import Mesh
from strands_robots.mesh.security import MAX_PEER_ID_LEN

#: Envelope values that must never reach a key expression. Wildcards widen the
#: routing; ``/`` adds segments; whitespace, control bytes and a non-string are
#: refused by the same identifier rule the teleop seams already use.
UNROUTABLE = [
    pytest.param("**", id="zenoh-recursive-wildcard"),
    pytest.param("*", id="zenoh-single-wildcard"),
    pytest.param("op/extra", id="extra-key-segment"),
    pytest.param("op id", id="whitespace"),
    pytest.param("op\nid", id="line-break"),
    pytest.param("x" * (MAX_PEER_ID_LEN + 1), id="over-length"),
    pytest.param(7, id="non-string"),
]


class _Robot:
    """Records whether dispatch reached the robot at all.

    ``get_task_status`` is the first branch ``Mesh._dispatch`` takes for the
    ``status`` action, so the counter answers "did the command execute", and its
    return value is the operator-visible state a leaked response would carry.
    """

    def __init__(self) -> None:
        self.status_calls = 0

    def get_observation(self) -> dict[str, Any]:
        return {}

    def get_task_status(self) -> dict[str, Any]:
        self.status_calls += 1
        return {"status": "idle", "battery": 41}


def _mesh() -> tuple[Mesh, _Robot, list[tuple[str, dict]]]:
    robot = _Robot()
    mesh = Mesh(robot, peer_id="robot-a")
    puts: list[tuple[str, dict]] = []
    mesh.publish = lambda key, payload, **kw: puts.append((key, payload))  # type: ignore[method-assign]
    return mesh, robot, puts


@pytest.mark.parametrize("sender", UNROUTABLE)
def test_unroutable_sender_is_refused_before_it_can_address_a_response(sender: object) -> None:
    mesh, robot, puts = _mesh()
    mesh._exec_cmd({"sender_id": sender, "turn_id": "t1", "command": {"action": "status"}})
    assert puts == [], puts
    assert robot.status_calls == 0


@pytest.mark.parametrize("turn", UNROUTABLE)
def test_unroutable_turn_id_is_refused_before_it_can_address_a_response(turn: object) -> None:
    mesh, robot, puts = _mesh()
    mesh._exec_cmd({"sender_id": "operator-1", "turn_id": turn, "command": {"action": "status"}})
    assert puts == [], puts
    assert robot.status_calls == 0


def test_a_valid_envelope_still_answers_the_one_peer_that_asked() -> None:
    mesh, robot, puts = _mesh()
    mesh._exec_cmd({"sender_id": "operator-1", "turn_id": "turn-xyz", "command": {"action": "status"}})
    assert [k for k, _ in puts] == ["strands/operator-1/response/robot-a/turn-xyz"], puts
    assert robot.status_calls == 1


def test_an_absent_sender_still_dispatches_without_a_response() -> None:
    """The documented fire-and-forget shape: no route asked for, none built."""
    mesh, robot, puts = _mesh()
    mesh._exec_cmd({"sender_id": "", "command": {"action": "status"}})
    assert puts == [], puts
    assert robot.status_calls == 1


def test_an_absent_turn_id_still_routes_on_the_generated_one() -> None:
    mesh, robot, puts = _mesh()
    mesh._exec_cmd({"sender_id": "operator-1", "command": {"action": "status"}})
    assert len(puts) == 1, puts
    key = puts[0][0]
    assert key.startswith("strands/operator-1/response/robot-a/")
    assert robot.status_calls == 1


def test_the_refusal_names_the_field_without_putting_wire_bytes_on_the_log_line(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal is auditable, and cannot itself forge a second log record."""
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        "strands_robots.mesh.core.log_safety_event",
        lambda event, peer, payload: events.append((event, peer, payload)),
    )
    mesh, _robot, puts = _mesh()
    forging = "victim\n2026-01-01 00:00:00 CRITICAL mesh: estop cleared by operator"
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
        mesh._exec_cmd({"sender_id": forging, "turn_id": "t1", "command": {"action": "status"}})

    assert puts == [], puts
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, caplog.records
    assert all("\n" not in m and "\r" not in m for m in warnings), warnings
    assert any("sender_id" in m for m in warnings), warnings

    assert [e[0] for e in events] == ["command_rejected"], events
    payload = events[0][2]
    assert payload["field"] == "sender_id"
    assert payload["sender"] is None  # the offending value is not echoed as an identity
    assert len(payload["value"]) <= MAX_PEER_ID_LEN
    assert "\n" not in payload["value"]  # repr, so a structured reader sees the shape safely
