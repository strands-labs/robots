"""``signed_estop`` reports what the peers said, not only that the rail latched.

``Mesh.emergency_stop`` sets the issuer's lockout unconditionally, before it
broadcasts anything. ``MeshBridge.signed_estop`` returned that fact as
``lockout_engaged: True`` and nothing else, so one value described a stop that
reached every peer and a stop that reached nobody identically -- and the sheet
rendered it as "peers refuse all commands until resumed", which is a claim about
the room made from a fact about this process.

``lockout_engaged`` stays ``True``, because it is true and because the operator's
resume control is gated on it: making it ``False`` on an unacknowledged stop
would hide the only way to clear a lockout that really is engaged. What was
missing is the peer half, which ``Mesh.emergency_stop`` already computes for its
own ``strands/safety/estop`` envelope and audit record:

* ``responses_received`` -- REPLIES received, not stops confirmed, keeping the
  meaning #1680 gave it so the two numbers can be compared;
* ``peers_not_stopped`` -- responders that AFFIRMATIVELY reported they did not
  stop, graded by ``mesh.core._peers_that_did_not_stop`` rather than by a second
  copy of that rule here.

These pin that the three fleets read differently and that the grading is not
re-implemented. The route that forwards both fields to the operator (and the
sheet sentence derived from them) lives in ``dashboard.server`` and the
frontend, which land with the server slice of the #2848 decomposition (#2977);
the route-level pin travels with that slice.

Run with --no-cov.
"""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest

from strands_robots.dashboard.mesh_bridge import MeshBridge

# The response shapes _reports_failure_to_stop grades, per its docstring: the
# stop verbs disagree about their envelope, so both spellings must be graded.
ACK = {"responder_id": "arm-1", "result": {"ok": True, "status": "stopped"}}
ACK_2 = {"responder_id": "arm-4", "result": {"ok": True, "status": "stopped"}}
NO_STOP_TASK = {"responder_id": "arm-2", "result": {"ok": False, "error": "peer exposes no stop_task"}}
SIM_REFUSED = {"responder_id": "sim-1", "result": {"status": "error", "error": "stop_policy refused"}}
SIM_PARTIAL = {"responder_id": "sim-2", "result": {"ok": False, "not_stopped": ["r1"]}}
ANONYMOUS_FAILURE = {"result": {"ok": False}}  # no responder_id
UNRECOGNISED = {"responder_id": "arm-3", "result": {"pending": True}}


class FakeMesh:
    """Stands in for the safety Mesh: emergency_stop returns scripted replies."""

    peer_id = "dash-safety"
    alive = True

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses

    def emergency_stop(self) -> list[dict[str, Any]]:
        return list(self._responses)


def signed_estop_over(responses: list[dict[str, Any]]) -> dict[str, Any]:
    bridge = MeshBridge(peer_id="dash")
    with mock.patch.object(MeshBridge, "_safety_mesh", return_value=FakeMesh(responses)):
        return bridge.signed_estop()


def operator_visible(payload: dict[str, Any]) -> str:
    """What the route forwards: everything except the raw responses."""
    return json.dumps({k: v for k, v in payload.items() if k != "responses"}, sort_keys=True)


# --------------------------------------------------------------- the defect
def test_three_different_fleets_no_longer_read_identically():
    """The defect in one assertion: these three produced ONE payload."""
    everybody = signed_estop_over([ACK, ACK_2])
    nobody = signed_estop_over([])
    refused = signed_estop_over([NO_STOP_TASK, SIM_REFUSED])

    rendered = {operator_visible(p) for p in (everybody, nobody, refused)}
    assert len(rendered) == 3, f"these fleets are indistinguishable to an operator: {rendered}"


def test_a_stop_nobody_answered_reports_no_acknowledgement():
    out = signed_estop_over([])

    assert out["responses_received"] == 0
    assert out["peers_not_stopped"] == []
    # Still latched: a resume IS required, and the resume box is gated on this.
    assert out["lockout_engaged"] is True


def test_peers_that_reported_not_stopping_are_named_not_counted():
    out = signed_estop_over([ACK, NO_STOP_TASK, SIM_REFUSED])

    assert out["peers_not_stopped"] == ["arm-2", "sim-1"], "name them; a count is not actionable"
    # Replies received, not stops confirmed - the two numbers stay comparable.
    assert out["responses_received"] == 3


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (NO_STOP_TASK, ["arm-2"]),
        (SIM_REFUSED, ["sim-1"]),
        (SIM_PARTIAL, ["sim-2"]),
        (UNRECOGNISED, []),  # conservative: an unknown shape is not a failure report
        (ACK, []),
    ],
)
def test_each_graded_shape_reaches_the_operator(response, expected):
    assert signed_estop_over([response])["peers_not_stopped"] == expected


def test_an_unidentified_failure_still_counts():
    """A response with no responder_id must not vanish from the accounting."""
    out = signed_estop_over([ANONYMOUS_FAILURE])

    assert out["responses_received"] == 1
    assert len(out["peers_not_stopped"]) == 1
    assert "unidentified" in out["peers_not_stopped"][0]


def test_the_grading_has_one_owner_and_is_not_reimplemented():
    """The bridge's verdict must equal mesh.core's for the same replies.

    A second copy of this rule on the safety path is how the sim dispatch branch
    once reported ok=True over a refusal it already had in hand.
    """
    from strands_robots.mesh.core import _peers_that_did_not_stop

    replies = [ACK, NO_STOP_TASK, SIM_REFUSED, SIM_PARTIAL, UNRECOGNISED, ANONYMOUS_FAILURE]
    assert signed_estop_over(replies)["peers_not_stopped"] == sorted(_peers_that_did_not_stop(replies))


def test_the_latch_is_reported_on_every_fleet_because_resume_needs_it():
    for responses in ([], [ACK], [NO_STOP_TASK], [ACK, SIM_REFUSED]):
        assert signed_estop_over(responses)["lockout_engaged"] is True
