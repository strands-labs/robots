"""A robot-less peer must not be counted as having failed to stop.

``robot_mesh._gateway_mesh`` builds a ``Mesh`` with ``robot=None`` so a
coordinator process (dashboard, scheduler, logger) can reach the fleet. That
gateway is a peer like any other: ``Mesh.start`` subscribes it to
``strands/broadcast``, so the ``{"action": "stop"}`` fanout from
:meth:`Mesh.emergency_stop` is delivered to its ``_dispatch``.

With ``self.robot is None`` every stop branch is skipped -- ``stop_task``,
``stop_policy`` and the ``_sim_parent`` delegation all test the robot -- so the
answer used to be the terminal ``{"ok": False, "error": "peer exposes no
stop_task; nothing was stopped"}``. :func:`_peers_that_did_not_stop` reads
``ok is False`` as an affirmative report of failure, so the operator's own
gateway landed in ``peers_not_stopped`` and ``emergency_stop`` logged CRITICAL
"robots may still be executing; use a hardware cutoff" naming it -- on every
e-stop, whenever the gateway is used for its stated purpose.

That is the failure mode ``_peers_that_did_not_stop``'s docstring calls out as
the reason it is deliberately conservative: "a false 'did not stop' on the
safety path trains operators to ignore the warning". A peer with no robot has
nothing to halt, so "did not stop" is not conservative about it, it is wrong.

Pinned here: the robot-less answer is affirmative and empty, it is absent from
the aggregation, and -- the half that keeps the fix from being over-broad -- a
peer that *has* a robot with no stop surface is still reported as not stopped.
"""

from __future__ import annotations

from typing import Any

from strands_robots.mesh import Mesh
from strands_robots.mesh.core import _peers_that_did_not_stop


class _RobotWithoutStopSurface:
    """A registered robot exposing no stop verb at all.

    Deliberately not a ``Mock``: ``hasattr`` is the router's test, and a mock
    answers every ``hasattr`` truthfully-by-fabrication, which would make this
    fixture assert the opposite of what it is here to assert.
    """

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "idle"}


def _stop_response(peer_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Wrap a ``_dispatch`` result in the envelope ``broadcast`` collects."""
    return {"responder_id": peer_id, "result": result}


class TestARobotLessPeerAnswersTheStopAffirmatively:
    """``_dispatch`` distinguishes "nothing to stop" from "did not stop"."""

    def test_the_gateway_reports_ok_with_an_empty_stopped_list(self) -> None:
        gateway = Mesh(None, peer_id="gateway-dash-1", peer_type="gateway")

        out = gateway._dispatch({"action": "stop"})

        assert out["ok"] is True
        # Empty rather than absent: the caller is told what was halted, and
        # nothing was, which is why ok=True is not a lie here.
        assert out["stopped"] == []

    def test_the_answer_says_why_nothing_was_stopped(self) -> None:
        gateway = Mesh(None, peer_id="gateway-dash-1", peer_type="gateway")

        out = gateway._dispatch({"action": "stop"})

        # An operator reading the raw broadcast responses needs to tell this
        # apart from a robot that stopped nothing because nothing was running.
        assert "no robot registered" in out["note"]


class TestTheGatewayIsAbsentFromTheSafetyAggregation:
    """The e-stop accounting no longer names the coordinator's own gateway."""

    def test_a_gateway_response_is_not_counted_as_not_stopped(self) -> None:
        gateway = Mesh(None, peer_id="gateway-dash-1", peer_type="gateway")
        responses = [_stop_response(gateway.peer_id, gateway._dispatch({"action": "stop"}))]

        assert _peers_that_did_not_stop(responses) == set()

    def test_a_gateway_alongside_a_real_peer_leaves_only_the_real_failure(self) -> None:
        gateway = Mesh(None, peer_id="gateway-dash-1", peer_type="gateway")
        stopless = Mesh(_RobotWithoutStopSurface(), peer_id="arm-7")
        responses = [
            _stop_response(gateway.peer_id, gateway._dispatch({"action": "stop"})),
            _stop_response(stopless.peer_id, stopless._dispatch({"action": "stop"})),
        ]

        # The gateway drops out; the arm -- which really did not stop -- stays.
        assert _peers_that_did_not_stop(responses) == {"arm-7"}


class TestAPeerWithARobotIsStillHeldToTheStopContract:
    """The fix is scoped to ``robot is None``, not to "no stop surface"."""

    def test_a_registered_robot_without_a_stop_verb_still_reports_failure(self) -> None:
        stopless = Mesh(_RobotWithoutStopSurface(), peer_id="arm-7")

        out = stopless._dispatch({"action": "stop"})

        # This peer has a robot that may well be executing, so the affirmative
        # failure report -- and the [safety] ERROR log beside it -- must stand.
        assert out["ok"] is False
        assert "nothing was stopped" in out["error"]

    def test_that_peer_is_still_named_by_the_aggregation(self) -> None:
        stopless = Mesh(_RobotWithoutStopSurface(), peer_id="arm-7")
        responses = [_stop_response("arm-7", stopless._dispatch({"action": "stop"}))]

        assert _peers_that_did_not_stop(responses) == {"arm-7"}
