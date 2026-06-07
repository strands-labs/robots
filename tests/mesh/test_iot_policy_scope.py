"""Tests pinning the scope of the canonical IoT policies.

The robot and operator policies in :mod:`provision` deliberately avoid the
``strands/*`` wildcard for ``iot:Receive`` so neither role can eavesdrop
on the entire fleet's mesh traffic. These tests assert that scope:

* Robot ``Receive`` covers only the robot's own ``/cmd``, own
  ``/response/*``, ``broadcast``, ``safety/estop``, and ``+/presence``.
* Operator ``Receive`` covers monitoring topics (``presence``, ``state``,
  ``health``, ``safety/event``, ``safety/estop``) and not the
  command/response streams of other operators.

A future refactor that re-introduces the wildcard will fail these tests
loudly, surfacing the regression in code review.
"""

from __future__ import annotations

from strands_robots.mesh.iot.provision import (
    _OPERATOR_POLICY_DOC,
    _ROBOT_POLICY_DOC,
)


def _statements_by_sid(doc: dict) -> dict[str, dict]:
    return {st.get("Sid", ""): st for st in doc["Statement"]}


class TestRobotPolicy:
    def test_no_unconditional_receive_wildcard(self):
        """Robot policy must NOT contain iot:Receive on strands/*."""
        for st in _ROBOT_POLICY_DOC["Statement"]:
            actions = st.get("Action")
            if isinstance(actions, str):
                actions = [actions]
            if not any(a == "iot:Receive" or a == "iot:*" for a in actions):
                continue
            resources = st.get("Resource")
            if isinstance(resources, str):
                resources = [resources]
            for r in resources:
                # Wildcard on strands/* would expose the entire fleet;
                # specific-suffix patterns are OK.
                assert not r.endswith(":topic/strands/*"), f"Found wildcard Receive resource: {r!r}"

    def test_scoped_receive_present(self):
        """The replacement statement must exist and cover only the topics
        robots actually subscribe to."""
        sids = _statements_by_sid(_ROBOT_POLICY_DOC)
        assert "AllowReceiveScoped" in sids, "scoped-Receive statement missing"
        st = sids["AllowReceiveScoped"]
        resources = st["Resource"]
        # Must include own cmd + own response + broadcast + safety + presence.
        joined = "\n".join(resources)
        assert "${iot:Connection.Thing.ThingName}/cmd" in joined
        assert "${iot:Connection.Thing.ThingName}/response/*" in joined
        assert "/strands/broadcast" in joined
        assert "/strands/safety/estop" in joined
        assert "/strands/+/presence" in joined

    def test_own_health_subscribable_but_not_receivable(self):
        """Issue #253: Subscribe permits own ${ThingName}/* (incl. health)
        while Receive deliberately omits it, so the broker drops inbound
        copies. Pins the asymmetry against a future Receive-widening."""
        sids = _statements_by_sid(_ROBOT_POLICY_DOC)
        sub = sids["AllowOwnSubscriptions"]
        recv = sids["AllowReceiveScoped"]
        sub_actions = sub["Action"]
        if isinstance(sub_actions, str):
            sub_actions = [sub_actions]
        assert "iot:Subscribe" in sub_actions
        # Subscribe covers the broad own-thing wildcard.
        assert any(r.endswith("topicfilter/strands/${iot:Connection.Thing.ThingName}/*") for r in sub["Resource"]), (
            "Subscribe should cover own ${ThingName}/* wildcard"
        )
        # Receive must NOT grant the broad own-thing wildcard nor health/state.
        recv_joined = "\n".join(recv["Resource"])
        assert "${iot:Connection.Thing.ThingName}/*" not in recv_joined, (
            "Receive must not widen to own ${ThingName}/* (issue #253)"
        )
        assert "${iot:Connection.Thing.ThingName}/health" not in recv_joined
        assert "${iot:Connection.Thing.ThingName}/state" not in recv_joined
        assert "${iot:Connection.Thing.ThingName}/safety/event" not in recv_joined

    def test_publish_still_scoped_to_own_thing(self):
        """Sanity: publish remains scoped to the robot's own topics."""
        sids = _statements_by_sid(_ROBOT_POLICY_DOC)
        st = sids["AllowOwnTopics"]
        for r in st["Resource"]:
            assert "${iot:Connection.Thing.ThingName}/" in r

    def test_no_receive_on_arbitrary_camera(self):
        """A robot must not be able to subscribe to another robot's camera."""
        for st in _ROBOT_POLICY_DOC["Statement"]:
            actions = st.get("Action")
            if isinstance(actions, str):
                actions = [actions]
            if "iot:Subscribe" not in actions and "iot:Receive" not in actions:
                continue
            resources = st.get("Resource")
            if isinstance(resources, str):
                resources = [resources]
            for r in resources:
                assert "/camera/" not in r, f"Camera subscription leaked into robot policy: {r!r}"


class TestOperatorPolicy:
    def test_no_unconditional_receive_wildcard(self):
        """Operator policy must not allow Receive on strands/* either."""
        sids = _statements_by_sid(_OPERATOR_POLICY_DOC)
        st = sids["OperatorObserveFleet"]
        for r in st["Resource"]:
            assert not r.endswith(":topic/strands/*"), f"Operator wildcard Receive: {r!r}"

    def test_scoped_to_monitoring_topics(self):
        sids = _statements_by_sid(_OPERATOR_POLICY_DOC)
        st = sids["OperatorObserveFleet"]
        joined = "\n".join(st["Resource"])
        assert "/strands/+/presence" in joined
        assert "/strands/+/state" in joined
        assert "/strands/+/health" in joined
        assert "/strands/+/safety/event" in joined
        assert "/strands/safety/estop" in joined

    def test_no_camera_or_input_in_operator_observe(self):
        sids = _statements_by_sid(_OPERATOR_POLICY_DOC)
        st = sids["OperatorObserveFleet"]
        for r in st["Resource"]:
            assert "/camera/" not in r
            assert "/input/" not in r

    def test_publish_to_fleet_wildcard_is_deliberate(self):
        """Pin: OperatorPublishToFleet uses ``strands/*/cmd`` wildcard by design.

        The system has no per-operator-to-per-robot binding. A compromised
        operator credential has equivalent scope to a compromised fleet
        command authority. Mitigations are short-lived certs, the
        OperatorShadow attribute condition, and the operational audit log.
        A per-robot operator scope would require one policy document per
        robot, scaling policy count linearly with fleet size.

        If this test breaks, someone narrowed the operator publish scope --
        verify the corresponding transport/dispatch code still routes
        commands correctly.
        """
        sids = _statements_by_sid(_OPERATOR_POLICY_DOC)
        st = sids["OperatorPublishToFleet"]
        resources = st["Resource"]
        # The wildcard ``strands/*/cmd`` must exist for the operator to
        # address any robot without a per-robot policy.
        assert any(r.endswith(":topic/strands/*/cmd") for r in resources), (
            "OperatorPublishToFleet must retain the */cmd wildcard (deliberate design choice)"
        )
