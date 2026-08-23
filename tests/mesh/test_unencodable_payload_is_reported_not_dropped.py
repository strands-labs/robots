"""A payload that can never reach the wire is reported, not dropped at DEBUG.

Every transport's ``put`` is fire-and-forget, and
:meth:`strands_robots.mesh.transport.base.MeshTransport.put`
scopes that tolerance to a TRANSIENT failure - a closed session, a dropped broker, a socket-level write -
which the caller's next tick retries. A payload the JSON encoder refuses is not
transient: it fails identically forever, so the message never goes out and no
retry can change that.

Both encode sites absorbed the two into one ``logger.debug`` line, which left the
two halves of ONE call disagreeing.
:meth:`~strands_robots.mesh.sensors.SensorLoopsMixin.publish_safety_event` writes
its event to the wire AND to the local audit log; on an unencodable payload the
audit half records a ``sig="SERIALISE_FAILED"`` poison record and logs at ERROR,
while the wire half dropped the event with nothing above DEBUG. A
default-configured operator saw a forensic trail asserting a safety event had
been raised, with no peer having received it.

The tests below are grouped so a reader can tell regression evidence from a
boundary:

* :class:`TestAnUnencodablePayloadIsReported` is the regression - the report did
  not exist, so every case fails on the pre-fix tree.
* :class:`TestTheReportIsBounded` also fails pre-fix, for the same reason (there
  was no report to bound). It is separate because what it pins is the SHAPE of
  the report rather than its existence: one line per broken payload builder, and
  a second topic on its own terms. That is what fails if the guard is dropped or
  keyed process-wide.
* :class:`TestTheTransientContractIsUnchanged` is the only group that passes on
  both trees, and it is the no-overreach evidence: a retryable wire failure keeps
  its DEBUG tolerance, an encodable payload is published silently and
  byte-identically on both legs, and a closed mesh is still a silent no-op. These
  fail if the escalation is applied to the publish attempt instead of the encode.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.mesh import session as mesh_session
from strands_robots.mesh.transport.iot_transport import IotMqttTransport

#: A payload shape a robot safety event naturally carries. ``np.float32`` is what
#: a sensor reading is, and ``np.zeros(3)`` is what a pose or joint vector is;
#: neither is JSON-serialisable, and both reach the encoder through the public
#: ``publish_safety_event`` / ``publish`` surfaces.
UNENCODABLE: list[Any] = [
    pytest.param({"distance_m": np.float32(0.02)}, id="numpy-scalar-sensor-reading"),
    pytest.param({"pose": np.zeros(3)}, id="numpy-array-pose-vector"),
    pytest.param({"raw": b"\x01\x02"}, id="bytes"),
    pytest.param({"seen": {1, 2}}, id="set"),
]

#: The topic a safety event rides, used so the assertions read as the operator's
#: case rather than an abstract key.
SAFETY_TOPIC = "strands/arm-a/safety/event"


class _StubZenohSession:
    """Stands in for an open ``zenoh.Session``, recording what reached the wire."""

    def __init__(self) -> None:
        self.puts: list[tuple[str, bytes]] = []

    def put(self, key: str, payload: bytes) -> None:
        self.puts.append((key, payload))


class _StubMqttClient:
    """Stands in for the awscrt MQTT5 client, recording published packets."""

    def __init__(self) -> None:
        self.published: list[Any] = []

    def publish(self, packet: Any) -> None:
        self.published.append(packet)


@pytest.fixture(autouse=True)
def _fresh_once_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test an empty once-per-topic guard.

    The guard is module state that deliberately outlives a single publish, so a
    sibling test's report would otherwise silence this one's.
    """
    monkeypatch.setattr(mesh_session, "_unencodable_topics_warned", set(), raising=False)


@pytest.fixture
def zenoh_wire(monkeypatch: pytest.MonkeyPatch) -> _StubZenohSession:
    """An open session on the legacy Zenoh path - the default backend's encode site."""
    stub = _StubZenohSession()
    monkeypatch.setattr(mesh_session, "_SESSION", stub)
    return stub


@pytest.fixture
def mqtt_leg() -> IotMqttTransport:
    """A connected MQTT transport.

    Constructed without ``__init__`` because that opens a real mTLS connection.
    ``put`` reads only the client and the connected flag, and the encode now
    happens before the ``awscrt`` import, so this needs no ``[mesh-iot]`` extra.
    """
    transport = IotMqttTransport.__new__(IotMqttTransport)
    transport._client = _StubMqttClient()
    transport._connected = threading.Event()
    transport._connected.set()
    return transport


def _publish_on_each_leg(zenoh_wire: _StubZenohSession, mqtt_leg: IotMqttTransport, payload: dict[str, Any]) -> None:
    """Publish *payload* through both encode sites the mesh ships."""
    mesh_session._put_zenoh_directly(SAFETY_TOPIC, payload)
    mqtt_leg.put(SAFETY_TOPIC, payload)


class TestAnUnencodablePayloadIsReported:
    """The regression: a permanently-undeliverable message reaches the operator."""

    @pytest.mark.parametrize("payload", UNENCODABLE)
    def test_the_zenoh_leg_reports_at_a_level_a_default_operator_sees(
        self, payload: dict[str, Any], zenoh_wire: _StubZenohSession, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Nothing reaches the wire, and the drop is stated at WARNING or above.

        ``caplog.set_level(WARNING)`` is the operator's default: a report below it
        is invisible to a consumer that has configured nothing.
        """
        with caplog.at_level(logging.WARNING):
            mesh_session._put_zenoh_directly(SAFETY_TOPIC, payload)

        assert not zenoh_wire.puts, "premise: an unencodable payload cannot reach the wire"
        assert caplog.records, (
            f"the payload {payload!r} can never be published, and the drop was not "
            "reported at any level a default-configured operator sees"
        )

    @pytest.mark.parametrize("payload", UNENCODABLE)
    def test_the_mqtt_leg_reports_the_same_condition(
        self, payload: dict[str, Any], mqtt_leg: IotMqttTransport, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The second encode site answers the same way, so the rule is not per-leg."""
        with caplog.at_level(logging.WARNING):
            mqtt_leg.put(SAFETY_TOPIC, payload)

        client = mqtt_leg._client
        assert isinstance(client, _StubMqttClient), "premise: the stub client is installed"
        assert not client.published, "premise: nothing was published"
        assert caplog.records, f"the MQTT leg dropped {payload!r} without reporting it above DEBUG"

    def test_the_report_names_the_topic_and_the_reason(
        self, zenoh_wire: _StubZenohSession, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A report an operator can act on names WHICH topic and WHY.

        Without the topic the operator cannot find the payload builder at fault,
        and without the encoder's own reason they cannot tell an unencodable
        value from a wire problem.
        """
        with caplog.at_level(logging.WARNING):
            mesh_session._put_zenoh_directly(SAFETY_TOPIC, {"distance_m": np.float32(0.02)})

        text = "\n".join(record.getMessage() for record in caplog.records)
        assert SAFETY_TOPIC in text, f"the report does not name the topic: {text!r}"
        assert "float32" in text, f"the report does not carry the encoder's reason: {text!r}"

    def test_the_wire_and_the_audit_halves_of_one_call_agree(
        self, zenoh_wire: _StubZenohSession, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The two sinks of ``publish_safety_event`` report at the same level.

        The audit half already records a ``SERIALISE_FAILED`` poison record and
        logs at ERROR. A wire half that reported lower is what let a forensic
        trail assert an event no peer received.
        """
        from strands_robots.mesh import audit

        payload = {"distance_m": np.float32(0.02)}
        with caplog.at_level(logging.WARNING):
            mesh_session._put_zenoh_directly(SAFETY_TOPIC, payload)
        wire_levels = {record.levelno for record in caplog.records}

        assert wire_levels, "the wire half reported nothing"
        assert max(wire_levels) >= logging.ERROR, (
            "the audit half of the same call reports an unencodable payload at ERROR "
            f"(see {audit.__name__}'s SERIALISE_FAILED poison record); the wire half "
            f"reported only {sorted(logging.getLevelName(lv) for lv in wire_levels)}, so "
            "the forensic trail and the fleet disagree about whether the event went out"
        )


class TestTheTransientContractIsUnchanged:
    """Boundary: a retryable failure keeps its DEBUG-and-continue tolerance."""

    def test_a_wire_failure_stays_below_the_operator_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A session that refuses the write is transient, so it is not escalated.

        This is what fails if the fix is applied to the publish attempt rather
        than to the encode.
        """

        class _RefusingSession:
            def put(self, key: str, payload: bytes) -> None:
                raise ConnectionError("broker went away")

        monkeypatch.setattr(mesh_session, "_SESSION", _RefusingSession())
        with caplog.at_level(logging.WARNING):
            mesh_session._put_zenoh_directly(SAFETY_TOPIC, {"distance_m": 0.02})

        assert not caplog.records, (
            "a transient wire failure must keep its DEBUG tolerance - the next tick "
            f"retries it: {[r.getMessage() for r in caplog.records]}"
        )

    def test_an_encodable_payload_reaches_the_wire_untouched(
        self, zenoh_wire: _StubZenohSession, mqtt_leg: IotMqttTransport, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The happy path is byte-identical and silent on both legs."""
        payload = {"distance_m": 0.02, "peer_id": "arm-a"}
        with caplog.at_level(logging.WARNING):
            _publish_on_each_leg(zenoh_wire, mqtt_leg, payload)

        assert [key for key, _ in zenoh_wire.puts] == [SAFETY_TOPIC]
        assert json.loads(zenoh_wire.puts[0][1]) == payload
        assert not caplog.records, "an encodable payload must be published silently"

    def test_no_session_is_still_a_silent_no_op(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A closed mesh publishes nothing and says nothing, as documented."""
        monkeypatch.setattr(mesh_session, "_SESSION", None)
        with caplog.at_level(logging.WARNING):
            mesh_session._put_zenoh_directly(SAFETY_TOPIC, {"distance_m": np.float32(0.02)})

        assert not caplog.records, "a closed mesh is a no-op, not an unencodable payload"


class TestTheReportIsBounded:
    """Boundary: the report cannot flood a control-rate publisher."""

    def test_one_report_per_topic_however_many_ticks(
        self, zenoh_wire: _StubZenohSession, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 50 Hz publisher with a broken builder reports once, not per tick.

        A permanent failure repeats on every tick by definition, so an unbounded
        report would make the log unusable - the outcome the report exists to
        avoid.
        """
        payload = {"distance_m": np.float32(0.02)}
        with caplog.at_level(logging.WARNING):
            for _ in range(250):
                mesh_session._put_zenoh_directly(SAFETY_TOPIC, payload)

        assert len(caplog.records) == 1, f"250 ticks on one topic produced {len(caplog.records)} reports"

    def test_a_second_topic_is_reported_on_its_own_terms(
        self, zenoh_wire: _StubZenohSession, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The bound is per topic: a different builder is a different fact.

        This is what fails if the guard is keyed process-wide, which would hide
        every topic after the first.
        """
        payload = {"distance_m": np.float32(0.02)}
        with caplog.at_level(logging.WARNING):
            mesh_session._put_zenoh_directly(SAFETY_TOPIC, payload)
            mesh_session._put_zenoh_directly("strands/arm-a/input/frame", payload)

        reported = {record.getMessage() for record in caplog.records}
        assert len(reported) == 2, f"a second topic was not reported: {reported}"

    def test_both_legs_of_a_bridge_report_one_topic_once(
        self, zenoh_wire: _StubZenohSession, mqtt_leg: IotMqttTransport, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Under the bridge backend one broken builder is one fact, not two.

        Both legs encode the same payload with the same encoder, so a per-leg
        report would restate it.
        """
        with caplog.at_level(logging.WARNING):
            _publish_on_each_leg(zenoh_wire, mqtt_leg, {"distance_m": np.float32(0.02)})

        assert len(caplog.records) == 1, f"two legs on one topic produced {len(caplog.records)} reports"
