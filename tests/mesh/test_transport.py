"""Unit tests for the MeshTransport abstraction layer.

These tests do NOT touch real AWS or Zenoh — they verify protocol shape,
wildcard translation, QoS lookup, topic-filter matching, and ZenohTransport
delegation. The real AWS-backed integration test lives in
``tests_integ/test_iot_transport.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands_robots.mesh.transport import (
    IotMqttTransport,
    MeshTransport,
    Sample,
    ZenohTransport,
)
from strands_robots.mesh.transport.iot_transport import (
    _mqtt_topic_matches,
    _MqttSample,
    _qos_and_retain_for,
    _should_drop,
    _zenoh_to_mqtt_filter,
)

# Protocol shape


class TestProtocolShape:
    """Both transports satisfy the runtime-checkable Protocol."""

    def test_zenoh_satisfies_protocol(self):
        """ZenohTransport should satisfy the MeshTransport protocol."""
        t = ZenohTransport()
        assert isinstance(t, MeshTransport)

    def test_iot_satisfies_protocol(self):
        """IotMqttTransport should satisfy the MeshTransport protocol."""
        t = IotMqttTransport(thing_name="test", endpoint="x.iot.us-west-2.amazonaws.com")
        assert isinstance(t, MeshTransport)

    def test_mqtt_sample_satisfies_sample_protocol(self):
        """_MqttSample exposes .key_expr and .payload.to_bytes()."""
        s = _MqttSample("strands/foo/state", b'{"k":1}')
        assert isinstance(s, Sample)
        assert s.key_expr == "strands/foo/state"
        assert s.payload.to_bytes() == b'{"k":1}'


# Wildcard translation (Zenoh -> MQTT)


class TestZenohToMqttFilter:
    """Zenoh key-expression syntax → MQTT topic-filter syntax."""

    @pytest.mark.parametrize(
        "zenoh,mqtt",
        [
            # Concrete patterns Mesh actually uses
            ("strands/*/presence", "strands/+/presence"),
            ("strands/{peer}/cmd", "strands/{peer}/cmd"),
            ("strands/{peer}/response/**", "strands/{peer}/response/#"),
            ("strands/broadcast", "strands/broadcast"),
            # Edge cases
            ("strands/*/*/state", "strands/+/+/state"),
            ("**", "#"),
            ("*", "+"),
            ("a/b/c", "a/b/c"),
        ],
    )
    def test_translation(self, zenoh, mqtt):
        assert _zenoh_to_mqtt_filter(zenoh) == mqtt


# Per-topic QoS lookup


class TestTopicPolicy:
    """QoS / retain / drop defaults for our topic scheme."""

    def test_presence_is_qos1_retained(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/presence")
        assert qos == 1
        assert retain is True

    def test_state_is_qos0_no_retain(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/state")
        assert qos == 0
        assert retain is False

    def test_cmd_is_qos1(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/cmd")
        assert qos == 1
        assert retain is False

    def test_response_is_qos1(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/response/abc123def")
        assert qos == 1

    def test_broadcast_is_qos1(self):
        qos, retain = _qos_and_retain_for("strands/broadcast")
        assert qos == 1

    def test_health_is_retained(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/health")
        assert retain is True

    def test_safety_event_qos1_retained(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/safety/event")
        assert qos == 1
        assert retain is True

    def test_safety_estop_qos1_retained(self):
        qos, retain = _qos_and_retain_for("strands/safety/estop")
        assert qos == 1
        assert retain is True

    def test_lidar_summary(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/lidar/summary")
        assert qos == 0
        assert retain is False

    def test_lidar_state_retained(self):
        qos, retain = _qos_and_retain_for("strands/so100-01/lidar/state")
        assert retain is True

    def test_unknown_topic_default_qos0_no_retain(self):
        qos, retain = _qos_and_retain_for("strands/peer/somethingnew")
        assert qos == 0
        assert retain is False

    def test_camera_returns_drop(self):
        qos, retain = _qos_and_retain_for("strands/peer/camera/wrist")
        assert qos == -1  # explicit DROP


# Drop list (LAN-only topics)


class TestShouldDrop:
    @pytest.mark.parametrize(
        "topic,expected",
        [
            ("strands/peer/camera/wrist", True),
            ("strands/peer/camera/front", True),
            ("strands/peer/input/leader", True),
            ("strands/peer/input/gamepad", True),
            ("strands/peer/hand/right/state", True),
            ("strands/peer/presence", False),
            ("strands/peer/state", False),
            ("strands/peer/cmd", False),
            ("strands/peer/response/abc", False),
        ],
    )
    def test_should_drop(self, topic, expected):
        assert _should_drop(topic) is expected


# MQTT topic-filter matching


class TestMqttMatcher:
    """Standard MQTT v5 wildcard semantics."""

    @pytest.mark.parametrize(
        "filter_,topic,expected",
        [
            # Exact
            ("strands/peer1/cmd", "strands/peer1/cmd", True),
            ("strands/peer1/cmd", "strands/peer2/cmd", False),
            # + matches one segment
            ("strands/+/presence", "strands/peer1/presence", True),
            ("strands/+/presence", "strands/peer2/presence", True),
            ("strands/+/presence", "strands/peer1/state", False),
            ("strands/+/presence", "strands/a/b/presence", False),
            # # matches tail (zero or more)
            ("strands/peer/response/#", "strands/peer/response/abc", True),
            ("strands/peer/response/#", "strands/peer/response/a/b/c", True),
            ("strands/peer/response/#", "strands/peer/response", True),
            ("strands/peer/response/#", "strands/peer/cmd", False),
            # Different lengths
            ("strands/peer/cmd", "strands/peer/cmd/extra", False),
            ("strands/peer/cmd/extra", "strands/peer/cmd", False),
        ],
    )
    def test_match(self, filter_, topic, expected):
        assert _mqtt_topic_matches(filter_, topic) is expected


# IotMqttTransport — no live broker


class TestIotMqttTransportConfig:
    """Transport config validation without touching a live broker."""

    def test_missing_thing_name_returns_false(self, monkeypatch):
        monkeypatch.delenv("STRANDS_IOT_THING_NAME", raising=False)
        monkeypatch.delenv("STRANDS_IOT_ENDPOINT", raising=False)
        t = IotMqttTransport()
        assert t.connect() is False

    def test_missing_endpoint_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRANDS_IOT_THING_NAME", "test-thing")
        monkeypatch.delenv("STRANDS_IOT_ENDPOINT", raising=False)
        monkeypatch.setenv("STRANDS_IOT_CERT_DIR", str(tmp_path))
        t = IotMqttTransport()
        assert t.connect() is False

    def test_missing_cert_files_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRANDS_IOT_THING_NAME", "test-thing")
        monkeypatch.setenv("STRANDS_IOT_ENDPOINT", "x.iot.us-west-2.amazonaws.com")
        monkeypatch.setenv("STRANDS_IOT_CERT_DIR", str(tmp_path))
        # No cert files in tmp_path
        t = IotMqttTransport()
        assert t.connect() is False

    def test_thing_name_property(self):
        t = IotMqttTransport(
            thing_name="so100-spike-01",
            endpoint="x.iot.us-west-2.amazonaws.com",
        )
        assert t.thing_name == "so100-spike-01"

    def test_put_no_op_when_disconnected(self):
        """put() must NOT raise even when the client is None."""
        t = IotMqttTransport(thing_name="test", endpoint="x")
        # Should not raise
        t.put("strands/test/state", {"k": 1})

    def test_close_idempotent(self):
        """close() is safe to call before connect() and twice in a row."""
        t = IotMqttTransport(thing_name="test", endpoint="x")
        t.close()  # before connect
        t.close()  # double-close

    def test_is_alive_false_when_not_connected(self):
        t = IotMqttTransport(thing_name="test", endpoint="x")
        assert t.is_alive() is False


# ZenohTransport — delegating to mesh.session


class TestZenohTransportDelegation:
    """ZenohTransport is a thin wrapper over mesh.session."""

    def test_satisfies_protocol(self):
        t = ZenohTransport()
        assert isinstance(t, MeshTransport)

    def test_close_before_connect_is_safe(self):
        t = ZenohTransport()
        t.close()  # No-op, no exception

    def test_put_no_op_when_disconnected(self):
        """put() must NOT raise when no session exists."""
        # Make sure no session is open from a prior test
        from strands_robots.mesh import session as sess_mod

        with sess_mod._SESSION_LOCK:
            if sess_mod._SESSION is not None:
                try:
                    sess_mod._SESSION.close()
                except Exception:
                    pass
                sess_mod._SESSION = None
                sess_mod._SESSION_REFS = 0
        t = ZenohTransport()
        # Should be safe — delegates to session.put which is a no-op.
        t.put("strands/test/state", {"k": 1})

    def test_is_alive_false_when_no_session(self):
        from strands_robots.mesh import session as sess_mod

        with sess_mod._SESSION_LOCK:
            if sess_mod._SESSION is not None:
                try:
                    sess_mod._SESSION.close()
                except Exception:
                    pass
                sess_mod._SESSION = None
                sess_mod._SESSION_REFS = 0
        t = ZenohTransport()
        assert t.is_alive() is False

    def test_connect_pre_seeds_session_then_close_releases(self):
        """When session is pre-seeded, connect() takes a ref; close() releases it."""
        from strands_robots.mesh import session as sess_mod

        # Pre-seed: simulate "session already open with 0 refs" — this is
        # the state right after a get_session→release_session cycle would
        # leave it if it didn't auto-close. We construct it manually here
        # because we don't want this test to require zenoh as a hard dep.
        mock_session = MagicMock()
        with sess_mod._SESSION_LOCK:
            # Save state
            saved_session = sess_mod._SESSION
            saved_refs = sess_mod._SESSION_REFS
            sess_mod._SESSION = mock_session
            sess_mod._SESSION_REFS = 1  # already-open singleton
        try:
            t = ZenohTransport()
            ok = t.connect()
            assert ok is True
            assert t.is_alive() is True
            assert sess_mod._SESSION is mock_session
            assert sess_mod._SESSION_REFS == 2  # transport added one ref

            # Second connect on same instance: no-op (we already hold the ref)
            assert t.connect() is True
            assert sess_mod._SESSION_REFS == 2

            t.close()
            assert sess_mod._SESSION_REFS == 1  # transport released its one ref
            assert sess_mod._SESSION is mock_session  # still open

            # Double close: idempotent
            t.close()
            assert sess_mod._SESSION_REFS == 1
        finally:
            with sess_mod._SESSION_LOCK:
                sess_mod._SESSION = saved_session
                sess_mod._SESSION_REFS = saved_refs


class TestIotMqttTransportInternals:
    """White-box tests for IotMqttTransport methods that are not normally
    reached without a live broker (callbacks, _unsubscribe, _on_publish_received)."""

    def setup_method(self):
        """Install a mock awscrt module so lazy imports inside IoT transport work."""
        import sys
        from types import SimpleNamespace

        self._mock_mqtt5 = MagicMock()
        self._mock_mqtt5.QoS.AT_MOST_ONCE = 0
        self._mock_mqtt5.QoS.AT_LEAST_ONCE = 1
        # PublishPacket/UnsubscribePacket capture kwargs as attributes
        self._mock_mqtt5.PublishPacket = lambda **kw: SimpleNamespace(**kw)
        self._mock_mqtt5.UnsubscribePacket = lambda **kw: SimpleNamespace(**kw)
        self._mock_awscrt = MagicMock()
        self._mock_awscrt.mqtt5 = self._mock_mqtt5
        self._saved_awscrt = sys.modules.get("awscrt")
        self._saved_mqtt5 = sys.modules.get("awscrt.mqtt5")
        sys.modules["awscrt"] = self._mock_awscrt
        sys.modules["awscrt.mqtt5"] = self._mock_mqtt5

    def teardown_method(self):
        """Restore original module state."""
        import sys

        if self._saved_awscrt is None:
            sys.modules.pop("awscrt", None)
        else:
            sys.modules["awscrt"] = self._saved_awscrt
        if self._saved_mqtt5 is None:
            sys.modules.pop("awscrt.mqtt5", None)
        else:
            sys.modules["awscrt.mqtt5"] = self._saved_mqtt5

    def _make_transport_with_fake_client(self):
        from strands_robots.mesh.transport.iot_transport import IotMqttTransport

        t = IotMqttTransport(thing_name="test-thing", endpoint="x.iot")
        # Pretend connect() succeeded — directly install a fake client + flag.
        t._client = MagicMock()
        t._connected.set()
        return t

    def test_on_connection_success_sets_flag(self):
        t = self._make_transport_with_fake_client()
        t._connected.clear()
        t._on_connection_success(MagicMock())
        assert t._connected.is_set()

    def test_on_connection_failure_clears_flag(self):
        t = self._make_transport_with_fake_client()
        assert t._connected.is_set()
        data = MagicMock()
        data.exception = RuntimeError("net down")
        t._on_connection_failure(data)
        assert not t._connected.is_set()

    def test_on_disconnection_clears_flag(self):
        t = self._make_transport_with_fake_client()
        assert t._connected.is_set()
        t._on_disconnection(MagicMock())
        assert not t._connected.is_set()

    def test_close_idempotent_when_client_is_none(self):
        from strands_robots.mesh.transport.iot_transport import IotMqttTransport

        t = IotMqttTransport(thing_name="x", endpoint="y")
        t.close()  # no client yet
        t.close()  # double-close
        assert t._client is None

    def test_close_clears_handlers(self):
        t = self._make_transport_with_fake_client()
        t._handlers["filter1"] = [lambda s: None]
        t._handlers["filter2"] = [lambda s: None, lambda s: None]
        t.close()
        assert t._handlers == {}
        assert not t._connected.is_set()

    def test_put_no_op_when_disconnected(self):
        from strands_robots.mesh.transport.iot_transport import IotMqttTransport

        t = IotMqttTransport(thing_name="x", endpoint="y")
        # _client is None, _connected not set — must early-return without raising
        t.put("strands/p/state", {"k": 1})
        # Also when only one of the two is missing
        t._client = MagicMock()
        # _connected still not set
        t.put("strands/p/state", {"k": 1})
        t._client.publish.assert_not_called()

    def test_put_drops_camera_topics(self):
        t = self._make_transport_with_fake_client()
        t.put("strands/p/camera/wrist", {"data": "..."})
        t._client.publish.assert_not_called()

    def test_put_drops_input_topics(self):
        t = self._make_transport_with_fake_client()
        t.put("strands/p/input/leader", {"action": {}})
        t._client.publish.assert_not_called()

    def test_put_drops_hand_topics(self):
        t = self._make_transport_with_fake_client()
        t.put("strands/p/hand/right/state", {"x": 1})
        t._client.publish.assert_not_called()

    def test_put_publishes_with_correct_qos_for_presence(self):
        t = self._make_transport_with_fake_client()
        t.put("strands/p/presence", {"v": 1})
        assert t._client.publish.called
        pkt = t._client.publish.call_args.args[0]
        # presence is QoS 1, retained
        assert pkt.retain is True

    def test_put_publishes_state_qos0_no_retain(self):
        t = self._make_transport_with_fake_client()
        t.put("strands/p/state", {"v": 1})
        assert t._client.publish.called
        pkt = t._client.publish.call_args.args[0]
        assert pkt.retain is False

    def test_put_swallows_publish_errors(self):
        t = self._make_transport_with_fake_client()
        t._client.publish.side_effect = RuntimeError("network")
        # Must not raise — preserves Mesh.put() fire-and-forget contract
        t.put("strands/p/state", {"k": 1})

    def test_unsubscribe_removes_handler_then_unsubscribes_at_broker(self):
        t = self._make_transport_with_fake_client()

        def h1(s):
            pass

        def h2(s):
            pass

        t._handlers["strands/+/cmd"] = [h1, h2]

        # Removing one handler — broker subscription stays
        t._unsubscribe("strands/+/cmd")
        assert len(t._handlers["strands/+/cmd"]) == 1
        t._client.unsubscribe.assert_not_called()

        # Removing the last handler — broker unsubscribe is sent
        t._unsubscribe("strands/+/cmd")
        assert "strands/+/cmd" not in t._handlers
        t._client.unsubscribe.assert_called_once()

    def test_unsubscribe_unknown_filter_is_noop(self):
        t = self._make_transport_with_fake_client()
        t._unsubscribe("never-subscribed")
        t._client.unsubscribe.assert_not_called()

    def test_unsubscribe_swallows_broker_errors(self):
        t = self._make_transport_with_fake_client()
        t._handlers["x"] = [lambda s: None]
        t._client.unsubscribe.side_effect = RuntimeError("broker dead")
        # Must not raise
        t._unsubscribe("x")
        assert "x" not in t._handlers

    def test_on_publish_received_routes_to_matching_handlers(self):
        t = self._make_transport_with_fake_client()
        seen = []
        t._handlers["strands/+/state"] = [lambda s: seen.append(("a", s.key_expr))]
        t._handlers["strands/+/presence"] = [lambda s: seen.append(("b", s.key_expr))]

        # Build a fake publish_packet — keep awscrt's actual shape (.topic,.payload bytes)
        data = MagicMock()
        data.publish_packet.topic = "strands/peer1/state"
        data.publish_packet.payload = b'{"k":1}'
        t._on_publish_received(data)

        assert ("a", "strands/peer1/state") in seen
        assert all(label != "b" for label, _ in seen), "presence handler must not fire"

    def test_on_publish_received_no_match_is_noop(self):
        t = self._make_transport_with_fake_client()
        t._handlers["strands/peer1/cmd"] = [lambda s: None]
        data = MagicMock()
        data.publish_packet.topic = "strands/peer-other/cmd"
        data.publish_packet.payload = b"{}"
        # No raise, no handler fires
        t._on_publish_received(data)

    def test_on_publish_received_swallows_handler_errors(self):
        t = self._make_transport_with_fake_client()
        good_seen = []
        t._handlers["strands/+/state"] = [
            lambda s: (_ for _ in ()).throw(RuntimeError("handler boom")),
            lambda s: good_seen.append(s.key_expr),
        ]
        data = MagicMock()
        data.publish_packet.topic = "strands/p/state"
        data.publish_packet.payload = b"{}"
        # Must not raise; the second handler MUST still fire
        t._on_publish_received(data)
        assert good_seen == ["strands/p/state"]

    def test_declare_subscriber_when_disconnected_raises(self):
        from strands_robots.mesh.transport.iot_transport import IotMqttTransport

        t = IotMqttTransport(thing_name="x", endpoint="y")
        with pytest.raises(RuntimeError, match="not connected"):
            t.declare_subscriber("strands/+/cmd", lambda s: None)

    def test_subhandle_double_undeclare_is_safe(self):
        from strands_robots.mesh.transport.iot_transport import _MqttSubHandle

        t = self._make_transport_with_fake_client()
        t._handlers["strands/+/cmd"] = [lambda s: None]
        h = _MqttSubHandle(t, "strands/+/cmd")
        h.undeclare()
        # Second undeclare — guarded by _undeclared flag
        h.undeclare()
        # Broker unsubscribe should be called exactly once
        assert t._client.unsubscribe.call_count == 1
