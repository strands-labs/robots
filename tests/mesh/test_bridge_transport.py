"""Unit tests for BridgeTransport (Zenoh + IoT fan-out).

No real network - exercises the topic filter logic, suffix matching,
fan-out behaviour, subscription lifecycle, and graceful degradation when
either side fails.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands_robots.mesh.transport.bridge_transport import (
    _DEFAULT_BRIDGE_PREFIX_SUFFIXES,
    _DEFAULT_DEDUP_TTL_S,
    DEFAULT_BRIDGE_SUFFIXES,
    BridgeTransport,
    _BridgeSubHandle,
    _resolve_bridge_filter,
    _resolve_bridge_prefix_filter,
    _resolve_dedup_ttl,
    _should_bridge,
    _topic_suffix,
)

# Topic suffix extraction


class TestTopicSuffix:
    @pytest.mark.parametrize(
        "topic,expected",
        [
            ("strands/peer1/state", "state"),
            ("strands/peer1/lidar/summary", "lidar/summary"),
            ("strands/peer1/camera/wrist", "camera/wrist"),
            ("strands/broadcast", "broadcast"),
            ("strands/safety/estop", "safety/estop"),
            ("strands/peer1/safety/event", "safety/event"),
            ("not-strands/foo", ""),
        ],
    )
    def test_extracts_suffix(self, topic, expected):
        assert _topic_suffix(topic) == expected


# Default filter


class TestDefaultFilter:
    def test_default_set_contains_safety_topics(self):
        assert "safety/event" in DEFAULT_BRIDGE_SUFFIXES
        assert "safety/estop" in DEFAULT_BRIDGE_SUFFIXES
        assert "broadcast" in DEFAULT_BRIDGE_SUFFIXES
        assert "cmd" in DEFAULT_BRIDGE_SUFFIXES
        assert "response" in DEFAULT_BRIDGE_SUFFIXES
        assert "presence" in DEFAULT_BRIDGE_SUFFIXES
        assert "health" in DEFAULT_BRIDGE_SUFFIXES

    def test_default_excludes_high_volume(self):
        assert "state" not in DEFAULT_BRIDGE_SUFFIXES
        assert "pose" not in DEFAULT_BRIDGE_SUFFIXES
        assert "imu" not in DEFAULT_BRIDGE_SUFFIXES
        assert "input" not in DEFAULT_BRIDGE_SUFFIXES
        assert "camera" not in DEFAULT_BRIDGE_SUFFIXES


class TestEnvFilter:
    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BRIDGE_TOPICS", "presence,state")
        f = _resolve_bridge_filter()
        assert "presence" in f
        assert "state" in f
        assert "safety/event" not in f  # not in env list

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BRIDGE_TOPICS", "")
        f = _resolve_bridge_filter()
        assert f == DEFAULT_BRIDGE_SUFFIXES

    def test_unset_env_uses_default(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_BRIDGE_TOPICS", raising=False)
        assert _resolve_bridge_filter() == DEFAULT_BRIDGE_SUFFIXES

    def test_whitespace_only_env_falls_back_to_default(self, monkeypatch):
        """A comma/whitespace-only value yields no usable suffixes.

        After splitting and stripping, ``parts`` is empty, so the reader
        must fall back to the documented default rather than returning an
        empty (bridge-nothing) suffix set. Covers the ``if not parts``
        branch in ``_resolve_bridge_filter``.
        """
        monkeypatch.setenv("STRANDS_MESH_BRIDGE_TOPICS", "  ,  , ")
        assert _resolve_bridge_filter() == DEFAULT_BRIDGE_SUFFIXES


class TestDedupTtlEnv:
    """Pin the STRANDS_MESH_DEDUP_TTL operator knob contract.

    ``_resolve_dedup_ttl`` sets the TTL of the bridge command dedup cache.
    The documented contract (bridge_transport.py) is: unset -> 120s default,
    a positive float -> that value, and any non-positive or unparseable
    value -> the default (a bad env value must never silently disable or
    invert dedup, which would let duplicates through or over-suppress).
    """

    _HINT = (
        "STRANDS_MESH_DEDUP_TTL reader drifted from its documented contract; "
        "cross-check _resolve_dedup_ttl and _DEFAULT_DEDUP_TTL_S in "
        "bridge_transport.py"
    )

    def test_unset_env_uses_default(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_DEDUP_TTL", raising=False)
        assert _resolve_dedup_ttl() == _DEFAULT_DEDUP_TTL_S, self._HINT

    def test_positive_float_is_honored(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_DEDUP_TTL", "45.5")
        assert _resolve_dedup_ttl() == 45.5, self._HINT

    @pytest.mark.parametrize("raw", ["0", "0.0", "-5", "-0.1"])
    def test_non_positive_falls_back_to_default(self, monkeypatch, raw):
        """Zero/negative TTL is meaningless for a cache window -> default.

        Guards the ``v if v > 0 else default`` branch: a regression that
        returned ``v`` directly would disable dedup (ttl<=0 expires every
        entry immediately) and let cross-transport duplicates through.
        """
        monkeypatch.setenv("STRANDS_MESH_DEDUP_TTL", raw)
        assert _resolve_dedup_ttl() == _DEFAULT_DEDUP_TTL_S, self._HINT

    def test_unparseable_value_falls_back_to_default(self, monkeypatch):
        """A non-numeric value degrades to the default (ValueError branch)."""
        monkeypatch.setenv("STRANDS_MESH_DEDUP_TTL", "banana")
        assert _resolve_dedup_ttl() == _DEFAULT_DEDUP_TTL_S, self._HINT


class TestEnvPrefixFilter:
    """Pin STRANDS_MESH_BRIDGE_TOPICS_PREFIX reader and its unset-default.

    Regression for issue #244: the env-var reader (_resolve_bridge_prefix_filter)
    and its unset-default must stay consistent with the Phase-4 cloud-pollution
    hardening documented in bridge_transport.py:107-121 and the
    _DEFAULT_BRIDGE_PREFIX_SUFFIXES constant (bridge_transport.py:140-155).
    """

    _READER_HINT = (
        "reader _resolve_bridge_prefix_filter() drifted from documented default; "
        "cross-check bridge_transport.py _DEFAULT_BRIDGE_PREFIX_SUFFIXES "
        "and the Phase-4 hardening comment at bridge_transport.py:107-121"
    )

    def test_env_var_name_parses_comma_separated_prefixes(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BRIDGE_TOPICS_PREFIX", "response, telemetry")
        f = _resolve_bridge_prefix_filter()
        assert "response" in f, self._READER_HINT
        assert "telemetry" in f, self._READER_HINT
        # Verify whitespace is stripped (the reader calls .strip())
        assert " telemetry" not in f, "reader must strip whitespace around entries"

    def test_unset_default_is_response_prefix(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_BRIDGE_TOPICS_PREFIX", raising=False)
        f = _resolve_bridge_prefix_filter()
        # Literal pin is the authoritative assertion; constant equality is a
        # bonus sanity check that catches symbol renames.
        assert f == frozenset({"response"}), self._READER_HINT
        assert f == _DEFAULT_BRIDGE_PREFIX_SUFFIXES, self._READER_HINT

    def test_unset_default_bridges_response_tail(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_BRIDGE_TOPICS_PREFIX", raising=False)
        prefixes = _resolve_bridge_prefix_filter()
        assert (
            _should_bridge(
                "strands/peer1/response/turn-42",
                DEFAULT_BRIDGE_SUFFIXES,
                prefixes,
            )
            is True
        ), self._READER_HINT

    def test_unset_default_blocks_cmd_tail(self, monkeypatch):
        """Pin the Phase-4 security property: cmd is exact-match only.

        A future refactor that widens _DEFAULT_BRIDGE_PREFIX_SUFFIXES to
        include 'cmd' (or reverts the exact/prefix split) would let
        strands/<peer>/cmd/<attacker-tail> bridge to MQTT -- the exact
        regression #244 exists to prevent.
        """
        monkeypatch.delenv("STRANDS_MESH_BRIDGE_TOPICS_PREFIX", raising=False)
        prefixes = _resolve_bridge_prefix_filter()
        assert (
            _should_bridge(
                "strands/peer1/cmd/attacker-blob",
                DEFAULT_BRIDGE_SUFFIXES,
                prefixes,
            )
            is False
        ), (
            "cmd must be exact-match only; cmd/<tail> must NOT bridge. "
            "See Phase-4 hardening in bridge_transport.py:107-121"
        )

    def test_empty_env_prefix_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BRIDGE_TOPICS_PREFIX", "")
        assert _resolve_bridge_prefix_filter() == _DEFAULT_BRIDGE_PREFIX_SUFFIXES, self._READER_HINT

    def test_whitespace_only_env_prefix_falls_back_to_default(self, monkeypatch):
        """Covers the 'if not parts' branch when env is whitespace-only."""
        monkeypatch.setenv("STRANDS_MESH_BRIDGE_TOPICS_PREFIX", "   ")
        assert _resolve_bridge_prefix_filter() == _DEFAULT_BRIDGE_PREFIX_SUFFIXES, self._READER_HINT

    def test_topics_env_does_not_leak_into_prefix_default(self, monkeypatch):
        """Pin independence of STRANDS_MESH_BRIDGE_TOPICS vs _PREFIX."""
        monkeypatch.setenv("STRANDS_MESH_BRIDGE_TOPICS", "foo,bar")
        monkeypatch.delenv("STRANDS_MESH_BRIDGE_TOPICS_PREFIX", raising=False)
        assert _resolve_bridge_prefix_filter() == frozenset({"response"})


# _should_bridge - the real fan-out gate


class TestShouldBridge:
    @pytest.mark.parametrize(
        "topic,allowed",
        [
            # Allowed by default
            ("strands/peer1/presence", True),
            ("strands/peer1/health", True),
            ("strands/peer1/cmd", True),
            ("strands/peer1/response/abc123", True),
            ("strands/broadcast", True),
            ("strands/peer1/safety/event", True),
            ("strands/safety/estop", True),
            # Blocked by default
            ("strands/peer1/state", False),
            ("strands/peer1/pose", False),
            ("strands/peer1/imu", False),
            ("strands/peer1/odom", False),
            ("strands/peer1/lidar/summary", False),
            ("strands/peer1/camera/wrist", False),
            ("strands/peer1/input/leader", False),
            ("strands/peer1/hand/right/state", False),
            # Outside the strands/ namespace - never bridges
            ("not-strands/foo", False),
        ],
    )
    def test_bridge_decisions_match_default(self, topic, allowed):
        assert _should_bridge(topic, DEFAULT_BRIDGE_SUFFIXES) is allowed


# BridgeTransport behaviour - both transports mocked


@pytest.fixture
def fake_transports():
    """A pair of MagicMock-backed Zenoh + IoT transports plumbed together."""
    z = MagicMock()
    z.connect.return_value = True
    z.is_alive.return_value = True
    i = MagicMock()
    i.connect.return_value = True
    i.is_alive.return_value = True
    return z, i


class TestBridgeConnectAndClose:
    def test_succeeds_when_both_succeed(self, fake_transports):
        z, i = fake_transports
        b = BridgeTransport(zenoh=z, iot=i)
        assert b.connect() is True
        assert b.is_alive() is True

    def test_succeeds_when_only_zenoh_succeeds(self, fake_transports):
        z, i = fake_transports
        i.connect.return_value = False
        i.is_alive.return_value = False
        b = BridgeTransport(zenoh=z, iot=i)
        assert b.connect() is True

    def test_succeeds_when_only_iot_succeeds(self, fake_transports):
        z, i = fake_transports
        z.connect.return_value = False
        z.is_alive.return_value = False
        b = BridgeTransport(zenoh=z, iot=i)
        assert b.connect() is True

    def test_fails_when_both_fail(self, fake_transports):
        z, i = fake_transports
        z.connect.return_value = False
        z.is_alive.return_value = False
        i.connect.return_value = False
        i.is_alive.return_value = False
        b = BridgeTransport(zenoh=z, iot=i)
        assert b.connect() is False

    def test_close_idempotent(self, fake_transports):
        z, i = fake_transports
        b = BridgeTransport(zenoh=z, iot=i)
        b.connect()
        b.close()
        b.close()  # Should not raise
        # Both close()s called once each
        assert z.close.call_count == 2
        assert i.close.call_count == 2


class TestBridgeFanOutPut:
    def test_state_publishes_only_to_zenoh(self, fake_transports):
        """Default filter excludes ``state`` - must not bridge to MQTT."""
        z, i = fake_transports
        b = BridgeTransport(zenoh=z, iot=i)
        b.connect()
        b.put("strands/peer1/state", {"k": 1})
        z.put.assert_called_once_with("strands/peer1/state", {"k": 1})
        i.put.assert_not_called()

    def test_presence_publishes_to_both(self, fake_transports):
        z, i = fake_transports
        b = BridgeTransport(zenoh=z, iot=i)
        b.connect()
        b.put("strands/peer1/presence", {"x": 1})
        z.put.assert_called_once()
        i.put.assert_called_once()

    def test_camera_publishes_only_to_zenoh(self, fake_transports):
        """Camera frames should never traverse MQTT."""
        z, i = fake_transports
        b = BridgeTransport(zenoh=z, iot=i)
        b.connect()
        b.put("strands/peer1/camera/wrist", {"data": "..."})
        z.put.assert_called_once()
        i.put.assert_not_called()

    def test_zenoh_failure_does_not_block_iot(self, fake_transports):
        z, i = fake_transports
        z.put.side_effect = RuntimeError("zenoh broken")
        b = BridgeTransport(zenoh=z, iot=i)
        b.connect()
        # Must not raise, and IoT side must still publish.
        b.put("strands/peer1/presence", {"k": 1})
        i.put.assert_called_once()

    def test_no_publishes_when_neither_alive(self, fake_transports):
        z, i = fake_transports
        z.is_alive.return_value = False
        i.is_alive.return_value = False
        b = BridgeTransport(zenoh=z, iot=i)
        # connect() will fail but put() still must not crash.
        b.put("strands/peer1/presence", {"k": 1})
        z.put.assert_not_called()
        i.put.assert_not_called()


class TestBridgeSubscribe:
    def test_subscribes_on_both_sides(self, fake_transports):
        z, i = fake_transports
        z_sub = MagicMock()
        i_sub = MagicMock()
        z.declare_subscriber.return_value = z_sub
        i.declare_subscriber.return_value = i_sub
        b = BridgeTransport(zenoh=z, iot=i)
        b.connect()
        handler = MagicMock()
        h = b.declare_subscriber("strands/+/presence", handler)
        # The bridge wraps the handler with a dedup filter, so the *handler*
        # passed downstream is not the literal mock -- assert the *topic* is
        # correct on both sides and that a callable was passed.
        assert z.declare_subscriber.call_count == 1
        assert i.declare_subscriber.call_count == 1
        z_topic, z_handler = z.declare_subscriber.call_args.args
        i_topic, i_handler = i.declare_subscriber.call_args.args
        assert z_topic == "strands/+/presence"
        assert i_topic == "strands/+/presence"
        assert callable(z_handler)
        assert callable(i_handler)
        # Verify the wrapper still delegates to the user handler. Drive it
        # with a sample whose payload extracts to a unique nonce so dedup
        # passes through on the first call.
        from unittest.mock import MagicMock as _MM

        sample = _MM()
        sample.payload.to_bytes.return_value = b'{"nonce":"unique-once-test","payload":{}}'
        z_handler(sample)
        handler.assert_called_once_with(sample)
        # Undeclare should call both.
        h.undeclare()
        z_sub.undeclare.assert_called_once()
        i_sub.undeclare.assert_called_once()

    def test_subscribe_failure_on_one_side_still_succeeds(self, fake_transports):
        z, i = fake_transports
        z.declare_subscriber.side_effect = RuntimeError("zenoh sub failed")
        i_sub = MagicMock()
        i.declare_subscriber.return_value = i_sub
        b = BridgeTransport(zenoh=z, iot=i)
        b.connect()
        h = b.declare_subscriber("strands/peer1/cmd", lambda s: None)
        # Only IoT subscribed; undeclare gracefully tears that down.
        h.undeclare()
        i_sub.undeclare.assert_called_once()

    def test_subscribe_failure_on_both_sides_raises(self, fake_transports):
        z, i = fake_transports
        z.declare_subscriber.side_effect = RuntimeError("zenoh sub failed")
        i.declare_subscriber.side_effect = RuntimeError("iot sub failed")
        b = BridgeTransport(zenoh=z, iot=i)
        b.connect()
        with pytest.raises(RuntimeError, match="failed on both sides"):
            b.declare_subscriber("strands/peer1/cmd", lambda s: None)


class TestSubHandleIdempotence:
    def test_double_undeclare_safe(self):
        a, b = MagicMock(), MagicMock()
        h = _BridgeSubHandle(a, b)
        h.undeclare()
        h.undeclare()  # No exception
        a.undeclare.assert_called_once()
        b.undeclare.assert_called_once()

    def test_partial_handles(self):
        """One side missing - only the present one is undeclared."""
        a = MagicMock()
        h = _BridgeSubHandle(a, None)
        h.undeclare()
        a.undeclare.assert_called_once()


class TestDefaultBridgeSuffixesPinned:
    def test_default_bridge_suffixes_pinned_to_documented_set(self):
        """Pin DEFAULT_BRIDGE_SUFFIXES against the documented bridge-by-default set.

        These suffixes are documented in the module header and the mesh env-var
        matrix as the unset-default for STRANDS_MESH_BRIDGE_TOPICS. A future edit
        that adds or removes a suffix here must update that documentation in the
        same diff or this test fails.
        """
        assert DEFAULT_BRIDGE_SUFFIXES == frozenset(
            {
                "presence",
                "health",
                "safety/event",
                "safety/estop",
                "safety/resume",
                "cmd",
                "response",
                "broadcast",
            }
        )

    def test_high_volume_telemetry_not_bridged_by_default(self):
        """Pin the documented high-volume-telemetry not-bridged list.

        state/pose/imu/odom/lidar are high volume and camera/input/hand are
        LAN-only by definition. All are documented as not bridged by default;
        if one leaks into DEFAULT_BRIDGE_SUFFIXES the documentation regresses.
        """
        not_bridged = (
            "state",
            "pose",
            "imu",
            "odom",
            "lidar",
            "camera",
            "input",
            "hand",
        )
        for suffix in not_bridged:
            assert suffix not in DEFAULT_BRIDGE_SUFFIXES, (
                f"{suffix!r} is documented as not bridged by default but is in "
                f"DEFAULT_BRIDGE_SUFFIXES; remove it or update the documentation."
            )


class TestSubHandleTeardownFailSoft:
    """``_BridgeSubHandle.undeclare`` must swallow the documented transport-
    teardown failure surface (already-closed handle / socket race) so idempotent
    cleanup never propagates, while still tearing down the other side and
    letting unexpected exceptions escape."""

    @pytest.mark.parametrize("exc", [RuntimeError, AttributeError, OSError])
    def test_undeclare_swallows_documented_transport_errors(self, exc):
        raising, healthy = MagicMock(), MagicMock()
        raising.undeclare.side_effect = exc("teardown race")
        h = _BridgeSubHandle(raising, healthy)

        h.undeclare()  # must not raise despite the zenoh side failing

        raising.undeclare.assert_called_once()
        # A failure on the first side must not abort teardown of the second.
        healthy.undeclare.assert_called_once()

    def test_undeclare_propagates_unexpected_error(self):
        raising = MagicMock()
        raising.undeclare.side_effect = ValueError("not a teardown error")
        h = _BridgeSubHandle(raising, MagicMock())

        with pytest.raises(ValueError):
            h.undeclare()


class TestBridgeCloseFailSoft:
    """``BridgeTransport.close`` must swallow the documented transport-close
    failure surface on either backend, still attempt to close both, and let
    unexpected exceptions propagate."""

    @pytest.mark.parametrize("exc", [RuntimeError, ConnectionError, OSError])
    def test_close_swallows_documented_transport_errors(self, fake_transports, exc):
        z, i = fake_transports
        z.close.side_effect = exc("zenoh close race")
        i.close.side_effect = exc("iot close race")
        b = BridgeTransport(zenoh=z, iot=i)
        b.connect()

        b.close()  # must not raise despite both backends failing to close

        # A failure closing zenoh must not skip closing iot.
        z.close.assert_called_once()
        i.close.assert_called_once()

    def test_close_propagates_unexpected_error(self, fake_transports):
        z, i = fake_transports
        z.close.side_effect = ValueError("not a close error")
        b = BridgeTransport(zenoh=z, iot=i)
        b.connect()

        with pytest.raises(ValueError):
            b.close()
