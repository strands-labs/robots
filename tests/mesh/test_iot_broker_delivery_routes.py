#!/usr/bin/env python3
"""Routes that decide whether an IoT MQTT message reaches the broker at all.

``tests/mesh/test_iot_transport_session.py`` pins the common publish/subscribe
path and the ``camera/<name>`` never-bridge prefix. This module covers the
remaining bookkeeping routes on that path -- the ones a message has to survive
to be published, and the ones a subscription has to survive to be torn down:

  - the explicit ``DROP`` short-circuit in :meth:`put`. It is *not* redundant
    with ``_should_drop``: that prefix test requires a trailing ``camera/``
    segment, so a bare ``strands/<peer>/camera`` passes it and the ``qos < 0``
    branch is the only thing that stops the publish.
  - ``_unsubscribe`` with a handler that is already gone, which must not
    disturb the subscribers that remain.
  - ``_unsubscribe`` reaching the broker step after a *failed reconnect* left
    the client ``None`` while the handler map was still populated. Only
    ``close()`` clears that map, so this is reachable from the public API. The
    broad ``except`` around the broker call would swallow the resulting
    ``AttributeError``, so what the guard actually buys is silence: without it
    an ordinary teardown logs an internal ``NoneType`` error against a session
    the caller already knows is down.

One layout-(a) ``DROP`` branch is unreachable by construction rather than
untested; :class:`TestDropPolicyReachability` pins the property that makes it so.

The AWS IoT SDK is never reached over the network: ``mtls_from_path`` is patched
to hand back the session module's ``_FakeMqtt5Client``.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from strands_robots.mesh.transport.iot_transport import (
    _TOPIC_POLICY,
    _qos_and_retain_for,
    _should_drop,
)

from .test_iot_transport_session import _connect, _FakeMqtt5Client

_STATE_FILTER = "strands/+/state"


@pytest.fixture
def builder(monkeypatch):
    """Patch ``mtls_from_path`` to return a captured fake client.

    ``build_exc_after`` makes the *n*-th build raise, which is how a reconnect
    is failed: :meth:`connect` nulls the stale client before rebuilding, so a
    build that raises leaves ``_client`` ``None`` with the handler map intact.
    """
    import awsiot.mqtt5_client_builder as awsiot_builder

    holder: dict[str, Any] = {
        "client": None,
        "auto_connack": True,
        "subscribe_exc": None,
        "build_calls": 0,
        "build_exc_after": None,
    }

    def fake_mtls_from_path(**kwargs):
        holder["build_calls"] += 1
        if holder["build_exc_after"] is not None and holder["build_calls"] > holder["build_exc_after"]:
            raise RuntimeError("simulated corrupt PEM on reconnect")
        client = _FakeMqtt5Client(holder=holder, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr(awsiot_builder, "mtls_from_path", fake_mtls_from_path)
    return holder


class TestExplicitDropShortCircuit:
    """A ``DROP`` policy entry stops the publish even when the prefix test does not."""

    def test_bare_camera_topic_is_not_caught_by_the_prefix_test(self):
        # Premise for the test below: _should_drop matches on a trailing
        # 'camera/' segment, so the bare kind slips past it entirely.
        assert _should_drop("strands/thor-arm/camera") is False
        assert _should_drop("strands/thor-arm/camera/wrist") is True

    def test_bare_camera_topic_resolves_to_the_drop_sentinel(self):
        assert _qos_and_retain_for("strands/thor-arm/camera") == (-1, False)

    def test_bare_camera_topic_is_never_published(self, tmp_path, builder):
        # The only route that stops this one: the prefix test above passes it.
        t = _connect(tmp_path)
        t.connect()
        t.put("strands/thor-arm/camera", {"jpeg": "x" * 64})
        assert builder["client"].published == []

    def test_a_usable_topic_on_the_same_session_still_publishes(self, tmp_path, builder):
        # Over-reach control: the short-circuit is scoped to DROP topics.
        t = _connect(tmp_path)
        t.connect()
        t.put("strands/thor-arm/camera", {"jpeg": "x"})
        t.put("strands/thor-arm/state", {"x": 1})
        assert [p.topic for p in builder["client"].published] == ["strands/thor-arm/state"]

    def test_camera_reference_metadata_is_exempt_from_both_routes(self, tmp_path, builder):
        # The /ref pointer is small and cloud-relevant, so neither route drops it.
        t = _connect(tmp_path)
        t.connect()
        t.put("strands/thor-arm/camera/wrist/ref", {"key": "s3://b/k"})
        assert [p.topic for p in builder["client"].published] == ["strands/thor-arm/camera/wrist/ref"]


class TestUnsubscribeBookkeeping:
    """Removing a handler twice, or after the client is gone, stays quiet."""

    def test_removing_an_already_gone_handler_keeps_the_live_subscriber(self, tmp_path, builder):
        t = _connect(tmp_path)
        t.connect()
        seen: list[Any] = []
        first = lambda sample: None  # noqa: E731 - identity is what is removed
        t.declare_subscriber(_STATE_FILTER, first)
        t.declare_subscriber(_STATE_FILTER, lambda sample: seen.append(sample))

        t._unsubscribe(_STATE_FILTER, first)
        t._unsubscribe(_STATE_FILTER, first)  # already gone: must not raise

        # The surviving subscriber still receives, and the broker subscription
        # was never torn down.
        builder["client"].fire_inbound("strands/thor-arm/state", b"{}")
        assert len(seen) == 1
        assert builder["client"].unsubscribed == []

    def test_undeclare_after_a_failed_reconnect_is_quiet(self, tmp_path, builder, caplog):
        t = _connect(tmp_path)
        t.connect()
        handle = t.declare_subscriber(_STATE_FILTER, lambda sample: None)
        client = builder["client"]

        # Broker drops, then a caller-driven reconnect fails while building the
        # replacement client: _client is None and the handler map is untouched
        # (only close() clears it).
        client.fire_disconnect()
        builder["build_exc_after"] = 1
        assert t.connect() is False

        with caplog.at_level(logging.DEBUG, logger="strands_robots.mesh.transport.iot_transport"):
            handle.undeclare()  # no client to unsubscribe against

        # The broker call is never attempted, so the teardown records nothing:
        # reaching it would log an internal NoneType error against a session the
        # caller already knows is down.
        assert client.unsubscribed == []
        assert [r.getMessage() for r in caplog.records if "unsubscribe error" in r.getMessage()] == []
        assert t.is_alive() is False

    def test_undeclare_with_a_live_client_still_unsubscribes(self, tmp_path, builder):
        # Over-reach control for the route above: the guard is about a missing
        # client, not about skipping the broker step in general.
        t = _connect(tmp_path)
        t.connect()
        handle = t.declare_subscriber(_STATE_FILTER, lambda sample: None)
        handle.undeclare()
        assert [p.topic_filters for p in builder["client"].unsubscribed] == [[_STATE_FILTER]]


class TestDropPolicyReachability:
    """Why the layout-(a) ``DROP`` branch is unreachable rather than untested.

    ``_qos_and_retain_for`` resolves two topic layouts. Layout (a) is entered
    only when the first segment is a reserved top-level kind, and every
    candidate it builds therefore starts with that kind -- so a ``DROP`` entry
    can only be reached from layout (b). This pin fails the day a top-level
    kind gains a ``DROP`` entry, which is the point at which that branch needs
    a behavioural test of its own.
    """

    def test_no_drop_entry_is_reachable_from_a_top_level_layout(self):
        top_level_kinds = {"broadcast", "safety"}
        drop_keys = [key for key, (qos, _retain) in _TOPIC_POLICY.items() if qos == "DROP"]
        assert drop_keys, "non-vacuity: the policy table must declare at least one DROP entry"
        for key in drop_keys:
            assert key.split("/")[0] not in top_level_kinds, (
                f"policy DROP entry {key!r} is now reachable from the top-level layout; "
                "cover that branch in TestExplicitDropShortCircuit"
            )

    def test_a_top_level_kind_with_no_entry_falls_through_rather_than_dropping(self):
        # The fall-through returns the safe default, never the DROP sentinel.
        assert _qos_and_retain_for("strands/safety/unknown-kind") == (0, False)
