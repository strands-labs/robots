"""Resilience + serialization contracts for Mesh.subscribe / unsubscribe / publish_step.

These pin the failure-tolerant edges of the user-facing mesh pub/sub surface so a
future refactor cannot silently regress them:

- ``subscribe`` returns ``None`` (and registers nothing) when the underlying
  Zenoh ``declare_subscriber`` raises, instead of leaking a half-registered sub.
- The per-sample handler swallows a raising user callback so one bad consumer
  cannot kill the subscriber thread.
- ``unsubscribe`` is idempotent for unknown names and tolerates an ``undeclare``
  failure as well as a subscriber already absent from the tracking list.
- ``publish_step`` serializes list/tuple observation and action values into
  JSON-safe lists before publishing the VLA execution step.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh.core import Mesh


class _FakeRobot:
    """Minimal robot stub; Mesh.__init__ only stores the reference."""

    def __init__(self) -> None:
        self.tool_name_str = "resilience-bot"
        self.robot = None


def _running_mesh(peer_id: str = "resilience-peer") -> Mesh:
    """A Mesh in the running state without starting real Zenoh loops."""
    mesh = Mesh(_FakeRobot(), peer_id=peer_id)
    mesh._running = True
    return mesh


class TestSubscribeResilience:
    def test_subscribe_returns_none_when_declare_fails(self, monkeypatch):
        """A declare_subscriber failure yields None and registers no sub."""
        session = MagicMock()
        session.declare_subscriber.side_effect = RuntimeError("router unreachable")
        monkeypatch.setattr(mesh_core, "current_session", lambda: session)

        mesh = _running_mesh()
        result = mesh.subscribe("test/topic", name="t")

        assert result is None
        assert mesh._user_subs == {}
        assert mesh._subs == []

    def test_subscribe_returns_none_when_no_session(self, monkeypatch):
        """No active session -> subscribe is a no-op returning None."""
        monkeypatch.setattr(mesh_core, "current_session", lambda: None)

        mesh = _running_mesh()
        assert mesh.subscribe("test/topic") is None

    def test_handler_swallows_raising_callback(self, monkeypatch):
        """A user callback that raises must not propagate out of the handler."""
        session = MagicMock()
        session.declare_subscriber.return_value = MagicMock()
        monkeypatch.setattr(mesh_core, "current_session", lambda: session)

        calls: list[str] = []

        def bad_callback(key: str, data: dict) -> None:
            calls.append(key)
            raise ValueError("consumer blew up")

        mesh = _running_mesh()
        assert mesh.subscribe("test/topic", callback=bad_callback, name="t") == "t"

        handler = session.declare_subscriber.call_args.args[1]
        sample = MagicMock()
        sample.key_expr = "test/topic"
        sample.payload.to_bytes.return_value = b'{"x": 1}'

        # Must not raise even though the callback does.
        handler(sample)
        assert calls == ["test/topic"]

    def test_handler_non_json_payload_wrapped_as_raw(self, monkeypatch):
        """Non-JSON payloads are delivered as {"raw": <text>} rather than dropped."""
        session = MagicMock()
        session.declare_subscriber.return_value = MagicMock()
        monkeypatch.setattr(mesh_core, "current_session", lambda: session)

        received: list[tuple[str, dict]] = []
        mesh = _running_mesh()
        mesh.subscribe("test/topic", callback=lambda k, d: received.append((k, d)), name="t")

        handler = session.declare_subscriber.call_args.args[1]
        sample = MagicMock()
        sample.key_expr = "test/topic"
        sample.payload.to_bytes.return_value = b"not-json"
        handler(sample)

        assert received == [("test/topic", {"raw": "not-json"})]


class TestUnsubscribeResilience:
    def test_unsubscribe_unknown_name_is_noop(self):
        """Unsubscribing an unregistered name returns quietly."""
        mesh = _running_mesh()
        mesh.unsubscribe("never-registered")  # no raise
        assert mesh._user_subs == {}

    def test_unsubscribe_swallows_undeclare_failure(self):
        """An undeclare() failure is tolerated and inbox state still cleared."""
        mesh = _running_mesh()
        sub = MagicMock()
        sub.undeclare.side_effect = RuntimeError("already gone")
        mesh._subs.append(sub)
        mesh._user_subs["t"] = sub
        mesh.inbox["t"] = [("k", {"v": 1})]

        mesh.unsubscribe("t")

        sub.undeclare.assert_called_once()
        assert "t" not in mesh._user_subs
        assert "t" not in mesh.inbox
        assert sub not in mesh._subs

    def test_unsubscribe_tolerates_sub_missing_from_tracking_list(self):
        """A sub tracked in _user_subs but absent from _subs is handled cleanly."""
        mesh = _running_mesh()
        sub = MagicMock()
        # Registered by name but never added to the _subs list.
        mesh._user_subs["t"] = sub

        mesh.unsubscribe("t")

        sub.undeclare.assert_called_once()
        assert "t" not in mesh._user_subs


class TestPublishStepSerialization:
    def test_publish_step_serializes_list_and_tuple_values(self):
        """List observation values and tuple action values become JSON-safe lists."""
        mesh = _running_mesh("stream-peer")
        published: list[tuple[str, dict]] = []
        mesh.publish = lambda key, payload: published.append((key, payload))

        mesh.publish_step(
            step=7,
            observation={"joints": [0.1, 0.2, 0.3]},
            action={"target": (1.0, 2.0)},
            instruction="pick",
            policy="mock",
        )

        assert len(published) == 1
        key, payload = published[0]
        assert key == "strands/stream-peer/stream"
        assert payload["step"] == 7
        assert payload["instruction"] == "pick"
        assert payload["policy"] == "mock"
        assert payload["observation"] == {"joints": [0.1, 0.2, 0.3]}
        # Tuple action coerced to list so json.dumps on the wire is lossless.
        assert payload["action"] == {"target": [1.0, 2.0]}
        assert isinstance(payload["action"]["target"], list)

    def test_publish_step_drops_multidim_observations(self):
        """Multi-dimensional (image-like) observation arrays are not streamed."""
        np = pytest.importorskip("numpy")
        mesh = _running_mesh("stream-peer")
        published: list[tuple[str, dict]] = []
        mesh.publish = lambda key, payload: published.append((key, payload))

        mesh.publish_step(
            step=0,
            observation={
                "state": np.array([1.0, 2.0]),
                "image": np.zeros((4, 4, 3)),
            },
            action={"gripper": 1},
        )

        _, payload = published[0]
        assert payload["observation"] == {"state": [1.0, 2.0]}
        assert "image" not in payload["observation"]
        assert payload["action"] == {"gripper": 1}

    def test_publish_step_noop_when_not_running(self):
        """publish_step is inert until the mesh is running."""
        mesh = Mesh(_FakeRobot(), peer_id="idle-peer")
        published: list = []
        mesh.publish = lambda key, payload: published.append((key, payload))

        mesh.publish_step(step=0, observation={}, action={})
        assert published == []
