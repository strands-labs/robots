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
- A user subscription never silently fails to exist: every ``subscribe``
  refusal says why, and ``stop`` says how many it dropped. The cases above
  pinned the ``None`` *return* for two of the three refusal paths and nothing
  about the report, so the two that answered without a word - not on the mesh,
  no session - were the ones a caller re-subscribing after a ``stop()`` hit.
"""

from __future__ import annotations

import logging
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


_CORE_LOGGER = "strands_robots.mesh.core"


def _refusals(caplog, topic: str) -> list[str]:
    """Every WARNING naming *topic*, i.e. what the caller is told about it.

    Keyed on the topic rather than a call spelling: the three refusal paths
    word themselves differently (``subscribe(...) refused`` for the two this
    class added, ``declare_subscriber(...) failed`` for the pre-existing one)
    and the contract is that the caller learns which topic did not attach.
    """
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and topic in r.getMessage()]


class TestASubscriptionNeverSilentlyFailsToExist:
    """Each way a subscription fails to exist reports itself.

    ``subscribe`` has three ``None`` returns. One - a raising
    ``declare_subscriber`` - has always logged a WARNING naming the topic, in
    line with the six other client-side refusals in
    :mod:`strands_robots.mesh.core`. The other two answered ``None`` with no
    record at all, and they are the pair a caller meets on the only rejoin the
    class offers: ``stop()`` drops every subscription this method records, so
    the re-subscribe that follows a ``start()`` is exactly where a caller needs
    to be told why nothing was declared.
    """

    def test_a_subscribe_before_start_says_why(self, caplog):
        """Not on the mesh is a refusal, so it names itself and the topic."""
        mesh = Mesh(_FakeRobot(), peer_id="quiet-peer")

        with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
            result = mesh.subscribe("strands/quiet/topic")

        assert result is None
        said = _refusals(caplog, "strands/quiet/topic")
        assert said, "subscribe answered None and said nothing"
        assert "not on the mesh" in said[0], said

    def test_a_subscribe_after_a_stop_says_why(self, caplog, monkeypatch):
        """The rejoin path: a subscription dropped by stop() cannot be replaced silently."""
        session = MagicMock()
        monkeypatch.setattr(mesh_core, "current_session", lambda: session)
        mesh = _running_mesh("rejoin-peer")
        attached = mesh.subscribe("strands/rejoin/topic", name="t")
        assert attached == "t", "premise: it subscribes while running"

        mesh.stop()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
            result = mesh.subscribe("strands/rejoin/topic", name="t")

        assert result is None
        said = _refusals(caplog, "strands/rejoin/topic")
        assert said, "a re-subscribe after stop() answered None and said nothing"
        assert "not on the mesh" in said[0], said

    def test_a_subscribe_without_a_session_says_why(self, caplog, monkeypatch):
        """No session is a refusal too, and it is distinguishable from the others."""
        monkeypatch.setattr(mesh_core, "current_session", lambda: None)
        mesh = _running_mesh("sessionless-peer")

        with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
            result = mesh.subscribe("strands/sessionless/topic")

        assert result is None
        said = _refusals(caplog, "strands/sessionless/topic")
        assert said, "subscribe answered None and said nothing"
        assert "no mesh session" in said[0], said

    def test_the_declare_failure_still_says_why(self, caplog, monkeypatch):
        """Control: the one path that always reported still reports, unchanged."""
        session = MagicMock()
        session.declare_subscriber.side_effect = RuntimeError("router unreachable")
        monkeypatch.setattr(mesh_core, "current_session", lambda: session)
        mesh = _running_mesh("declare-fail-peer")

        with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
            refused = mesh.subscribe("strands/declare/topic")
        assert refused is None

        said = _refusals(caplog, "strands/declare/topic")
        assert said, "the pre-existing declare_subscriber WARNING was lost"
        assert "router unreachable" in said[0], said

    def test_a_successful_subscribe_says_nothing_about_refusing(self, caplog, monkeypatch):
        """Control: the accepted path is not made noisy by the refusals above."""
        session = MagicMock()
        monkeypatch.setattr(mesh_core, "current_session", lambda: session)
        mesh = _running_mesh("accepted-peer")

        with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
            accepted = mesh.subscribe("strands/accepted/topic", name="ok")
        assert accepted == "ok"

        assert _refusals(caplog, "strands/accepted/topic") == []

    def test_stop_reports_the_subscriptions_it_dropped(self, caplog, monkeypatch):
        """A rejoining caller is told what it has to re-declare."""
        session = MagicMock()
        monkeypatch.setattr(mesh_core, "current_session", lambda: session)
        mesh = _running_mesh("dropping-peer")
        first = mesh.subscribe("strands/a", name="a")
        second = mesh.subscribe("strands/b", name="b")
        assert (first, second) == ("a", "b"), "premise: two subscriptions to drop"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
            mesh.stop()

        dropped = [r.getMessage() for r in caplog.records if "dropped" in r.getMessage()]
        assert dropped, "stop() discarded two subscriptions and said nothing"
        assert "2 subscription" in dropped[0], dropped
        assert "a" in dropped[0] and "b" in dropped[0], dropped
        assert mesh._user_subs == {}, "premise: they really are gone"

    def test_a_stop_with_no_subscriptions_says_nothing_about_dropping(self, caplog):
        """Control: ordinary teardown stays quiet, so the notice means something."""
        mesh = _running_mesh("undropping-peer")

        with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
            mesh.stop()

        assert [r.getMessage() for r in caplog.records if "dropped" in r.getMessage()] == []

    def test_a_stop_keeps_the_peer_id_and_the_lockout(self, monkeypatch):
        """Control: leaving the mesh is not a way to forget an e-stop.

        The lockout and the peer's identity are what a rejoin has to preserve,
        and both already survive ``stop()``. Pinned here so the reporting added
        beside them cannot be mistaken for - or grow into - a reset.
        """
        session = MagicMock()
        monkeypatch.setattr(mesh_core, "current_session", lambda: session)
        mesh = _running_mesh("latched-peer")
        mesh._estop_lockout.set()
        mesh._last_estop_ts = 12345.0

        mesh.stop()

        assert mesh.peer_id == "latched-peer"
        assert mesh._estop_lockout.is_set(), "stop() forgot an engaged e-stop"
        assert mesh._last_estop_ts == 12345.0

    def test_the_return_contract_names_every_refusal(self):
        """The docstring accounts for all three ways it answers None."""
        doc = Mesh.subscribe.__doc__ or ""
        assert "Returns:" in doc, "subscribe documents no return"
        for reason in ("not on the mesh", "no session", "declare_subscriber"):
            assert reason in doc, f"the {reason!r} refusal is undocumented: {doc!r}"
        assert "does not survive" in doc, "the stop() interaction is undocumented"
