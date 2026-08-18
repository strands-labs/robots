"""``ZenohTransport`` reaches the raw Zenoh session, never the backend router.

:mod:`strands_robots.mesh.session` exposes each Zenoh operation twice: a public,
**backend-aware** entry point that resolves whatever ``STRANDS_MESH_BACKEND``
selects, and a private ``_*_directly`` helper that always takes the raw Zenoh
path. :class:`~strands_robots.mesh.transport.zenoh_transport.ZenohTransport`
must use the second kind for *every* delegation, because under
``STRANDS_MESH_BACKEND=bridge`` the backend-aware entry points resolve the
:class:`~strands_robots.mesh.transport.bridge_transport.BridgeTransport` that
owns that very ``ZenohTransport`` - so a backend-aware call routes straight back
into the caller.

Both re-entries are silent in production, which is why they need pinning here:

* ``put`` re-enters ``BridgeTransport.put`` until the stack is exhausted. The
  resulting ``RecursionError`` is a ``RuntimeError`` subclass, so the bridge's
  own narrow ``except (RuntimeError, ConnectionError, OSError)`` absorbs it and
  logs one DEBUG line - the publish reaches neither leg and ``put`` returns
  normally, exactly as a successful fire-and-forget publish does.
* ``close`` re-enters the transport factory's non-reentrant lock from the thread
  already holding it, so teardown never returns and the session is never closed.

Every existing bridge test injects a *fake* Zenoh leg whose ``put``/``close``
record instead of delegating, so no fixture in the suite could reach either
path. These tests drive a real ``ZenohTransport`` over a fake ``zenoh.Session``
object, which needs no broker and no ``zenoh`` wheel.
"""

from __future__ import annotations

import ast
import inspect
import json
import threading
from typing import Any, cast

import pytest

from strands_robots.mesh import session as session_mod
from strands_robots.mesh.transport import factory
from strands_robots.mesh.transport.bridge_transport import BridgeTransport
from strands_robots.mesh.transport.zenoh_transport import ZenohTransport

# A bridged topic under the bridge's exact-suffix match policy, so the IoT leg
# is genuinely reachable and "the WAN leg is untouched" is a real assertion.
BRIDGED_KEY = "strands/probe-peer/state"


class _FakeZenohSession:
    """Stands in for ``zenoh.Session``: records publishes, tracks close."""

    def __init__(self) -> None:
        self.payloads: list[tuple[str, bytes]] = []
        self.closed = False

    def put(self, key: str, payload: bytes) -> None:
        self.payloads.append((key, payload))

    def close(self) -> None:
        self.closed = True


class _StubIot:
    """A WAN leg that records, so the bridge's IoT half stays observable."""

    def __init__(self) -> None:
        self.payloads: list[str] = []

    def connect(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def is_alive(self) -> bool:
        return True

    def put(self, key: str, data: dict[str, Any]) -> None:
        self.payloads.append(key)

    def declare_subscriber(self, key_expr: str, handler: Any) -> Any:
        raise NotImplementedError

    @property
    def raw_session(self) -> Any | None:
        return None


class _CountingBridge(BridgeTransport):
    """Counts how many times one publish enters the bridge."""

    put_entries = 0

    def put(self, key: str, data: dict[str, Any]) -> None:
        self.put_entries += 1
        super().put(key, data)


@pytest.fixture
def fake_session(monkeypatch) -> _FakeZenohSession:
    """Install a fake open Zenoh session with one outstanding reference."""
    fake = _FakeZenohSession()
    monkeypatch.setattr(session_mod, "_SESSION", fake)
    monkeypatch.setattr(session_mod, "_SESSION_REFS", 1)
    return fake


@pytest.fixture
def bridge_backend(monkeypatch) -> None:
    """Select the bridge backend and give the factory throwaway state.

    ``_LOCK`` is replaced so that a pre-fix deadlock wedges a lock nothing
    outside this test will ever wait on.
    """
    monkeypatch.setenv("STRANDS_MESH_BACKEND", "bridge")
    monkeypatch.setattr(factory, "_LOCK", threading.Lock())
    monkeypatch.setattr(factory, "_TRANSPORT", None)
    monkeypatch.setattr(factory, "_TRANSPORT_REFS", 0)
    monkeypatch.setattr(factory, "_TRANSPORT_BACKEND", "")


def _mounted_bridge(iot: _StubIot) -> _CountingBridge:
    """A connected bridge over a real ``ZenohTransport``, installed as the
    factory singleton exactly as bridge mode does."""
    zenoh_leg = ZenohTransport()
    zenoh_leg._has_ref = True  # the fake session's outstanding reference
    # cast: the stub satisfies the MeshTransport protocol the bridge uses, not
    # the concrete IotMqttTransport its signature names (as in test_bridge_dedup).
    bridge = _CountingBridge(zenoh=zenoh_leg, iot=cast(Any, iot), bridge_suffixes=frozenset({"state"}))
    bridge._zenoh_alive = True
    bridge._iot_alive = True
    factory._TRANSPORT = bridge
    factory._TRANSPORT_REFS = 1
    factory._TRANSPORT_BACKEND = "bridge"
    return bridge


class TestPublishReachesTheWire:
    """A bridge publish must land on the Zenoh session, not on the bridge."""

    def test_a_bridged_publish_enters_the_bridge_once_and_reaches_the_session(self, fake_session, bridge_backend):
        """The headline: one publish, one bridge entry, one payload on the wire.

        Pre-fix the Zenoh leg published through the backend-aware
        ``session.put``, which resolved this same bridge, so the call re-entered
        the bridge until the stack ran out and delivered nothing to either leg
        while still returning normally.
        """
        iot = _StubIot()
        bridge = _mounted_bridge(iot)

        bridge.put(BRIDGED_KEY, {"v": 1})

        assert bridge.put_entries == 1, (
            f"one publish entered BridgeTransport.put {bridge.put_entries} times "
            f"and delivered {len(fake_session.payloads)} payload(s) to the Zenoh "
            "session: the Zenoh leg published through the backend-aware router, "
            "which resolves back to this bridge"
        )
        assert len(fake_session.payloads) == 1, (
            f"the publish reached the Zenoh session {len(fake_session.payloads)} "
            "time(s); a fire-and-forget put that returns normally must have "
            "published"
        )
        assert fake_session.payloads[0][0] == BRIDGED_KEY

    def test_the_iot_leg_receives_a_bridged_publish_exactly_once(self, fake_session, bridge_backend):
        """One logical publish must reach the WAN leg exactly once.

        ``IotMqttTransport.put`` talks to its own MQTT client, so the WAN leg
        was never the one routing back into the bridge - but it sits below the
        re-entry, so pre-fix each re-entry republished the same payload to MQTT.
        """
        iot = _StubIot()
        bridge = _mounted_bridge(iot)

        bridge.put(BRIDGED_KEY, {"v": 1})

        assert iot.payloads == [BRIDGED_KEY]

    def test_the_published_payload_is_the_json_the_legacy_path_encodes(self, fake_session, bridge_backend):
        """Bypassing the router must not change the bytes on the wire."""
        iot = _StubIot()
        bridge = _mounted_bridge(iot)

        bridge.put(BRIDGED_KEY, {"v": 1})

        assert len(fake_session.payloads) == 1
        _, payload = fake_session.payloads[0]
        assert isinstance(payload, bytes)
        assert json.loads(payload.decode()) == {"v": 1}


class TestTeardownCompletes:
    """Releasing the bridge must not re-enter the factory lock."""

    def test_releasing_the_last_reference_closes_the_session_and_returns(self, fake_session, bridge_backend):
        """The factory's last release must complete and clear the singleton.

        Pre-fix the Zenoh leg released through the backend-aware
        ``session.release_session``, which delegates back to
        ``factory.release_transport`` - re-entering the factory's non-reentrant
        lock from the thread already inside it, so teardown never returned.
        Run on a worker thread with a bounded join so a pre-fix hang is a
        failed assertion rather than a suite that stops making progress.
        """
        iot = _StubIot()
        bridge = _mounted_bridge(iot)
        assert bridge._zenoh._has_ref, "premise: the Zenoh leg holds a reference"

        finished = threading.Event()

        def teardown() -> None:
            factory.release_transport()
            finished.set()

        worker = threading.Thread(target=teardown, daemon=True)
        worker.start()
        completed = finished.wait(timeout=10.0)

        assert completed, (
            "factory.release_transport() did not return within 10s: closing the "
            "bridge released its Zenoh leg through the backend-aware "
            "session.release_session, which re-enters the factory lock this "
            "thread already holds"
        )
        assert factory._TRANSPORT is None
        assert fake_session.closed, "the last release must close the Zenoh session"
        assert not bridge._zenoh._has_ref


class TestTheLegacyZenohBackendIsUnchanged:
    """Controls: the default backend and the router itself must still work."""

    def test_the_transport_publishes_to_the_session_on_the_zenoh_backend(self, fake_session, monkeypatch):
        """With no bridge in play the transport still reaches the session."""
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "zenoh")
        transport = ZenohTransport()
        transport._has_ref = True

        transport.put("strands/probe-peer/state", {"v": 2})

        assert len(fake_session.payloads) == 1

    def test_session_put_still_routes_to_the_active_transport_under_bridge(self, fake_session, bridge_backend):
        """``session.put`` stays backend-aware.

        Fails if the recursion is "fixed" by making the public router publish
        raw Zenoh instead - callers that want backend routing must keep it.
        Asserts only that the router *reached* the transport, so it states the
        same thing on either side of the fix.
        """
        bridge = _mounted_bridge(_StubIot())

        session_mod.put(BRIDGED_KEY, {"v": 3})

        assert bridge.put_entries >= 1, "session.put must resolve the active transport"

    def test_release_session_still_delegates_to_the_factory_under_bridge(self, fake_session, bridge_backend):
        """``session.release_session`` stays backend-aware.

        Fails if the deadlock is "fixed" by dropping the backend branch, which
        would strand the transport singleton. Holds a second reference so the
        release decrements without closing, keeping this a statement about
        routing alone - and so it states the same thing on either side of the
        fix.
        """
        bridge = _mounted_bridge(_StubIot())
        factory._TRANSPORT_REFS = 2

        session_mod.release_session()

        assert factory._TRANSPORT_REFS == 1, "release_session must reach release_transport"
        assert factory._TRANSPORT is bridge


class TestEveryDelegationTakesTheRawPath:
    """The root cause: a delegation that names a backend-aware entry point.

    Structural, so a seventh delegation added later cannot reintroduce either
    re-entry without failing here.
    """

    def test_the_transport_imports_only_bypass_helpers_from_session(self):
        from strands_robots.mesh.transport import zenoh_transport

        tree = ast.parse(inspect.getsource(zenoh_transport))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "strands_robots.mesh.session":
                imported.extend(alias.name for alias in node.names)

        assert imported, "premise: the transport delegates to the session module"
        backend_aware = sorted(name for name in imported if not name.endswith("_directly"))
        assert not backend_aware, (
            f"ZenohTransport delegates to backend-aware session entry point(s) "
            f"{backend_aware}: under STRANDS_MESH_BACKEND=bridge those resolve the "
            "BridgeTransport that owns this transport, so the call routes back "
            "into the caller. Use the matching _*_directly helper."
        )
