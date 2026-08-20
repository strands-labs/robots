"""Unit tests for the transport factory (process-wide singleton).

Exercises STRANDS_MESH_BACKEND selection without requiring real Zenoh or AWS.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from strands_robots.mesh._backend_select import select_backend
from strands_robots.mesh.transport import factory
from strands_robots.mesh.transport.iot_transport import IotMqttTransport
from strands_robots.mesh.transport.zenoh_transport import ZenohTransport


@pytest.fixture(autouse=True)
def reset_factory_singleton():
    """Reset the module-level singleton between tests so cross-test state
    doesn't leak through ``factory._TRANSPORT``."""
    with factory._LOCK:
        factory._TRANSPORT = None
        factory._TRANSPORT_REFS = 0
        factory._TRANSPORT_BACKEND = ""
    yield
    with factory._LOCK:
        if factory._TRANSPORT is not None:
            try:
                factory._TRANSPORT.close()
            except Exception:
                pass
        factory._TRANSPORT = None
        factory._TRANSPORT_REFS = 0
        factory._TRANSPORT_BACKEND = ""


class TestBackendSelection:
    """``select_backend`` resolves the env var safely.

    The vocabulary lives in :mod:`strands_robots.mesh._backend_select` so this
    factory and the session gate in front of it cannot disagree about it; the
    factory asks it in :func:`~strands_robots.mesh.transport.factory.get_transport`.
    """

    def test_default_is_zenoh(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_BACKEND", raising=False)
        assert select_backend() == "zenoh"

    def test_explicit_iot(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        assert select_backend() == "iot"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "IOT")
        assert select_backend() == "iot"

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", " zenoh ")
        assert select_backend() == "zenoh"

    def test_unknown_falls_back_to_zenoh(self, monkeypatch, caplog):
        """Typos must NOT crash the host; warn and default to zenoh.

        The vocabulary and the report live in
        :mod:`strands_robots.mesh._backend_select`, which both this factory and
        the session gate in front of it ask, so this asserts the report names
        the offending value rather than which module emitted it.
        """
        import logging

        from strands_robots.mesh import _backend_select

        monkeypatch.setattr(_backend_select, "_UNKNOWN_WARNED", set())
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "unknownXYZ")
        with caplog.at_level(logging.WARNING):
            assert _backend_select.select_backend() == "zenoh"
            # The warning message includes the typo'd value
            assert any(
                "unknownxyz" in r.getMessage() for r in caplog.records if "STRANDS_MESH_BACKEND" in r.getMessage()
            )


class TestRefCounting:
    """``get_transport`` / ``release_transport`` are ref-counted."""

    def test_first_call_constructs_zenoh_when_connect_fails_returns_none(self, monkeypatch):
        """If the constructed transport's connect() returns False, get_transport
        returns None and does NOT install a singleton."""
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "zenoh")
        with patch.object(ZenohTransport, "connect", return_value=False):
            result = factory.get_transport()
            assert result is None
            assert factory._TRANSPORT is None
            assert factory._TRANSPORT_REFS == 0

    def test_second_call_reuses_singleton(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "zenoh")
        with patch.object(ZenohTransport, "connect", return_value=True):
            with patch.object(ZenohTransport, "is_alive", return_value=True):
                t1 = factory.get_transport()
                t2 = factory.get_transport()
                assert t1 is t2
                assert factory._TRANSPORT_REFS == 2

    def test_release_decrements_then_closes(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "zenoh")
        with patch.object(ZenohTransport, "connect", return_value=True):
            with patch.object(ZenohTransport, "is_alive", return_value=True):
                with patch.object(ZenohTransport, "close") as mock_close:
                    factory.get_transport()
                    factory.get_transport()
                    assert factory._TRANSPORT_REFS == 2

                    factory.release_transport()
                    assert factory._TRANSPORT_REFS == 1
                    mock_close.assert_not_called()

                    factory.release_transport()
                    assert factory._TRANSPORT_REFS == 0
                    mock_close.assert_called_once()
                    assert factory._TRANSPORT is None
                    assert factory._TRANSPORT_BACKEND == ""

    def test_release_below_zero_is_safe(self):
        """Calling release more times than acquired must NOT crash."""
        factory.release_transport()  # no-op, no exception
        factory.release_transport()
        assert factory._TRANSPORT_REFS == 0

    def test_current_transport_does_not_bump_refcount(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "zenoh")
        with patch.object(ZenohTransport, "connect", return_value=True):
            with patch.object(ZenohTransport, "is_alive", return_value=True):
                factory.get_transport()
                assert factory._TRANSPORT_REFS == 1
                t = factory.current_transport()
                assert t is not None
                assert factory._TRANSPORT_REFS == 1

    def test_current_backend_after_init(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "zenoh")
        with patch.object(ZenohTransport, "connect", return_value=True):
            assert factory.current_backend() == ""
            factory.get_transport()
            assert factory.current_backend() == "zenoh"
            factory.release_transport()
            assert factory.current_backend() == ""


class TestIotBackendSelection:
    """When STRANDS_MESH_BACKEND=iot, the factory builds IotMqttTransport."""

    def test_iot_backend_constructs_iot_transport(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        with patch.object(IotMqttTransport, "connect", return_value=True):
            with patch.object(IotMqttTransport, "is_alive", return_value=True):
                t = factory.get_transport()
                assert t is not None
                assert isinstance(t, IotMqttTransport)
                assert factory.current_backend() == "iot"

    def test_iot_connect_failure_returns_none(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        with patch.object(IotMqttTransport, "connect", return_value=False):
            assert factory.get_transport() is None
            assert factory._TRANSPORT is None


class TestThreadSafety:
    """Multiple threads calling get_transport concurrently must agree on
    the singleton."""

    def test_concurrent_get_returns_same_instance(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "zenoh")
        results: list = []
        with patch.object(ZenohTransport, "connect", return_value=True):
            with patch.object(ZenohTransport, "is_alive", return_value=True):
                threads = [threading.Thread(target=lambda: results.append(factory.get_transport())) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                assert len(results) == 8
                assert all(r is results[0] for r in results)
                assert factory._TRANSPORT_REFS == 8


class TestBridgeBackendSelection:
    """When STRANDS_MESH_BACKEND=bridge, the factory builds a BridgeTransport.

    ``bridge`` is the third documented backend (Zenoh + IoT together). Its
    factory-level construction path was previously unexercised, so a regression
    that dropped it from ``_construct`` (or from ``_select_backend``'s allow
    list) would go unnoticed while the two other backends stayed green.
    """

    def test_bridge_backend_constructs_bridge_transport(self, monkeypatch):
        from strands_robots.mesh.transport.bridge_transport import BridgeTransport

        monkeypatch.setenv("STRANDS_MESH_BACKEND", "bridge")
        with patch.object(BridgeTransport, "connect", return_value=True):
            with patch.object(BridgeTransport, "is_alive", return_value=True):
                t = factory.get_transport()
                assert t is not None
                assert isinstance(t, BridgeTransport)
                assert factory.current_backend() == "bridge"

    def test_bridge_connect_failure_returns_none(self, monkeypatch):
        from strands_robots.mesh.transport.bridge_transport import BridgeTransport

        monkeypatch.setenv("STRANDS_MESH_BACKEND", "bridge")
        with patch.object(BridgeTransport, "connect", return_value=False):
            assert factory.get_transport() is None
            assert factory._TRANSPORT is None
            assert factory._TRANSPORT_REFS == 0


class TestReleaseIsResilientToRaisingClose:
    """Releasing the final reference must reset the singleton even when the
    transport's ``close()`` raises.

    ``release_transport`` runs during host teardown; a transport whose
    ``close()`` throws (e.g. a Zenoh session already torn down by a signal
    handler) must NOT propagate that exception or leave a stale singleton
    installed. The factory swallows the error and clears its state so the next
    ``get_transport`` starts clean.
    """

    def test_raising_close_is_swallowed_and_singleton_cleared(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "zenoh")
        with patch.object(ZenohTransport, "connect", return_value=True):
            with patch.object(ZenohTransport, "is_alive", return_value=True):
                with patch.object(ZenohTransport, "close", side_effect=RuntimeError("session already gone")):
                    factory.get_transport()
                    assert factory._TRANSPORT_REFS == 1

                    # Must not raise despite close() throwing on the last release.
                    factory.release_transport()

                    assert factory._TRANSPORT is None
                    assert factory._TRANSPORT_REFS == 0
                    assert factory._TRANSPORT_BACKEND == ""
