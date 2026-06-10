"""Tests for session.py's transport backend delegation.

Verifies that ``STRANDS_MESH_BACKEND`` switches get_session/put/release_session/
current_session/session_alive to delegate to the transport factory, and that
the legacy zenoh path is byte-identical when the env var is unset/zenoh.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh import session as sess_mod


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset both session.py state AND the transport factory between tests."""
    from strands_robots.mesh.transport import factory

    with sess_mod._SESSION_LOCK:
        sess_mod._SESSION = None
        sess_mod._SESSION_REFS = 0
    with factory._LOCK:
        if factory._TRANSPORT is not None:
            try:
                factory._TRANSPORT.close()
            except Exception:
                pass
        factory._TRANSPORT = None
        factory._TRANSPORT_REFS = 0
        factory._TRANSPORT_BACKEND = ""
    yield
    with sess_mod._SESSION_LOCK:
        sess_mod._SESSION = None
        sess_mod._SESSION_REFS = 0
    with factory._LOCK:
        factory._TRANSPORT = None
        factory._TRANSPORT_REFS = 0
        factory._TRANSPORT_BACKEND = ""


class TestBackendChoice:
    def test_default_is_zenoh(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_BACKEND", raising=False)
        assert sess_mod._backend_choice() == "zenoh"
        assert sess_mod._is_transport_backend() is False

    def test_iot_is_transport_backend(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        assert sess_mod._backend_choice() == "iot"
        assert sess_mod._is_transport_backend() is True

    def test_bridge_is_transport_backend(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "bridge")
        assert sess_mod._is_transport_backend() is True

    def test_unknown_falls_back_to_zenoh(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "garbage")
        assert sess_mod._backend_choice() == "zenoh"


class TestPutDelegation:
    def test_zenoh_path_uses_session_directly(self, monkeypatch):
        """No env var → put encodes JSON and writes to _SESSION."""
        monkeypatch.delenv("STRANDS_MESH_BACKEND", raising=False)
        mock_session = MagicMock()
        sess_mod._SESSION = mock_session
        sess_mod.put("strands/test", {"k": 1})
        mock_session.put.assert_called_once()
        topic, payload = mock_session.put.call_args.args
        assert topic == "strands/test"
        # JSON-encoded bytes
        import json

        assert json.loads(payload.decode()) == {"k": 1}

    def test_iot_path_delegates_to_transport(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        mock_transport = MagicMock()
        with patch(
            "strands_robots.mesh.transport.factory.current_transport",
            return_value=mock_transport,
        ):
            sess_mod.put("strands/test", {"k": 1})
        mock_transport.put.assert_called_once_with("strands/test", {"k": 1})

    def test_iot_path_no_op_when_no_transport(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        with patch(
            "strands_robots.mesh.transport.factory.current_transport",
            return_value=None,
        ):
            # No exception
            sess_mod.put("strands/test", {"k": 1})

    def test_iot_swallows_put_errors(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        mock_transport = MagicMock()
        mock_transport.put.side_effect = RuntimeError("network")
        with patch(
            "strands_robots.mesh.transport.factory.current_transport",
            return_value=mock_transport,
        ):
            # No exception
            sess_mod.put("strands/test", {"k": 1})


class TestGetSessionDelegation:
    def test_zenoh_path_uses_legacy_session(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_BACKEND", raising=False)
        # Pre-seed legacy session so we don't need real zenoh.
        mock = MagicMock()
        sess_mod._SESSION = mock
        sess_mod._SESSION_REFS = 1
        result = sess_mod.get_session()
        assert result is mock
        assert sess_mod._SESSION_REFS == 2

    def test_iot_path_delegates_to_factory_get_transport(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        mock_transport = MagicMock()
        with patch(
            "strands_robots.mesh.transport.factory.get_transport",
            return_value=mock_transport,
        ):
            assert sess_mod.get_session() is mock_transport
        # Legacy refcount untouched.
        assert sess_mod._SESSION_REFS == 0


class TestReleaseSessionDelegation:
    def test_iot_path_delegates_to_factory_release(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        with patch("strands_robots.mesh.transport.factory.release_transport") as mock_release:
            sess_mod.release_session()
            mock_release.assert_called_once()


class TestCurrentSessionDelegation:
    def test_iot_returns_factory_current(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        sentinel = MagicMock()
        with patch(
            "strands_robots.mesh.transport.factory.current_transport",
            return_value=sentinel,
        ):
            assert sess_mod.current_session() is sentinel

    def test_zenoh_returns_session(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_BACKEND", raising=False)
        sess_mod._SESSION = "zenoh-session"
        assert sess_mod.current_session() == "zenoh-session"


class TestSessionAliveDelegation:
    def test_iot_alive_when_transport_is_alive(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        t = MagicMock()
        t.is_alive.return_value = True
        with patch(
            "strands_robots.mesh.transport.factory.current_transport",
            return_value=t,
        ):
            assert sess_mod.session_alive() is True

    def test_iot_dead_when_transport_dead(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        t = MagicMock()
        t.is_alive.return_value = False
        with patch(
            "strands_robots.mesh.transport.factory.current_transport",
            return_value=t,
        ):
            assert sess_mod.session_alive() is False

    def test_iot_dead_when_no_transport(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
        with patch(
            "strands_robots.mesh.transport.factory.current_transport",
            return_value=None,
        ):
            assert sess_mod.session_alive() is False
