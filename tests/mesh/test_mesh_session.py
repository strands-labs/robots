"""Tests for strands_robots.mesh.session — session singleton + peer registry.

All tests mock zenoh so no network or real zenoh install is required.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.session import (
    PeerInfo,
    clear_peers,
    get_peer,
    get_peers,
    peer_count,
    prune_peers,
    put,
    update_peer,
)

# ---------------------------------------------------------------------------
# PeerInfo dataclass
# ---------------------------------------------------------------------------


class TestPeerInfo:
    """PeerInfo stores discovery metadata and exposes age/to_dict."""

    def test_defaults(self) -> None:
        p = PeerInfo(peer_id="arm-1")
        assert p.peer_id == "arm-1"
        assert p.peer_type == "robot"
        assert p.hostname == ""
        assert p.caps == {}

    def test_age_increases(self) -> None:
        p = PeerInfo(peer_id="arm-1", last_seen=time.time() - 5.0)
        assert p.age >= 5.0

    def test_to_dict_includes_caps(self) -> None:
        p = PeerInfo(
            peer_id="g1",
            peer_type="sim",
            hostname="jetson-01",
            last_seen=time.time(),
            caps={"tool_name": "unitree_g1", "connected": True},
        )
        d = p.to_dict()
        assert d["peer_id"] == "g1"
        assert d["type"] == "sim"
        assert d["hostname"] == "jetson-01"
        assert d["tool_name"] == "unitree_g1"
        assert d["connected"] is True
        assert "age" in d

    def test_to_dict_age_is_rounded(self) -> None:
        p = PeerInfo(peer_id="x", last_seen=time.time() - 1.234)
        d = p.to_dict()
        # age is rounded to 1 decimal
        assert isinstance(d["age"], float)
        assert d["age"] == round(d["age"], 1)


# ---------------------------------------------------------------------------
# Peer registry
# ---------------------------------------------------------------------------


class TestPeerRegistry:
    """Peer registry: thread-safe upsert, prune, query."""

    @pytest.fixture(autouse=True)
    def _clean_peers(self) -> Iterator[None]:
        """Ensure a clean registry for every test."""
        clear_peers()
        yield
        clear_peers()

    def test_update_peer_new_returns_true(self) -> None:
        assert update_peer("arm-1", "robot", "host-a", {}) is True

    def test_update_peer_existing_returns_false(self) -> None:
        update_peer("arm-1", "robot", "host-a", {})
        assert update_peer("arm-1", "robot", "host-a", {}) is False

    def test_get_peers_returns_all(self) -> None:
        update_peer("arm-1", "robot", "h1", {"hw": "so100"})
        update_peer("arm-2", "sim", "h2", {})
        peers = get_peers()
        assert len(peers) == 2
        ids = {p["peer_id"] for p in peers}
        assert ids == {"arm-1", "arm-2"}

    def test_get_peer_found(self) -> None:
        update_peer("arm-1", "robot", "h1", {})
        p = get_peer("arm-1")
        assert p is not None
        assert p["peer_id"] == "arm-1"

    def test_get_peer_not_found(self) -> None:
        assert get_peer("nonexistent") is None

    def test_peer_count(self) -> None:
        assert peer_count() == 0
        update_peer("a", "robot", "", {})
        update_peer("b", "robot", "", {})
        assert peer_count() == 2

    def test_prune_removes_stale(self) -> None:
        update_peer("fresh", "robot", "", {})
        # Manually backdate one peer
        from strands_robots.mesh.session import _PEERS, _PEERS_LOCK

        with _PEERS_LOCK:
            _PEERS["stale"] = PeerInfo(peer_id="stale", last_seen=time.time() - 30)

        pruned = prune_peers(timeout=10.0)
        assert "stale" in pruned
        assert get_peer("stale") is None
        assert get_peer("fresh") is not None

    def test_prune_returns_empty_when_all_fresh(self) -> None:
        update_peer("a", "robot", "", {})
        pruned = prune_peers(timeout=10.0)
        assert pruned == []

    def test_clear_peers(self) -> None:
        update_peer("a", "robot", "", {})
        update_peer("b", "robot", "", {})
        clear_peers()
        assert peer_count() == 0

    def test_concurrent_updates(self) -> None:
        """Multiple threads updating peers simultaneously don't corrupt state."""
        errors: list[Exception] = []

        def worker(prefix: str) -> None:
            try:
                for i in range(50):
                    update_peer(f"{prefix}-{i}", "robot", "", {})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"t{n}",)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert peer_count() == 200  # 4 threads × 50 peers


# ---------------------------------------------------------------------------
# Session lifecycle (mocked zenoh)
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    """get_session / release_session with mocked zenoh."""

    @pytest.fixture(autouse=True)
    def _reset_session(self) -> Iterator[None]:
        """Reset module-level session state between tests."""
        import strands_robots.mesh.session as mod

        with mod._SESSION_LOCK:
            if mod._SESSION is not None:
                try:
                    mod._SESSION.close()
                except Exception:
                    pass
            mod._SESSION = None
            mod._SESSION_REFS = 0
        yield
        with mod._SESSION_LOCK:
            if mod._SESSION is not None:
                try:
                    mod._SESSION.close()
                except Exception:
                    pass
            mod._SESSION = None
            mod._SESSION_REFS = 0

    def test_returns_none_when_zenoh_missing(self) -> None:
        from strands_robots.mesh.session import get_session

        with patch.dict("sys.modules", {"zenoh": None}):
            with patch("builtins.__import__", side_effect=ImportError("no zenoh")):
                result = get_session()
        assert result is None

    def test_session_opened_as_listener(self) -> None:
        """First process should try to listen, succeeding makes it the router."""
        mock_zenoh = MagicMock()
        mock_session = MagicMock()
        mock_zenoh.open.return_value = mock_session
        mock_zenoh.Config.return_value = MagicMock()

        from strands_robots.mesh.session import get_session

        with patch.dict("sys.modules", {"zenoh": mock_zenoh}), patch.dict("os.environ", {}, clear=False):
            # Remove any env overrides that might interfere
            import os

            os.environ.pop("ZENOH_CONNECT", None)
            os.environ.pop("ZENOH_LISTEN", None)

            session = get_session()

        assert session is mock_session
        mock_zenoh.open.assert_called_once()

    def test_refcount_increments(self) -> None:
        """Second call to get_session increments refcount, doesn't re-open."""
        import strands_robots.mesh.session as mod

        mock_session = MagicMock()
        with mod._SESSION_LOCK:
            mod._SESSION = mock_session
            mod._SESSION_REFS = 1

        s = mod.get_session()
        assert s is mock_session
        assert mod._SESSION_REFS == 2

    def test_release_decrements(self) -> None:
        import strands_robots.mesh.session as mod

        mock_session = MagicMock()
        with mod._SESSION_LOCK:
            mod._SESSION = mock_session
            mod._SESSION_REFS = 2

        mod.release_session()
        assert mod._SESSION_REFS == 1
        assert mod._SESSION is mock_session  # still open

    def test_release_closes_at_zero(self) -> None:
        import strands_robots.mesh.session as mod

        mock_session = MagicMock()
        with mod._SESSION_LOCK:
            mod._SESSION = mock_session
            mod._SESSION_REFS = 1

        mod.release_session()
        assert mod._SESSION is None
        assert mod._SESSION_REFS == 0
        mock_session.close.assert_called_once()

    def test_release_noop_when_no_session(self) -> None:
        """release_session on an already-closed session doesn't crash."""
        import strands_robots.mesh.session as mod

        mod.release_session()  # should not raise
        assert mod._SESSION_REFS == 0

    def test_session_alive(self) -> None:
        import strands_robots.mesh.session as mod

        assert mod.session_alive() is False
        with mod._SESSION_LOCK:
            mod._SESSION = MagicMock()
            mod._SESSION_REFS = 1
        assert mod.session_alive() is True

    def test_listener_fallback_to_client(self) -> None:
        """If listen fails (port taken), should fall back to client mode."""
        import strands_robots.mesh.session as mod

        mock_zenoh = MagicMock()
        mock_session = MagicMock()

        call_count = 0

        def open_side_effect(cfg: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (listener) fails — port taken
                raise OSError("Address already in use")
            # Second call (client) succeeds
            return mock_session

        mock_zenoh.open.side_effect = open_side_effect
        mock_zenoh.Config.return_value = MagicMock()

        with patch.dict("sys.modules", {"zenoh": mock_zenoh}), patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("ZENOH_CONNECT", None)
            os.environ.pop("ZENOH_LISTEN", None)

            session = mod.get_session()

        assert session is mock_session
        assert mock_zenoh.open.call_count == 2


# ---------------------------------------------------------------------------
# put() helper
# ---------------------------------------------------------------------------


class TestPut:
    """put() publishes JSON or is a no-op when session is None."""

    @pytest.fixture(autouse=True)
    def _reset_session(self) -> Iterator[None]:
        import strands_robots.mesh.session as mod

        original = mod._SESSION
        yield
        with mod._SESSION_LOCK:
            mod._SESSION = original

    def test_put_noop_when_no_session(self) -> None:
        import strands_robots.mesh.session as mod

        with mod._SESSION_LOCK:
            mod._SESSION = None

        # Should not raise
        put("strands/test/presence", {"peer_id": "test"})

    def test_put_publishes_json(self) -> None:
        import strands_robots.mesh.session as mod

        mock_session = MagicMock()
        with mod._SESSION_LOCK:
            mod._SESSION = mock_session

        payload = {"peer_id": "arm-1", "t": 1234}
        put("strands/arm-1/presence", payload)

        mock_session.put.assert_called_once()
        call_args = mock_session.put.call_args
        assert call_args[0][0] == "strands/arm-1/presence"
        assert json.loads(call_args[0][1].decode()) == payload

    def test_put_swallows_exception(self) -> None:
        """put() logs but doesn't raise on publish failure."""
        import strands_robots.mesh.session as mod

        mock_session = MagicMock()
        mock_session.put.side_effect = RuntimeError("network down")
        with mod._SESSION_LOCK:
            mod._SESSION = mock_session

        # Should not raise
        put("strands/test/state", {"x": 1})


# ---------------------------------------------------------------------------
# Connection config from env vars
# ---------------------------------------------------------------------------


class TestConnectionConfig:
    """_build_config reads ZENOH_CONNECT and ZENOH_LISTEN from env."""

    def test_explicit_connect(self) -> None:
        mock_zenoh = MagicMock()
        mock_config = MagicMock()
        mock_zenoh.Config.return_value = mock_config

        with (
            patch.dict("sys.modules", {"zenoh": mock_zenoh}),
            patch.dict("os.environ", {"ZENOH_CONNECT": "tcp/10.0.0.1:7447,tcp/10.0.0.2:7447"}),
        ):
            from strands_robots.mesh.session import _build_config

            _build_config()

        mock_config.insert_json5.assert_any_call(
            "connect/endpoints",
            json.dumps(["tcp/10.0.0.1:7447", "tcp/10.0.0.2:7447"]),
        )

    def test_explicit_listen(self) -> None:
        mock_zenoh = MagicMock()
        mock_config = MagicMock()
        mock_zenoh.Config.return_value = mock_config

        with (
            patch.dict("sys.modules", {"zenoh": mock_zenoh}),
            patch.dict("os.environ", {"ZENOH_LISTEN": "tcp/0.0.0.0:7448"}),
        ):
            from strands_robots.mesh.session import _build_config

            _build_config()

        mock_config.insert_json5.assert_any_call(
            "listen/endpoints",
            json.dumps(["tcp/0.0.0.0:7448"]),
        )


# ---------------------------------------------------------------------------
# atexit cleanup
# ---------------------------------------------------------------------------


class TestAtexitCleanup:
    """_atexit_cleanup closes session without raising."""

    @pytest.fixture(autouse=True)
    def _reset_session(self) -> Iterator[None]:
        import strands_robots.mesh.session as mod

        with mod._SESSION_LOCK:
            mod._SESSION = None
            mod._SESSION_REFS = 0
        yield
        with mod._SESSION_LOCK:
            mod._SESSION = None
            mod._SESSION_REFS = 0

    def test_cleanup_closes_session(self) -> None:
        import strands_robots.mesh.session as mod

        mock_session = MagicMock()
        with mod._SESSION_LOCK:
            mod._SESSION = mock_session
            mod._SESSION_REFS = 3

        mod._atexit_cleanup()

        assert mod._SESSION is None
        assert mod._SESSION_REFS == 0
        mock_session.close.assert_called_once()

    def test_cleanup_noop_when_no_session(self) -> None:
        import strands_robots.mesh.session as mod

        with mod._SESSION_LOCK:
            mod._SESSION = None
            mod._SESSION_REFS = 0

        mod._atexit_cleanup()  # should not raise
