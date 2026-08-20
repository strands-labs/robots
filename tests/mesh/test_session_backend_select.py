"""Tests for session.py's transport backend delegation.

Verifies that ``STRANDS_MESH_BACKEND`` switches get_session/put/release_session/
current_session/session_alive to delegate to the transport factory, and that
the legacy zenoh path is byte-identical when the env var is unset/zenoh.
"""

from __future__ import annotations

import logging
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


class TestAnUnknownBackendIsReported:
    """A typo in ``STRANDS_MESH_BACKEND`` must reach the operator who made it.

    Two resolvers read the variable. ``session._backend_choice`` runs first, on
    every session and publish path, and its verdict is what
    ``_is_transport_backend()`` gates on; the factory's ``_select_backend`` runs
    only once that verdict is ``iot`` or ``bridge``. So an unknown value
    resolves to ``zenoh`` at the gate, the factory is never consulted, and a
    report emitted only there cannot be reached by the input class it describes:
    ``STRANDS_MESH_BACKEND=iott`` produced a plain Zenoh session
    indistinguishable from an explicit ``zenoh``, with nothing naming the
    variable. ``strands_robots.mesh.iot.provision`` writes
    ``STRANDS_MESH_BACKEND=iot`` into an operator's environment file, so a
    mistyped value is the ordinary way this happens, and the cost is a fleet
    talking over local Zenoh instead of the cloud broker it was pointed at.
    """

    #: Every spelling the vocabulary accepts, including the case and whitespace
    #: forms the resolver normalises. None of these may be reported.
    VALID = ("zenoh", "iot", "bridge", "IOT", " zenoh ", "Bridge")

    @staticmethod
    def _fresh_warn_once(monkeypatch):
        """Clear the process-wide warn-once set for this test.

        The guard is keyed by offending value so a second distinct typo is still
        news, but it outlives a test, so a sibling using the same value would
        otherwise silence this one.
        """
        from strands_robots.mesh import _backend_select

        # raising=False so this states a contract rather than an import shape:
        # the tests below assert on what an operator sees, and fail on that
        # rather than on the absence of the guard's own name.
        monkeypatch.setattr(_backend_select, "_UNKNOWN_WARNED", set(), raising=False)

    @staticmethod
    def _reports(caplog):
        """Captured messages that name the variable."""
        return [r.getMessage() for r in caplog.records if "STRANDS_MESH_BACKEND" in r.getMessage()]

    def test_a_typo_names_the_variable_the_value_and_the_vocabulary(self, monkeypatch, caplog):
        """The resolver on the live path reports an unknown value."""
        self._fresh_warn_once(monkeypatch)
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iott")

        with caplog.at_level(logging.WARNING):
            assert sess_mod._backend_choice() == "zenoh"

        reports = self._reports(caplog)
        assert reports, (
            "STRANDS_MESH_BACKEND=iott resolved to 'zenoh' and said nothing. The operator "
            "asked for a transport that does not exist and got the default, "
            "indistinguishable from an explicit STRANDS_MESH_BACKEND=zenoh."
        )
        message = reports[0]
        assert "iott" in message, f"the report does not name the offending value: {message!r}"
        for backend in ("zenoh", "iot", "bridge"):
            assert backend in message, f"the report does not name the valid value {backend!r}: {message!r}"

    def test_a_typo_is_reported_on_the_publish_gate_too(self, monkeypatch, caplog):
        """``_is_transport_backend()`` is what ``put()`` asks; it reports as well."""
        self._fresh_warn_once(monkeypatch)
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "brige")

        with caplog.at_level(logging.WARNING):
            assert sess_mod._is_transport_backend() is False

        reports = self._reports(caplog)
        assert reports, "the gate put() and get_session() consult resolved a typo silently"
        assert "brige" in reports[0]

    def test_the_factory_resolver_is_never_consulted_for_a_typo(self, monkeypatch):
        """Root cause: a report emitted only in the factory is unreachable.

        The factory is reached exclusively through ``_is_transport_backend()``
        being true, which an unknown value never makes it.
        """
        from strands_robots.mesh.transport import factory

        self._fresh_warn_once(monkeypatch)
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iott")

        calls: list[str] = []
        real_get_transport = factory.get_transport

        def _spy():
            calls.append("get_transport")
            return real_get_transport()

        monkeypatch.setattr(factory, "get_transport", _spy)

        sess_mod._is_transport_backend()

        assert calls == [], (
            "the transport factory was reached for an unknown backend, so this test's premise "
            "- that a report living only in the factory cannot fire - is stale"
        )

    def test_a_typo_is_reported_once_not_once_per_message(self, monkeypatch, caplog):
        """The gate runs per published message; the report must not."""
        self._fresh_warn_once(monkeypatch)
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iott")

        with caplog.at_level(logging.WARNING):
            for _ in range(500):  # 50 Hz telemetry for ten seconds
                sess_mod._is_transport_backend()

        reports = self._reports(caplog)
        assert len(reports) == 1, (
            f"{len(reports)} reports for 500 publish-path calls - a per-call report puts one "
            "line per telemetry sample in the operator's log"
        )

    def test_each_distinct_typo_gets_its_own_report(self, monkeypatch, caplog):
        """Warn-once is keyed by value: a second, different typo is news."""
        self._fresh_warn_once(monkeypatch)

        with caplog.at_level(logging.WARNING):
            monkeypatch.setenv("STRANDS_MESH_BACKEND", "iott")
            sess_mod._backend_choice()
            monkeypatch.setenv("STRANDS_MESH_BACKEND", "brige")
            sess_mod._backend_choice()

        reports = self._reports(caplog)
        assert len(reports) == 2, f"a second distinct typo was silenced: {reports}"
        assert any("iott" in r for r in reports)
        assert any("brige" in r for r in reports)

    def test_both_resolvers_share_one_owner(self, monkeypatch):
        """The vocabulary and the report have one owner, so they cannot drift.

        This is the shape the previous docstring claimed ("matches
        strands_robots.mesh.transport.factory") while re-reading the variable
        independently, which is how one side came to report an unknown value and
        the other not to.
        """
        from strands_robots.mesh import _backend_select

        self._fresh_warn_once(monkeypatch)
        calls: list[str] = []
        real_select = _backend_select.select_backend

        def _spy() -> str:
            calls.append("select_backend")
            return real_select()

        monkeypatch.setattr(sess_mod, "select_backend", _spy)
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")

        assert sess_mod._backend_choice() == "iot"
        assert calls == ["select_backend"], (
            "session._backend_choice resolved the variable itself instead of asking the owner, "
            "so the two vocabularies can diverge again"
        )

    def test_the_resolver_survives_a_world_where_every_import_raises(self, monkeypatch, caplog):
        """The gate must not reach for an import while resolving.

        ``get_session`` is documented to return ``None`` when zenoh is absent
        rather than propagate, and
        ``test_mesh_session.py::TestSessionLifecycle::test_returns_none_when_zenoh_missing``
        pins that by making every import raise. The owner is therefore bound at
        module import time: a call-time import here would turn that documented
        quiet degradation into an ``ImportError`` out of ``get_session``.
        """
        self._fresh_warn_once(monkeypatch)
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "iott")

        with caplog.at_level(logging.WARNING), patch("builtins.__import__", side_effect=ImportError("no zenoh")):
            assert sess_mod._backend_choice() == "zenoh"
            assert sess_mod._is_transport_backend() is False

        assert self._reports(caplog), "the report itself needed an import to be emitted"

    # -- controls: an accepted value stays silent, and every value resolves as before --

    @pytest.mark.parametrize("value", VALID)
    def test_an_accepted_value_is_never_reported(self, monkeypatch, caplog, value):
        """Only an unknown value is news."""
        self._fresh_warn_once(monkeypatch)
        monkeypatch.setenv("STRANDS_MESH_BACKEND", value)

        with caplog.at_level(logging.WARNING):
            sess_mod._backend_choice()
            sess_mod._is_transport_backend()

        reports = self._reports(caplog)
        assert reports == [], f"STRANDS_MESH_BACKEND={value!r} is valid and was reported: {reports}"

    def test_an_unset_variable_is_never_reported(self, monkeypatch, caplog):
        """The default is not a typo."""
        self._fresh_warn_once(monkeypatch)
        monkeypatch.delenv("STRANDS_MESH_BACKEND", raising=False)

        with caplog.at_level(logging.WARNING):
            assert sess_mod._backend_choice() == "zenoh"

        assert self._reports(caplog) == []

    @pytest.mark.parametrize(
        ("value", "expected", "transport"),
        [
            ("iott", "zenoh", False),
            ("garbage", "zenoh", False),
            ("", "zenoh", False),
            ("zenoh", "zenoh", False),
            ("iot", "iot", True),
            ("bridge", "bridge", True),
            ("IOT", "iot", True),
            (" zenoh ", "zenoh", False),
        ],
    )
    def test_the_resolved_backend_is_unchanged(self, monkeypatch, value, expected, transport):
        """Reporting a typo must not change which backend any value selects."""
        self._fresh_warn_once(monkeypatch)
        monkeypatch.setenv("STRANDS_MESH_BACKEND", value)

        assert sess_mod._backend_choice() == expected
        assert sess_mod._is_transport_backend() is transport
