"""Zenoh lifecycle error classification for the mesh session and node.

Best-effort Zenoh lifecycle operations (``open`` / ``declare_subscriber`` /
``undeclare`` / ``close``) must catch the errors real ``eclipse-zenoh`` raises
so a transport-side failure cannot escape partial-failure cleanup or
``Mesh.stop()``. The native ``zenoh.ZError`` subclasses ``Exception`` directly
- NOT ``RuntimeError`` - so it has to be named explicitly in the ``except``
tuple. :func:`strands_robots.mesh.session.zenoh_error_types` centralises that
tuple; these tests pin the contract and the fail-soft teardown behaviour on
both surfaces that consume it - ``Mesh.stop()``'s ``undeclare`` calls and the
session's own ``close`` calls in ``release_session`` / ``_atexit_cleanup``.
"""

from __future__ import annotations

import logging
import sys
import threading
from unittest.mock import MagicMock

import pytest

import strands_robots.mesh.session as session_mod
from strands_robots.mesh.core import Mesh


def test_error_tuple_always_includes_transport_builtins():
    types = session_mod.zenoh_error_types()
    assert isinstance(types, tuple)
    for builtin in (RuntimeError, OSError, ConnectionError):
        assert builtin in types
    # Programmer errors must NOT be swallowed by best-effort cleanup.
    assert TypeError not in types
    assert AttributeError not in types


def test_error_tuple_names_zenoh_zerror_explicitly():
    zenoh = pytest.importorskip("zenoh")
    zerr = getattr(zenoh, "ZError", None)
    if not isinstance(zerr, type):
        pytest.skip("zenoh.ZError is not a real class in this environment")
    # The whole reason ZError is named explicitly: it is an Exception subclass,
    # NOT a RuntimeError, so (RuntimeError, OSError) alone would let it escape.
    assert not issubclass(zerr, RuntimeError)
    assert zerr in session_mod.zenoh_error_types()


def _bare_mesh_for_stop(subs=None, safety_publishers=None):
    """A ``Mesh`` carrying only the attributes ``stop()`` touches.

    Built with ``Mesh.__new__`` (the shared mesh-unit-test pattern) so no zenoh
    transport or live robot is required.
    """
    mesh = Mesh.__new__(Mesh)
    mesh.peer_id = "test__arm"
    mesh._running = True
    mesh._lifecycle_lock = threading.RLock()
    mesh._stop_event = MagicMock()
    mesh._subs_lock = threading.RLock()
    mesh._subs = list(subs or [])
    mesh._user_subs = set()
    mesh._inbox_lock = threading.RLock()
    mesh.inbox = []
    mesh._safety_publishers_lock = threading.RLock()
    mesh._safety_publishers = dict(safety_publishers or {})
    mesh._rpc_lock = threading.RLock()
    mesh._pending = {}
    mesh._responses = {}
    mesh._has_session_ref = False
    return mesh


def _zerror(message: str) -> BaseException:
    """A real ``zenoh.ZError`` instance, or skip when zenoh is unavailable."""
    zenoh = pytest.importorskip("zenoh")
    zerr = getattr(zenoh, "ZError", None)
    if not isinstance(zerr, type):
        pytest.skip("zenoh.ZError is not a real class in this environment")
    return zerr(message)


def _zerror_raiser():
    """A publisher/subscriber whose ``undeclare`` raises a real ``zenoh.ZError``."""
    obj = MagicMock()
    obj.undeclare.side_effect = _zerror("simulated broker drop during teardown")
    return obj


def test_stop_is_fail_soft_when_safety_publisher_undeclare_raises_zerror():
    pub = _zerror_raiser()
    mesh = _bare_mesh_for_stop(safety_publishers={"estop": pub})

    mesh.stop()  # pre-fix: ZError escapes the (RuntimeError, OSError) catch

    pub.undeclare.assert_called_once()
    assert mesh.alive is False
    assert mesh._safety_publishers == {}


def test_stop_is_fail_soft_when_subscriber_undeclare_raises_zerror():
    sub = _zerror_raiser()
    mesh = _bare_mesh_for_stop(subs=[sub])

    mesh.stop()

    sub.undeclare.assert_called_once()
    assert mesh.alive is False
    assert mesh._subs == []


def test_stop_does_not_swallow_programmer_error_from_undeclare():
    # A TypeError is a bug, not a transport failure: it must surface rather than
    # be silently swallowed by best-effort teardown.
    sub = MagicMock()
    sub.undeclare.side_effect = TypeError("undeclare() takes no arguments")
    mesh = _bare_mesh_for_stop(subs=[sub])

    with pytest.raises(TypeError):
        mesh.stop()


# ---------------------------------------------------------------------------
# The session's own ``close`` paths.
#
# :func:`zenoh_error_types` names ``close`` among the operations it covers, and
# ``release_session`` / ``_atexit_cleanup`` are the only two ``close`` calls in
# the module that defines it. Both swallowed bare ``Exception`` and neither
# recorded anything, so a failed close was indistinguishable from a clean one -
# ``release_session``'s only visible line said the session had closed - and the
# programmer errors the tuple's docstring excludes were swallowed here too.
# ---------------------------------------------------------------------------

_SESSION_LOGGER = "strands_robots.mesh.session"

# The transport-failure surface the tuple documents. ``ZError`` is added by the
# fixture below when a real ``zenoh`` is importable.
_TRANSPORT_ERRORS = [RuntimeError, ConnectionError, OSError]

# Bugs, not transport faults: the tuple's docstring excludes these so they
# surface loudly instead of being swallowed by a best-effort teardown.
_PROGRAMMER_ERRORS = [TypeError, AttributeError]


@pytest.fixture
def failing_session():
    """Install a module session whose ``close`` raises, and reset after.

    Yields a factory taking the exception to raise; the caller drives
    ``release_session`` / ``_atexit_cleanup`` against it. The module globals are
    restored unconditionally so a propagating error cannot leak a session into
    the next test.
    """

    def install(exc: BaseException):
        session = MagicMock()
        session.close.side_effect = exc
        with session_mod._SESSION_LOCK:
            session_mod._SESSION = session
            session_mod._SESSION_REFS = 1
        return session

    try:
        yield install
    finally:
        with session_mod._SESSION_LOCK:
            session_mod._SESSION = None
            session_mod._SESSION_REFS = 0


@pytest.mark.parametrize("exc_type", _TRANSPORT_ERRORS, ids=lambda t: t.__name__)
def test_release_session_records_a_transport_close_failure(failing_session, caplog, exc_type):
    session = failing_session(exc_type("broker drop racing with close"))

    with caplog.at_level(logging.WARNING, logger=_SESSION_LOGGER):
        session_mod.release_session()  # fail-soft: a close fault is not the caller's problem

    session.close.assert_called_once()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "a close failure left no record at all"
    assert "close failed" in warnings[0].message
    assert "broker drop racing with close" in warnings[0].message
    # The reference is dropped either way, so nothing can retry the close.
    assert session_mod._SESSION is None
    assert session_mod._SESSION_REFS == 0


def test_release_session_records_a_zerror_close_failure(failing_session, caplog):
    # The whole reason the tuple names ZError: it subclasses Exception directly,
    # so (RuntimeError, OSError, ConnectionError) alone would let it escape.
    session = failing_session(_zerror("simulated broker drop during teardown"))

    with caplog.at_level(logging.WARNING, logger=_SESSION_LOGGER):
        session_mod.release_session()

    session.close.assert_called_once()
    assert any("close failed" in r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert session_mod._SESSION is None


def test_release_session_does_not_report_a_close_it_could_not_complete(failing_session, caplog):
    """A failed close must not still be reported as a clean one.

    This is what made the swallow silent rather than merely quiet: the success
    line was emitted unconditionally, so it was the only thing an operator saw
    and it contradicted what had happened.
    """
    failing_session(OSError("socket already closed"))

    with caplog.at_level(logging.DEBUG, logger=_SESSION_LOGGER):
        session_mod.release_session()

    assert not any("mesh session closed" in r.message for r in caplog.records)


def test_release_session_still_reports_a_clean_close(caplog):
    """The success line survives for the close that did complete."""
    session = MagicMock()
    with session_mod._SESSION_LOCK:
        session_mod._SESSION = session
        session_mod._SESSION_REFS = 1
    try:
        with caplog.at_level(logging.INFO, logger=_SESSION_LOGGER):
            session_mod.release_session()
    finally:
        with session_mod._SESSION_LOCK:
            session_mod._SESSION = None
            session_mod._SESSION_REFS = 0

    session.close.assert_called_once()
    assert any("mesh session closed" in r.message for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("exc_type", _PROGRAMMER_ERRORS, ids=lambda t: t.__name__)
def test_release_session_does_not_swallow_a_programmer_error_from_close(failing_session, exc_type):
    failing_session(exc_type("close() takes no arguments"))

    with pytest.raises(exc_type):
        session_mod.release_session()


@pytest.mark.parametrize("exc_type", _TRANSPORT_ERRORS, ids=lambda t: t.__name__)
def test_atexit_cleanup_records_a_transport_close_failure(failing_session, caplog, exc_type):
    session = failing_session(exc_type("broker drop racing with exit"))

    with caplog.at_level(logging.DEBUG, logger=_SESSION_LOGGER):
        session_mod._atexit_cleanup()

    session.close.assert_called_once()
    records = [r for r in caplog.records if "close failed at exit" in r.message]
    assert records, "the exit-time close failure left no record at all"
    # DEBUG: this path makes no success claim to contradict.
    assert records[0].levelno == logging.DEBUG
    assert session_mod._SESSION is None
    assert session_mod._SESSION_REFS == 0


def test_atexit_cleanup_records_a_zerror_close_failure(failing_session, caplog):
    session = failing_session(_zerror("simulated broker drop at exit"))

    with caplog.at_level(logging.DEBUG, logger=_SESSION_LOGGER):
        session_mod._atexit_cleanup()

    session.close.assert_called_once()
    assert any("close failed at exit" in r.message for r in caplog.records)


@pytest.mark.parametrize("exc_type", _PROGRAMMER_ERRORS, ids=lambda t: t.__name__)
def test_atexit_cleanup_does_not_swallow_a_programmer_error_from_close(failing_session, exc_type):
    failing_session(exc_type("close() takes no arguments"))

    with pytest.raises(exc_type):
        session_mod._atexit_cleanup()


def test_error_tuple_falls_back_to_the_builtins_without_zenoh(monkeypatch):
    """Without an importable ``zenoh`` the tuple is still usable.

    Both ``close`` sites evaluate the tuple when an exception arrives, so a
    partially provisioned install must still get a narrow, non-empty surface
    rather than an import error from the handler itself.
    """
    monkeypatch.setitem(sys.modules, "zenoh", None)  # makes ``import zenoh`` raise

    types = session_mod.zenoh_error_types()

    assert types == (RuntimeError, OSError, ConnectionError)
    assert all(issubclass(t, BaseException) for t in types)
