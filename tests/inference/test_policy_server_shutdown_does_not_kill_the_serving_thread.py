"""Stopping a :class:`PolicyServer` must not kill the thread that serves it.

:meth:`PolicyServer.start` runs the websockets sync server's accept loop on a
background thread; :meth:`PolicyServer.stop` ends it by closing the listening
socket from the caller's thread. That socket is exactly what the loop is waiting
on, and the sync server does not synchronize the two, so a stop that lands while
the loop is coming up raises straight out of ``serve_forever()``.

Both shapes come out of the *same* call - ``serve_forever()`` registering the
listening socket with its selector - and which one surfaces is decided by where
in that call the close lands rather than by the release:

* the close lands before ``selectors`` reads ``fileno()``, so the descriptor is
  ``-1`` -> ``ValueError: Invalid file descriptor: -1``;
* the close lands between that read and ``epoll_ctl``, so the descriptor still
  looks live -> ``OSError: [Errno 9] Bad file descriptor``.

The release decides only which of the two is already handled upstream. 12.0
wraps the call in nothing and lets both escape (30 ``ValueError`` and 3
``OSError`` over 200 contended cycles); 13.0 through 17.x wrap it in
``except ValueError: return``, so only the ``OSError`` escapes (0 and 2 over the
same 200). Raising the dependency floor therefore narrows this race and cannot
close it, which is why the fix lives here.

Either way the serving thread dies, and nothing reports it: the thread is a
daemon, ``stop()`` returns normally, ``_server`` is cleared, and the server looks
cleanly shut down. That is why a lifecycle test asserting on the server's own
state - ``test_stop_is_idempotent`` in ``test_policy_server_lifecycle.py`` - was
green while the thread it started was crashing, and why the tests here assert on
``threading.excepthook`` instead.

The fix keys on a stop being in progress rather than on the exception type, so
these tests pin both halves of that: a teardown failure is absorbed, and the same
failure with no stop pending still propagates.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from strands_robots.inference import PolicyServer

#: The two failure shapes one registration call can raise, keyed by the window
#: that produces each. Every supported release can raise the ``OSError``; only
#: 12.0 lets the ``ValueError`` escape, so these are windows and not versions.
_TEARDOWN_FAILURES = [
    pytest.param(ValueError("Invalid file descriptor: -1"), id="closed-before-fileno-ValueError"),
    pytest.param(OSError(9, "Bad file descriptor"), id="closed-before-epoll_ctl-OSError"),
]

#: A real start/stop pair loses the race often but not always, so the end-to-end
#: guard repeats it. Measured 9 of 20 on websockets 12.0 and 3 of 12 on 17.0.1.
_CYCLES = 12


class _FakeSocket:
    """Minimal stand-in for the listening socket ``start()`` reads the port from."""

    def getsockname(self) -> tuple[str, int]:
        return ("127.0.0.1", 54321)


class _RacingServer:
    """A sync server whose accept loop always loses the shutdown race.

    Reproduces the window deterministically instead of hoping the scheduler
    lands in it: ``serve_forever`` blocks until ``shutdown`` closes the socket
    and then raises, which is what the real server does when a stop closes the
    socket underneath the registration that is about to hand it to the selector.
    """

    def __init__(self, failure: BaseException) -> None:
        self.socket = _FakeSocket()
        self._failure = failure
        self._closed = threading.Event()
        self.serve_forever_calls = 0

    def serve_forever(self) -> None:
        self.serve_forever_calls += 1
        self._closed.wait(timeout=5.0)
        raise self._failure

    def shutdown(self) -> None:
        self._closed.set()


def _collect_thread_exceptions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Capture every exception that escapes a thread for the test's duration.

    Replacing the hook also takes it off pytest's own thread-exception plugin,
    which only *warns*; that warning is precisely how this failure stayed
    invisible.
    """
    escaped: list[tuple[str, str]] = []

    def record(args: Any) -> None:
        escaped.append((args.exc_type.__name__, str(args.exc_value)))

    monkeypatch.setattr(threading, "excepthook", record)
    return escaped


def _install_racing_server(monkeypatch: pytest.MonkeyPatch, failure: BaseException) -> _RacingServer:
    """Make ``start()`` build a server whose accept loop loses the race.

    ``start()`` imports ``serve`` from ``websockets.sync.server`` at call time,
    so patching the attribute on that module is what the production import sees.
    """
    websockets_sync_server = pytest.importorskip("websockets.sync.server")
    racing = _RacingServer(failure)
    monkeypatch.setattr(websockets_sync_server, "serve", lambda *args, **kwargs: racing, raising=True)
    return racing


@pytest.mark.parametrize("failure", _TEARDOWN_FAILURES)
def test_a_teardown_failure_does_not_escape_the_serving_thread(
    failure: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop that closes the socket under the accept loop must be absorbed."""
    escaped = _collect_thread_exceptions(monkeypatch)
    racing = _install_racing_server(monkeypatch, failure)

    server = PolicyServer(policy_provider="mock", port=0)
    server.start()
    server.stop()

    assert racing.serve_forever_calls == 1, "the accept loop never ran, so nothing was tested"
    assert not escaped, f"stopping the server killed its serving thread: {escaped}"


@pytest.mark.parametrize("failure", _TEARDOWN_FAILURES)
def test_the_same_failure_without_a_stop_pending_still_propagates(
    failure: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A socket failure that is not a teardown is a real one and must surface.

    This is the boundary: absorbing the exception whenever it matched a type
    would hide a listening socket that failed on its own. Here the loop raises
    with no stop in progress, so it has to reach the thread hook.
    """
    escaped = _collect_thread_exceptions(monkeypatch)
    racing = _install_racing_server(monkeypatch, failure)

    server = PolicyServer(policy_provider="mock", port=0)
    server.start()
    racing.shutdown()  # release the loop WITHOUT going through stop()
    assert server._thread is not None
    server._thread.join(timeout=5.0)

    assert [name for name, _ in escaped] == [type(failure).__name__], (
        f"a failure with no stop pending must reach the thread hook, got {escaped}"
    )
    server.stop()


def test_a_stopped_server_can_be_started_again_and_still_absorbs_a_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stop marker must not latch, or the second run swallows real failures.

    ``stop()`` sets the marker and nothing else clears it, so a restarted server
    would treat any socket failure as a teardown. Restarting is a supported
    sequence - ``stop()`` documents itself as safe to call twice and ``start()``
    only refuses while a server is running.
    """
    escaped = _collect_thread_exceptions(monkeypatch)

    first = _install_racing_server(monkeypatch, OSError(9, "Bad file descriptor"))
    server = PolicyServer(policy_provider="mock", port=0)
    server.start()
    server.stop()
    assert first.serve_forever_calls == 1
    assert not escaped

    second = _install_racing_server(monkeypatch, OSError(9, "Bad file descriptor"))
    server.start()
    second.shutdown()  # a real failure this time, no stop pending
    assert server._thread is not None
    server._thread.join(timeout=5.0)

    assert [name for name, _ in escaped] == ["OSError"], (
        f"after a restart a genuine socket failure must still surface, got {escaped}"
    )
    server.stop()


def test_repeated_real_start_stop_cycles_leave_no_thread_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same guarantee end to end, against the real websockets server.

    Nothing is stubbed here: this binds and tears down a real listening socket
    the number of times needed to lose the race in practice.
    """
    pytest.importorskip("websockets.sync.server")
    escaped = _collect_thread_exceptions(monkeypatch)

    for _ in range(_CYCLES):
        server = PolicyServer(policy_provider="mock", port=0)
        server.start()
        assert server.port > 0
        server.stop()

    assert not escaped, f"{len(escaped)} of {_CYCLES} start/stop cycles killed the serving thread: {escaped}"
