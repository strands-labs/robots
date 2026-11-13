"""``Redial.poll`` reads the thread's liveness before its result, so a session stored as the thread exits is adopted.

Forced interleaving: the lock poll() takes is instrumented so that, right after poll() has read an empty result,
the redial thread stores its session (or refusal) and exits before poll() looks at liveness. Read in the other
order, that successful reconnect came back as "no server came back within reconnect_s".
"""

from __future__ import annotations

import threading

import pytest

from strands_robots.policy.base import PolicyFault
from strands_robots.policy.lerobot.redial import Redial


class _StoreRightAfterPollReads:
    """A lock that, on poll()'s first release, lets the redial thread finish and waits for it to exit."""

    def __init__(self, go: threading.Event) -> None:
        self._real, self._go = threading.Lock(), go
        self.redial: Redial | None = None
        self._fired = False

    def __enter__(self) -> None:
        self._real.acquire()

    def __exit__(self, *exc: object) -> None:
        self._real.release()
        if threading.current_thread() is threading.main_thread() and not self._fired and self.redial is not None:
            self._fired = True
            self._go.set()
            self.redial._thread.join(5.0)


def _redial(outcome) -> tuple[Redial, threading.Event]:
    go = threading.Event()

    def open_fn():
        assert go.wait(5.0)
        return outcome()

    r = Redial(open_fn, window_s=10.0, who="test")
    lock = _StoreRightAfterPollReads(go)
    r._lock, lock.redial = lock, r  # the thread is blocked in open_fn, so it has not taken the lock yet
    return r, go


def test_a_session_stored_as_the_thread_exits_is_adopted_not_reported_as_a_timeout():
    session = object()
    r, _ = _redial(lambda: session)
    assert r.poll() is None  # alive when looked at; the session it stores right after is adopted next tick
    assert r.poll() is session


def test_a_refusal_stored_as_the_thread_exits_surfaces_as_itself():
    def refuse():
        raise PolicyFault("test: a different deployment answered")

    r, _ = _redial(refuse)
    assert r.poll() is None
    with pytest.raises(PolicyFault, match="different deployment"):
        r.poll()
