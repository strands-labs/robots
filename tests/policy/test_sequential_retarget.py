"""``Sequential.retarget``: the old instruction's plan is never played under the new one; one infer at a time."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lerobot.remote_inference")

from lerobot.remote_inference import RequestCancelled  # noqa: E402

from strands_robots.policy.lerobot.sequential import Sequential  # noqa: E402


class _Contract:
    def to_keys_chunk(self, chunk):
        return np.asarray(chunk, dtype=float).reshape(-1, 2)


class _Chunk:
    def __init__(self, rows):
        self.canonical_actions = SimpleNamespace(
            detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: rows))
        )
        self.server_durations = {}


class _Client:
    """One ``infer`` at a time, like ``RemoteClient``; each call blocks until the test releases it or cancels it."""

    def __init__(self):
        self.descriptor = {"limits": {"max_input_chars": 200}}
        self.config = SimpleNamespace(deployment="d")
        self.requests: list = []
        self.gate = threading.Event()
        self.started = threading.Semaphore(0)
        self.inside, self.max_inside, self.cancelled = 0, 0, 0
        self.present = True  # RemoteClient.present: the server's liveliness token, latched False once lost
        self._lock = threading.Lock()

    def infer(self, req, *, cancelled=lambda: False):
        with self._lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        self.requests.append(req)
        self.started.release()
        try:
            while not self.gate.wait(0.005):
                if cancelled():
                    self.cancelled += 1
                    raise RequestCancelled("superseded")
            return _Chunk(np.full((3, 2), float(len(self.requests))))  # rows carry the request number
        finally:
            with self._lock:
                self.inside -= 1


def _drain(seq: Sequential) -> None:
    t = seq._inflight
    if t is not None:
        t.join(2.0)
        assert not t.is_alive()


def test_a_reply_that_landed_before_the_retarget_is_dropped_not_played_under_the_new_task():
    client = _Client()
    seq = Sequential(client, _Contract(), "p")
    frame = {"observation.state": np.zeros(2)}
    seq.retarget("old")
    assert seq.step(frame) is None and client.started.acquire(timeout=2.0)  # request 1 sent for "old"
    client.gate.set()  # the chunk lands between ticks: it is in _reply now
    _drain(seq)
    assert seq._reply is not None
    client.gate.clear()
    seq.retarget("new")  # before the next tick
    assert seq._reply is None and seq.queued == 0
    out = seq.step(frame)  # must ask again for "new", not play request 1's rows
    assert out is None and client.started.acquire(timeout=2.0)
    assert [r.observation.task for r in client.requests] == ["old", "new"]
    client.gate.set()
    _drain(seq)
    row = seq.step(frame)
    assert row is not None and row.tolist() == [2.0, 2.0]  # request 2's rows, booked under "new"
    assert [c["instruction"] for c in seq.calls] == ["new"]
    seq.close()


def test_a_retarget_cancels_the_request_in_flight_and_never_runs_two_infers_at_once():
    client = _Client()
    seq = Sequential(client, _Contract(), "p")
    frame = {"observation.state": np.zeros(2)}
    seq.retarget("old")
    assert seq.step(frame) is None and client.started.acquire(timeout=2.0)
    seq.retarget("new")  # while request 1 is still waiting on the server
    for _ in range(50):  # the next ticks hold until the superseded request has drained, then ask once for "new"
        if client.started.acquire(timeout=0.05):
            break
        assert seq.step(frame) is None
    else:
        pytest.fail("no request was sent for the new instruction")
    assert client.cancelled == 1 and client.max_inside == 1
    assert [r.observation.task for r in client.requests] == ["old", "new"]
    client.gate.set()
    _drain(seq)
    assert seq.step(frame).tolist() == [2.0, 2.0] and seq.calls[-1]["instruction"] == "new"  # no PolicyFault surfaced
    seq.close()
    assert seq.queued == 0


def test_close_drops_the_plan_under_the_lock():
    client = _Client()
    seq = Sequential(client, _Contract(), "p")
    seq.retarget("x")
    assert seq.step({"observation.state": np.zeros(2)}) is None and client.started.acquire(timeout=2.0)
    client.gate.set()
    _drain(seq)
    seq.close()
    assert seq._reply is None and seq.queued == 0 and seq.task is None


def test_a_lost_server_faults_once_the_received_plan_is_played_not_after_the_action_timeout():
    """LeRobot's engine notices a dead server by action starvation; a sequential loop holds as a matter of course, so
    it must read the liveliness latch itself: finish the rows it has, then fault at once (not 5 s later)."""
    from strands_robots.policy.base import PolicyFault

    client = _Client()
    seq = Sequential(client, _Contract(), "p")
    frame = {"observation.state": np.zeros(2)}
    seq.retarget("x")
    assert seq.step(frame) is None and client.started.acquire(timeout=2.0)
    client.gate.set()
    _drain(seq)
    assert seq.step(frame) is not None  # row 1 of 3 arrived
    client.present = False  # the server leaves while two rows are still queued
    assert seq.step(frame) is not None and seq.step(frame) is not None  # the local buffer is played to its end
    with pytest.raises(PolicyFault, match="server presence lost"):
        seq.step(frame)  # queue empty, nobody to ask: fault NOW, no request is sent
    assert len(client.requests) == 1


def test_a_request_in_flight_ends_when_the_server_leaves():
    from strands_robots.policy.base import PolicyFault

    client = _Client()
    seq = Sequential(client, _Contract(), "p")
    frame = {"observation.state": np.zeros(2)}
    seq.retarget("x")
    assert seq.step(frame) is None and client.started.acquire(timeout=2.0)
    client.present = False  # mid-request: the cancelled= hook ends the wait instead of action_timeout_s
    _drain(seq)
    assert client.cancelled == 1
    with pytest.raises(PolicyFault, match="server presence lost"):
        seq.step(frame)


def test_a_reply_that_lands_between_the_two_reads_of_a_tick_is_never_asked_for_twice():
    """The worker stores its reply and exits in the gap between step()'s two reads. Liveness must be read first: then
    the tick sees the request still in flight and holds; the next tick takes the reply. Read the other way round, the
    tick sees no reply AND no request and asks a second time for the same frame (two chunks from one pose)."""
    client = _Client()
    seq = Sequential(client, _Contract(), "p")
    frame = {"observation.state": np.zeros(2)}
    seq.retarget("x")
    assert seq.step(frame) is None and client.started.acquire(timeout=2.0)  # request 1 in flight
    worker = seq._inflight
    inner = seq._lock

    class LetTheWorkerFinishOnRelease:
        """The loop thread's lock: when it is released the worker is let through and joined, so whatever the loop
        read under the lock is stale by the time it runs its next statement (the reviewer's interleaving)."""

        def __enter__(self):
            inner.acquire()

        def __exit__(self, *exc):
            inner.release()
            if threading.current_thread() is not worker and worker.is_alive():
                client.gate.set()
                worker.join(2.0)
                assert not worker.is_alive()

    seq._lock = LetTheWorkerFinishOnRelease()  # the worker also stores through seq._lock: same inner lock
    out = seq.step(frame)  # tick 2: the reply lands between the two reads
    assert out is None and len(client.requests) == 1, "a second request was sent for the same still frame"
    seq._lock = inner
    row = seq.step(frame)  # tick 3: the reply is taken, the plan plays
    assert row is not None and len(seq.calls) == 1 and seq.queued == 2 and len(client.requests) == 1
