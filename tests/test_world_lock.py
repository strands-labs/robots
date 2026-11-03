"""The world lock is fair: a waiting act() gets the world at the next release, not after a looping reader (#4814)."""

from __future__ import annotations

import signal
import threading
import time

import pytest

from strands_robots import Robot, Simulation
from strands_robots.simulation.lock import WorldLock


def test_acts_beside_a_reader_that_loops_on_the_lock_finish_promptly():
    with Simulation() as sim:
        arm = Robot("so101", sim=sim)
        stop, reads = threading.Event(), [0]

        def reader() -> None:
            while not stop.is_set():
                with sim.lock:
                    arm.observe(images=False)
                    sim.describe()
                reads[0] += 1

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        while not reads[0]:
            time.sleep(0.001)
        start = time.perf_counter()
        for _ in range(20):
            arm.act(arm.observe().joints)
        took = time.perf_counter() - start
        stop.set()
        t.join(2)
        assert took < 2.0, f"20 acts took {took:.2f} s beside a reader ({reads[0]} reads)"


def test_waiters_are_served_in_arrival_order():
    lock, order = WorldLock(), []
    lock.acquire()

    def take(i: int) -> None:
        with lock:
            order.append(i)

    threads = [threading.Thread(target=take, args=(i,)) for i in range(5)]
    for t in threads:  # each waiter is queued before the next one starts
        t.start()
        while len(lock._queue) <= threads.index(t):
            time.sleep(0.001)
    lock.release()
    for t in threads:
        t.join(2)
    assert order == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("case", "expect"),
    [
        ("owner re-enters", True),
        ("other thread, non-blocking", False),
        ("other thread, timeout", False),
    ],
)
def test_reentry_and_refusals(case, expect):
    lock, got = WorldLock(), []
    with lock, lock:
        if case == "owner re-enters":
            got.append(lock.acquire())
            lock.release()
        else:
            kw = {"blocking": False} if "non-blocking" in case else {"timeout": 0.05}
            t = threading.Thread(target=lambda: got.append(lock.acquire(**kw)))
            t.start()
            t.join(2)
    assert got == [expect]
    assert lock.acquire(blocking=False)  # fully released, and a timed-out waiter left no trace in the queue
    lock.release()


def test_release_by_a_thread_that_does_not_hold_it_raises():
    with pytest.raises(RuntimeError, match="un-acquired"):
        WorldLock().release()


def test_a_waiter_interrupted_mid_wait_leaves_the_queue():
    class Interrupted(Exception):
        pass

    def interrupt(*_: object) -> None:
        raise Interrupted

    lock, held, go, got = WorldLock(), threading.Event(), threading.Event(), []

    def holder() -> None:
        with lock:
            held.set()
            go.wait()

    threading.Thread(target=holder, daemon=True).start()
    held.wait()
    previous = signal.signal(signal.SIGUSR1, interrupt)
    try:
        threading.Timer(0.05, signal.pthread_kill, (threading.main_thread().ident, signal.SIGUSR1)).start()
        with pytest.raises(Interrupted):
            lock.acquire()
    finally:
        signal.signal(signal.SIGUSR1, previous)
    go.set()
    t = threading.Thread(target=lambda: got.append(lock.acquire(timeout=1.0)))
    t.start()
    t.join(2)
    assert got == [True] and not lock._queue
