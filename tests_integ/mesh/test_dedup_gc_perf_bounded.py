"""Issue #231 regression: bound _CommandDeduplicator GC cost under sustained
pressure on two axes -- total wall-clock and max GC lock-hold time.

These pin the perf rework against the pre-fix implementation, which (a) ran the
sort-and-slice on every call once the cache hovered at the cap (no hysteresis
band) and (b) held self._lock for the entire O(n log n) sort. Both axes are
asserted with generous absolute bounds so the test is a coarse regression guard,
not a flaky tight-loop benchmark.
"""

import threading
import time

from strands_robots.mesh.transport.bridge_transport import (
    _MAX_DEDUP_ENTRIES_HARD,
    _CommandDeduplicator,
)


def _make_payload(i: int) -> dict:
    return {"sender_id": "s", "turn_id": str(i), "command": {"op": i}}


def _fill_to_hard_cap(dedup: _CommandDeduplicator) -> None:
    # Push the cache past the hard boundary so GC is armed.
    for i in range(_MAX_DEDUP_ENTRIES_HARD + 200):
        dedup.is_duplicate("topic", _make_payload(i))


def test_sustained_pressure_total_wall_clock_is_bounded():
    """Axis (a): N further is_duplicate() calls past the cap stay well under a
    coarse wall-clock ceiling. The pre-fix sort-and-slice-every-call path blows
    past this once the cache sits at the cap."""
    dedup = _CommandDeduplicator(ttl_s=3600.0, strict=True)
    _fill_to_hard_cap(dedup)

    n_calls = 2000
    start = time.monotonic()
    for i in range(n_calls):
        dedup.is_duplicate("topic", _make_payload(_MAX_DEDUP_ENTRIES_HARD + 1000 + i))
    elapsed = time.monotonic() - start

    # Hysteresis means most of these calls do no heap-select work at all.
    # 2000 calls in well under 2s even on a loaded CI box.
    assert elapsed < 2.0, f"sustained pressure too slow: {elapsed:.3f}s for {n_calls} calls"


def test_gc_max_lock_hold_does_not_serialise_concurrent_caller():
    """Axis (b): while one thread drives the cache past the hard cap (triggering
    GC), a contender thread timing its own is_duplicate() call must not be
    blocked for the full GC compute. The pre-fix path held the lock across the
    whole sort, so the contender's max single-call latency spiked; the
    snapshot-then-apply pattern keeps it bounded."""
    dedup = _CommandDeduplicator(ttl_s=3600.0, strict=True)
    _fill_to_hard_cap(dedup)

    stop = threading.Event()
    max_latency = [0.0]

    def contender():
        i = 0
        while not stop.is_set():
            t0 = time.monotonic()
            dedup.is_duplicate("other", _make_payload(10_000_000 + i))
            dt = time.monotonic() - t0
            if dt > max_latency[0]:
                max_latency[0] = dt
            i += 1

    t = threading.Thread(target=contender, daemon=True)
    t.start()
    try:
        # Drive sustained GC pressure on the main topic.
        for i in range(3000):
            dedup.is_duplicate("topic", _make_payload(20_000_000 + i))
    finally:
        stop.set()
        t.join(timeout=5.0)

    # Single is_duplicate() call must never block for the full GC sweep.
    # Heap walk is outside the lock, so the contender's worst case is the
    # snapshot copy + eviction apply, not the O(n log k) compute.
    assert max_latency[0] < 0.05, (
        f"contender single-call latency too high: {max_latency[0] * 1000:.1f}ms "
        "(GC likely holding the lock across the heap walk)"
    )


def test_dedup_correctness_unchanged_after_gc():
    """The snapshot-then-apply GC must not break dedup identity semantics:
    a freshly inserted entry is still reported duplicate on immediate repeat."""
    dedup = _CommandDeduplicator(ttl_s=3600.0, strict=True)
    _fill_to_hard_cap(dedup)

    payload = _make_payload(99_999_999)
    assert dedup.is_duplicate("topic", payload) is False
    assert dedup.is_duplicate("topic", payload) is True
