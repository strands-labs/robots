"""Regression pins for issue #273: lockout state mutation must stay inside
``_estop_replay_lock`` so concurrent estops from distinct issuers cannot race
the check-then-set.

The fix (Option A in #273) moved the ``_estop_lockout.set()`` /
``_last_estop_ts`` / ``_last_estop_mono`` mutation inside the existing
``with self._estop_replay_lock:`` block and emits the audit event from a
snapshot (``lockout_was_engaged``) taken under the lock. These tests pin
that invariant against future refactors:

1. Two-thread race: the lock serializes the check-then-set so the audit
   trail records exactly one ``remote_estop_engaged`` and one
   ``remote_estop_redundant`` event under a forced interleave.
2. Source-text pin: the ``_estop_lockout.set()`` mutation lives inside a
   ``with self._estop_replay_lock:`` block (the reliable anti-refactor
   guard, since a true data-race is non-deterministic under the GIL).
3. Timestamp atomicity: ``_last_estop_ts`` and ``_last_estop_mono`` are
   written together in the same lock-guarded engage branch so a later
   corroboration check never observes a mixed (one-updated, one-stale)
   pair.
"""

import inspect
import json
import threading
import time
from types import SimpleNamespace

from strands_robots.mesh import audit as audit_mod
from strands_robots.mesh import core


def _stub_mesh() -> core.Mesh:
    """Minimal Mesh stub for safety-handler testing (no network I/O)."""
    m = core.Mesh.__new__(core.Mesh)
    m.peer_id = "test-peer"
    m._estop_replay_cache = {}
    m._resume_replay_cache = {}
    m._estop_replay_lock = threading.Lock()
    m._resume_replay_lock = threading.Lock()
    m._estop_lockout = threading.Event()
    m._last_estop_ts = 0.0
    m._last_estop_mono = 0.0
    # publish_safety_event is gated on self._running; flip it on without
    # calling start() (which does network I/O).
    m._running = True
    m.publish = lambda key, payload: None  # type: ignore[method-assign]
    return m


def _envelope(t: float, peer_id: str) -> SimpleNamespace:
    body = {"peer_id": peer_id, "t": t, "type": "estop"}
    raw = json.dumps(body).encode()
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))


def _reset_audit_state() -> None:
    if hasattr(audit_mod, "_AUDIT_STATE"):
        audit_mod._AUDIT_STATE.psk_fingerprint = None
        audit_mod._AUDIT_STATE.seq_loaded = False
    if hasattr(audit_mod, "_SEQ_COUNTERS"):
        audit_mod._SEQ_COUNTERS.clear()


class TestEstopLockoutRace:
    def test_concurrent_distinct_issuer_estops_emit_one_engage_one_redundant(self, tmp_path, monkeypatch):
        """Two threads fire estops from distinct issuers; the lock must
        serialize the check-then-set so the audit trail records exactly one
        ``remote_estop_engaged`` and one ``remote_estop_redundant``.

        To make the race deterministic rather than timing-dependent, the
        first thread to read the lockout state is paused at that point while
        a second reader is invited to rendezvous. If the engage block were
        OUTSIDE the lock, both threads would read ``is_set() == False`` and
        emit two ``remote_estop_engaged`` events. Because the fix keeps the
        check-then-set INSIDE ``_estop_replay_lock``, the second thread
        cannot reach the read until the first releases the lock, so the
        forced-interleave barrier times out harmlessly and the second thread
        correctly observes the lockout already engaged.
        """
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
        _reset_audit_state()

        m = _stub_mesh()

        real_is_set = m._estop_lockout.is_set
        # Barrier of 2 with a short timeout: if both threads can read the
        # lockout state concurrently (engage block outside the lock) the
        # barrier trips and the race window is exercised. With the fix the
        # second thread is blocked on the lock, the barrier times out, and
        # only the first thread proceeds to engage.
        interleave = threading.Barrier(2, timeout=1.0)
        first_read = threading.Event()

        def instrumented_is_set():
            result = real_is_set()
            if not result and not first_read.is_set():
                # First reader to see "not engaged": try to rendezvous with a
                # concurrent reader. This only succeeds if the read happens
                # outside the lock (the buggy path).
                first_read.set()
                try:
                    interleave.wait()
                except threading.BrokenBarrierError:
                    # Expected under the fix: the second reader is blocked on
                    # _estop_replay_lock and never reaches this rendezvous, so
                    # the 1.0s barrier times out and trips BrokenBarrierError.
                    # That timeout IS the pass condition -- swallow it so the
                    # winning thread proceeds to engage the lockout normally.
                    pass
            return result

        m._estop_lockout.is_set = instrumented_is_set  # type: ignore[method-assign]

        base_t = time.time()
        start_barrier = threading.Barrier(2)

        def fire(peer_id: str, t: float) -> None:
            start_barrier.wait()
            m._on_safety_estop(_envelope(t, peer_id))

        t1 = threading.Thread(target=fire, args=("op-1", base_t))
        t2 = threading.Thread(target=fire, args=("op-2", base_t + 0.5))
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not t1.is_alive() and not t2.is_alive(), "estop handler threads deadlocked"
        assert m._estop_lockout.is_set()

        records = audit_mod.read_audit_log()
        events = [r["event"] for r in records]
        engaged = events.count("remote_estop_engaged")
        redundant = events.count("remote_estop_redundant")

        assert engaged == 1, (
            f"expected exactly one remote_estop_engaged, got {engaged}: {events}. "
            "Two engages means the lockout check-then-set raced outside the lock (issue #273)."
        )
        assert redundant == 1, (
            f"expected exactly one remote_estop_redundant, got {redundant}: {events}. "
            "The losing concurrent estop must still be preserved on the forensic record."
        )


class TestEstopLockoutLockContainment:
    def test_lockout_mutation_is_inside_replay_lock(self):
        """Source-text pin: the ``_estop_lockout.set()`` mutation must appear
        inside a ``with self._estop_replay_lock:`` block. Guards against a
        future refactor moving the engage block back out of the lock.
        """
        source = inspect.getsource(core.Mesh._on_safety_estop)
        lines = source.split("\n")

        with_idx = None
        with_indent = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("with ") and "_estop_replay_lock" in stripped:
                with_idx = i
                with_indent = len(line) - len(line.lstrip())
                break

        assert with_idx is not None, (
            "no `with self._estop_replay_lock:` block found in _on_safety_estop -- "
            "issue #273 fix relies on it guarding the lockout mutation."
        )

        # Find the `_estop_lockout.set()` mutation and assert no dedent back
        # to/under the `with` indent happens before reaching it (i.e. it is
        # still inside the block).
        set_idx = None
        for i in range(with_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= with_indent:
                # dedented out of the with-block before reaching the set()
                break
            if "self._estop_lockout.set()" in line:
                set_idx = i
                break

        assert set_idx is not None, (
            "self._estop_lockout.set() is not inside the `with self._estop_replay_lock:` "
            "block in _on_safety_estop -- concurrent-estop race regression (issue #273)."
        )

    def test_timestamps_mutated_inside_lock_with_set(self):
        """Source-text pin: ``_last_estop_ts`` and ``_last_estop_mono`` are
        both written in the same engage branch as the lockout set(), so a
        later corroboration check cannot observe a mixed pair.
        """
        source = inspect.getsource(core.Mesh._on_safety_estop)
        lines = source.split("\n")

        set_idx = next(
            (i for i, line in enumerate(lines) if "self._estop_lockout.set()" in line),
            None,
        )
        assert set_idx is not None, "engage branch missing _estop_lockout.set()"

        set_indent = len(lines[set_idx]) - len(lines[set_idx].lstrip())
        # The two timestamp writes must be siblings of set() (same branch,
        # same indent) within a small window of following lines.
        window = "\n".join(lines[set_idx : set_idx + 4])
        assert "self._last_estop_ts = time.time()" in window, (
            "_last_estop_ts not written alongside the lockout set() -- timestamp atomicity regression (issue #273)."
        )
        assert "self._last_estop_mono = time.monotonic()" in window, (
            "_last_estop_mono not written alongside the lockout set() -- timestamp atomicity regression (issue #273)."
        )
        for needle in (
            "self._last_estop_ts = time.time()",
            "self._last_estop_mono = time.monotonic()",
        ):
            line = next(line for line in lines if needle in line)
            indent = len(line) - len(line.lstrip())
            assert indent == set_indent, (
                f"{needle!r} indent {indent} != set() indent {set_indent}; the timestamp "
                "writes must be in the same lock-guarded engage branch (issue #273)."
            )
