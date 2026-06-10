"""Pin test for corroboration-window clock domain.

The corroboration branch in ``_on_safety_estop`` compares the time since
the last estop lockout against a 0.2s window. This MUST use
``time.monotonic()`` (the same domain as every other local bookkeeping
timestamp in the safety subsystem), NOT ``time.time()`` (wall-clock).

A wall-clock backward step (NTP correction, VM resume from suspend)
can make ``time.time() - _last_estop_ts`` go *negative*, which still
passes ``< 0.2`` -- acceptable (conservative direction). But a forward
step (NTP step forward, VM wakeup after 5s sleep) can push the
difference past 0.2s for two estops that arrived within 0.2s of
monotonic time, demoting a legitimate cross-session corroboration to
``estop_replay_rejected`` (severity ``warning``).

Pre-fix code used ``time.time() - self._last_estop_ts < 0.2``.
Fixed code uses ``time.monotonic() - self._last_estop_mono < 0.2``.

Pin: monkeypatch ``time.time`` to simulate a forward NTP step
(+5s jump after first estop), assert the corroboration branch still
fires (because monotonic did not jump). If the fix regresses, the
second estop will be classified as ``estop_replay_rejected`` instead of
``estop_corroborated``.
"""

from __future__ import annotations

import inspect
import json
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest import mock

from strands_robots.mesh import core


def _stub_mesh() -> core.Mesh:
    """Minimal Mesh stub sufficient to exercise _on_safety_estop."""
    m = core.Mesh.__new__(core.Mesh)
    m.peer_id = "test-peer"
    m._estop_replay_cache = {}
    m._resume_replay_cache = {}
    m._estop_replay_lock = threading.Lock()
    m._resume_replay_lock = threading.Lock()
    m._estop_lockout = threading.Event()
    m._last_estop_ts = 0.0
    m._last_estop_mono = 0.0
    m.publish_safety_event = mock.MagicMock()  # type: ignore[method-assign]
    return m


def _envelope(t: float, peer_id: str = "issuer-A", source_zid: str | None = None) -> Any:
    """Build a minimal Zenoh sample stub for _on_safety_estop.

    Body includes ``source_zid`` to match the wire-level zid binding
    introduced in PR-225 (rejects mismatched wire-vs-body zid).
    """
    body: dict[str, Any] = {"peer_id": peer_id, "t": t}
    if source_zid is not None:
        body["source_zid"] = source_zid
    raw = json.dumps(body).encode()
    sample = SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))
    # Wire source_zid via the SourceInfo attribute that the handler extracts.
    if source_zid is not None:
        sample.source_info = SimpleNamespace(source_id=SimpleNamespace(zid=lambda s=source_zid: s))
    else:
        sample.source_info = None
    return sample


def test_corroboration_window_uses_monotonic_not_wall_clock() -> None:
    """Pin: corroboration window check MUST use monotonic clock.

    Scenario: first estop arrives at monotonic T=100.0, wall-clock
    W=1000000.0. NTP then jumps wall-clock forward by 5s (W becomes
    1000005.0). A second estop from a DIFFERENT wire_zid arrives
    at monotonic T=100.1 (within 0.2s) but wall-clock is now 1000005.1
    (5.1s after first -- would exceed 0.2s window under wall-clock).

    Expected: corroboration is detected because monotonic elapsed
    (0.1s) is < 0.2s. If the code uses wall-clock, the elapsed would
    be 5.1s and corroboration would be missed.
    """
    m = _stub_mesh()

    # Simulate first estop arriving: sets lockout + timestamps.
    first_t = 1000000.0
    mono_at_first = 100.0

    with mock.patch("time.time", return_value=first_t):
        with mock.patch("time.monotonic", return_value=mono_at_first):
            # Directly set state as if _on_safety_estop processed the
            # first envelope successfully (no cache hit = fresh slot).
            m._estop_lockout.set()
            m._last_estop_ts = time.time()
            m._last_estop_mono = time.monotonic()
            # Seed the cache with the first envelope's entry.
            wire_zid_a = "zid-session-A"
            m._estop_replay_cache[first_t] = ("issuer-A", mono_at_first, wire_zid_a)

    # Now simulate NTP step forward: wall-clock jumps +5s, monotonic
    # advances only 0.1s.
    wall_after_ntp = first_t + 5.1  # 5.1s later in wall-clock
    mono_after = mono_at_first + 0.1  # only 0.1s later in monotonic

    # Second estop from a DIFFERENT wire_zid, same `t` value (the
    # replay-vs-corroboration distinction is on wire_zid, not `t`).
    wire_zid_b = "zid-session-B"

    # Build envelope with same `t` so it hits the cache-slot branch.
    envelope = _envelope(t=first_t, peer_id="issuer-B", source_zid=wire_zid_b)

    with mock.patch("time.time", return_value=wall_after_ntp):
        with mock.patch("time.monotonic", return_value=mono_after):
            # Patch the MODULE-LEVEL _extract_sample_source_zid
            # (called from core.py at the wire-zid extract site).
            # Patching it on the instance with create=True was a no-op
            # because core.py calls the module-level function.
            with mock.patch.object(
                core,
                "_extract_sample_source_zid",
                return_value=wire_zid_b,
            ):
                m._on_safety_estop(envelope)

    # Assert: the corroboration branch fired (not replay_rejected).
    # Look for estop_corroborated event_type in the publish calls.
    calls = m.publish_safety_event.call_args_list  # type: ignore[attr-defined]
    event_types = [c.kwargs.get("event_type") for c in calls if c.kwargs]
    assert "estop_corroborated" in event_types, (
        f"Expected estop_corroborated but got events: {event_types}. "
        "This means the corroboration window is using wall-clock (time.time) "
        "instead of monotonic clock -- the NTP jump caused a false rejection."
    )


def test_corroboration_window_does_not_use_wall_clock_source() -> None:
    """Structural pin: verify the source code at the corroboration check
    uses ``_last_estop_mono`` not ``_last_estop_ts``."""
    source = inspect.getsource(core.Mesh._on_safety_estop)
    # The corroboration window MUST reference _last_estop_mono.
    assert "_last_estop_mono" in source, (
        "Corroboration window in _on_safety_estop does not reference _last_estop_mono -- clock domain regression."
    )
    # It must NOT use _last_estop_ts for the 0.2s window comparison.
    # The _last_estop_ts is still used for audit payloads and logging,
    # but the < 0.2 comparison must be on monotonic.
    lines = source.split("\n")
    for line in lines:
        if "< 0.2" in line or "<0.2" in line:
            assert "_last_estop_mono" in line, f"Found '< 0.2' comparison using wall-clock: {line.strip()}"
            assert "_last_estop_ts" not in line, f"The 0.2s window check must not use _last_estop_ts: {line.strip()}"
            break
    else:
        # The 0.2 literal might be extracted to a constant; check monotonic
        # is at least referenced in a comparison context.
        pass


def test_last_estop_mono_initialized_at_init() -> None:
    """Pin: _last_estop_mono MUST be initialized alongside _last_estop_ts."""
    source = inspect.getsource(core.Mesh.__init__)
    assert "_last_estop_mono" in source, (
        "_last_estop_mono not initialized in Mesh.__init__ -- new code path will hit AttributeError on first estop."
    )
