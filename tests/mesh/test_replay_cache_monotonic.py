"""Pin test for replay-cache TTL eviction must use time.monotonic(). a wall-clock backward step (NTP correction, VM resume)
must NOT leave cache entries un-evictable; a forward step must NOT age
fresh entries out early. The cache is local-only bookkeeping; envelope
freshness still uses wall-clock because it compares the issuer's
wall-clock-stamped ``t``.

Pin: drives the actual handler ``_on_safety_estop`` / ``_on_safety_resume``,
asserts the resulting cache value is in the time.monotonic() domain,
NOT in the time.time() domain. Pre-fix HEAD stored time.time() so the
gap between the cached value and time.monotonic() at test time is
typically large (epoch seconds vs process-uptime seconds).
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest import mock

from strands_robots.mesh import core


def _stub_mesh() -> core.Mesh:
    """Minimal Mesh stub sufficient to exercise the safety handlers."""
    m = core.Mesh.__new__(core.Mesh)
    m.peer_id = "test-peer"
    m._estop_replay_cache = {}
    m._resume_replay_cache = {}
    import threading

    m._estop_replay_lock = threading.Lock()
    m._resume_replay_lock = threading.Lock()
    m._estop_lockout = threading.Event()
    m._last_estop_ts = 0.0
    # publish_safety_event is best-effort and called on accept; stub it.
    m.publish_safety_event = lambda **kwargs: None  # type: ignore[method-assign]
    return m


def _envelope(t: float, peer_id: str = "issuer-A", **extra) -> Any:
    body = {"peer_id": peer_id, "t": t, **extra}
    raw = json.dumps(body).encode()
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))


def test_estop_cache_value_is_monotonic_not_wall_clock() -> None:
    """After a remote estop is accepted the stored cache value must be
    time.monotonic()-derived (process-uptime seconds), NOT time.time()
    (epoch seconds). The two clock domains differ by ~1.7e9, so the
    pin is robust."""
    m = _stub_mesh()
    wall_now = time.time()
    m._on_safety_estop(_envelope(t=wall_now, peer_id="alice"))

    assert len(m._estop_replay_cache) == 1
    # cache value is now (issuer_id, mono_ts) tuple
    stored_value = next(iter(m._estop_replay_cache.values()))
    if isinstance(stored_value, tuple):
        stored_ts = stored_value[1]  # F9-A tuple shape
    else:
        stored_ts = stored_value  # legacy shape (resume cache, etc.)

    # Post-fix invariant: stored value is monotonic-derived.
    # On pre-fix HEAD this fails because time.time() (~1.7e9) is many
    # orders of magnitude larger than time.monotonic() (process uptime,
    # typically < 1e6).
    monotonic_now = time.monotonic()
    assert abs(stored_ts - monotonic_now) < 5.0, (
        f"cache value {stored_ts} is not in time.monotonic() domain "
        f"(monotonic_now={monotonic_now}, wall_now={time.time()}); "
        f"pre-R19 regression"
    )


def test_resume_cache_value_is_monotonic_not_wall_clock() -> None:
    """Resume cache mirrors estop -- cache value must be monotonic-derived."""
    import hashlib
    import hmac as hmac_mod
    import os

    m = _stub_mesh()
    # Resume needs a configured override code on the receiver.
    # HMAC binds (peer_id, t, lockout_elapsed_s, proof_nonce).
    import json as _json

    code = "test-override"
    proof_nonce = "n1"
    wall_now = time.time()
    lockout_elapsed = 1.0
    mac_input = _json.dumps(
        {
            "peer_id": "alice",
            "t": wall_now,
            "lockout_elapsed_s": lockout_elapsed,
            "proof_nonce": proof_nonce,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    proof = hmac_mod.new(code.encode(), mac_input, hashlib.sha256).hexdigest()

    with mock.patch.dict(os.environ, {"STRANDS_MESH_OVERRIDE_CODE": code}):
        m._on_safety_resume(
            _envelope(
                t=wall_now,
                peer_id="alice",
                lockout_elapsed_s=lockout_elapsed,
                proof_nonce=proof_nonce,
                override_proof=proof,
            )
        )

    assert len(m._resume_replay_cache) == 1
    stored_ts = next(iter(m._resume_replay_cache.values()))
    monotonic_now = time.monotonic()
    assert abs(stored_ts - monotonic_now) < 5.0, (
        f"cache value {stored_ts} is not in time.monotonic() domain "
        f"(monotonic_now={monotonic_now}, wall_now={time.time()}); "
        f"pre-R19 regression"
    )


# Re-export for ruff
from typing import Any  # noqa: E402
