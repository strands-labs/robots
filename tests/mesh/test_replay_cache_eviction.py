"""Unit tests for :func:`strands_robots.mesh.core._evict_replay_cache`.

The helper is a single source of truth for the eviction strategy used by
both ``_estop_replay_cache`` and ``_resume_replay_cache``. End-to-end
behaviour through the handlers is already covered by
``test_estop_replay.py`` / ``test_resume_replay.py`` /
``test_replay_cache_monotonic.py``; this file pins the eviction
contract at the unit level so a refactor of either caller cannot
silently change the semantics.

Pin coverage:

- below cap is a no-op (cache untouched).
- TTL purge drops only stale entries.
- when the cache is full of in-window (fresh) entries, drop oldest 20
  percent (drop>=1, ordered by stored timestamp).
- generic over key type: works for ``float`` keys (estop shape) and
  ``tuple[str, str]`` keys (resume shape).

Pre-fix verification: the helper is new in the prior fix. The pre-the prior fix code
duplicates the eviction body inline; without this helper the duplicate
implementations could drift (e.g. one caller could silently revert to
``time.time()`` for the cutoff, or change the drop fraction). These
tests anchor the contract so any future divergence between the helper
and a hand-inlined re-implementation surfaces here, not in production.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh.core import _evict_replay_cache


def test_below_cap_keeps_fresh_entries() -> None:
    """If the cache is below ``max_size`` AND entries are fresh, no-op (issue #274)."""
    # Fresh entries: ts=180, 181 with cutoff=200-60=140 -> all fresh
    cache: dict[float, float] = {1.0: 180.0, 2.0: 181.0}
    _evict_replay_cache(cache, max_size=10, ttl_s=60.0, now_mono=200.0)
    assert cache == {1.0: 180.0, 2.0: 181.0}


def test_below_cap_still_purges_stale_entries() -> None:
    """Issue #274: TTL purge MUST run even when below max_size to prevent
    indefinite accumulation of stale entries on low-traffic meshes."""
    # ts=100, 101 with cutoff=200-60=140 -> both stale
    cache: dict[float, float] = {1.0: 100.0, 2.0: 101.0}
    _evict_replay_cache(cache, max_size=10, ttl_s=60.0, now_mono=200.0)
    # Stale entries dropped despite below-cap
    assert cache == {}


def test_at_cap_with_stale_entries_drops_only_stale() -> None:
    """At cap, stale entries (ts < now_mono - ttl_s) are dropped; fresh stay."""
    cache: dict[float, float] = {
        1.0: 50.0,  # stale (ts=50, cutoff=140)
        2.0: 60.0,  # stale
        3.0: 150.0,  # fresh
        4.0: 160.0,  # fresh
    }
    _evict_replay_cache(cache, max_size=4, ttl_s=60.0, now_mono=200.0)
    assert cache == {3.0: 150.0, 4.0: 160.0}


def test_at_cap_all_fresh_drops_oldest_twenty_percent() -> None:
    """If the cache is full of in-window entries, drop oldest 20%
    (rounded down, but at least one entry)."""
    cache: dict[float, float] = {float(i): 100.0 + i for i in range(10)}
    # cap=10, all 10 entries fresh (ttl very large).
    _evict_replay_cache(cache, max_size=10, ttl_s=1000.0, now_mono=200.0)
    # 10 // 5 == 2 entries dropped (the two oldest by stored ts).
    assert len(cache) == 8
    # Oldest two (stored ts 100.0 and 101.0) should be gone.
    assert 0.0 not in cache
    assert 1.0 not in cache
    # All fresher entries retained.
    for i in range(2, 10):
        assert float(i) in cache


def test_drop_at_least_one_when_small_full_cache() -> None:
    """``len // 5`` rounds down -- a cap-3 fresh cache must still drop one."""
    cache: dict[float, float] = {1.0: 100.0, 2.0: 101.0, 3.0: 102.0}
    _evict_replay_cache(cache, max_size=3, ttl_s=1000.0, now_mono=200.0)
    # 3 // 5 == 0 but max(1, 0) == 1 -- oldest one dropped.
    assert len(cache) == 2
    assert 1.0 not in cache  # the oldest by stored ts


def test_works_with_tuple_keys() -> None:
    """Resume cache uses ``tuple[str, str]`` keys; helper must be generic."""
    cache: dict[tuple[str, str], float] = {
        ("issuer-a", "nonce-1"): 50.0,  # stale
        ("issuer-b", "nonce-2"): 150.0,  # fresh
    }
    _evict_replay_cache(cache, max_size=2, ttl_s=60.0, now_mono=200.0)
    assert cache == {("issuer-b", "nonce-2"): 150.0}


def test_ttl_purge_runs_before_lru_drop() -> None:
    """If the TTL purge brings the cache under ``max_size``, the LRU
    drop branch must NOT run -- otherwise an in-window entry could be
    evicted unnecessarily."""
    cache: dict[float, float] = {
        1.0: 50.0,  # stale (will be purged)
        2.0: 150.0,  # fresh -- must survive
    }
    _evict_replay_cache(cache, max_size=2, ttl_s=60.0, now_mono=200.0)
    # After TTL purge cache has 1 entry (< max_size=2), LRU branch skipped.
    assert cache == {2.0: 150.0}


def test_empty_cache_is_noop() -> None:
    """Empty cache must not raise."""
    cache: dict[float, float] = {}
    _evict_replay_cache(cache, max_size=10, ttl_s=60.0, now_mono=200.0)
    assert cache == {}


@pytest.mark.parametrize("size", [100, 1000, 4096])
def test_cap_values_used_in_production(size: int) -> None:
    """Spot-check eviction at cap sizes the production code uses."""
    cache: dict[float, float] = {float(i): 100.0 + i for i in range(size)}
    _evict_replay_cache(cache, max_size=size, ttl_s=1000.0, now_mono=200.0)
    # Drop max(1, size // 5) entries.
    expected_remaining = size - max(1, size // 5)
    assert len(cache) == expected_remaining
