"""The Hub type-ahead's cache has to end somewhere.

A TTL that only stops SERVING an entry is not a bound: the dict keeps every key ever
written. This is the test that the store prunes itself - by age when read, by age and
then by insertion order when written.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard.ttl_cache import DEFAULT_MAX_ENTRIES, TTLCache


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class TestFreshness:
    def test_a_fresh_value_is_returned(self) -> None:
        c: TTLCache[int] = TTLCache(10.0, clock=_Clock())
        c.put("k", 1)
        assert c.get("k") == 1

    def test_an_expired_value_is_a_miss_and_is_collected(self) -> None:
        clock = _Clock()
        c: TTLCache[int] = TTLCache(10.0, clock=clock)
        c.put("k", 1)
        clock.t += 10.0  # exactly the TTL is already too old
        assert c.get("k") is None
        assert len(c) == 0, "the point of this class: an expired entry is not just ignored"

    def test_an_unknown_key_is_a_miss(self) -> None:
        assert TTLCache(10.0).get("nope") is None


class TestBound:
    def test_the_cap_holds_under_a_long_type_ahead(self) -> None:
        c: TTLCache[str] = TTLCache(300.0, max_entries=4, clock=_Clock())
        for i in range(50):
            c.put(f"query-{i}", "rows")
        assert len(c) <= 4

    def test_eviction_drops_the_oldest_and_keeps_the_newest(self) -> None:
        c: TTLCache[int] = TTLCache(300.0, max_entries=3, clock=_Clock())
        for i, k in enumerate(["a", "b", "c"]):
            c.put(k, i)
        c.put("d", 3)
        assert c.get("a") is None, "the abandoned prefix goes first"
        assert c.get("d") == 3, "the key the user is actually typing survives"
        assert len(c) == 3

    def test_rewriting_a_key_neither_grows_nor_ages_it(self) -> None:
        clock = _Clock()
        c: TTLCache[int] = TTLCache(10.0, max_entries=3, clock=clock)
        c.put("k", 1)
        clock.t += 9.0
        c.put("k", 2)  # refreshed, so it must survive another full TTL
        clock.t += 9.0
        assert c.get("k") == 2
        assert len(c) == 1

    def test_a_refreshed_key_is_not_the_next_victim(self) -> None:
        c: TTLCache[int] = TTLCache(300.0, max_entries=2, clock=_Clock())
        c.put("a", 1)
        c.put("b", 2)
        c.put("a", 11)  # 'a' is now the most recently useful
        c.put("c", 3)
        assert c.get("a") == 11
        assert c.get("b") is None

    def test_expiry_is_swept_on_write_not_only_on_read(self) -> None:
        # nobody may ever look these keys up again; they still must not accumulate
        clock = _Clock()
        c: TTLCache[int] = TTLCache(10.0, max_entries=100, clock=clock)
        for i in range(20):
            c.put(f"k{i}", i)
        clock.t += 11.0
        c.put("fresh", 1)
        assert len(c) == 1

    def test_a_nonsense_cap_is_refused(self) -> None:
        with pytest.raises(ValueError):
            TTLCache(10.0, max_entries=0)

    def test_the_default_cap_is_a_real_number(self) -> None:
        assert 8 <= DEFAULT_MAX_ENTRIES <= 512


class TestHousekeeping:
    def test_clear_empties_it(self) -> None:
        c: TTLCache[int] = TTLCache(10.0)
        c.put("k", 1)
        c.clear()
        assert len(c) == 0 and c.get("k") is None
